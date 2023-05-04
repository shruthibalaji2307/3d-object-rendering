import sys, os
import json
import bpy
import mathutils
import numpy as np
import argparse
import csv
import json
import numpy as np
import os
import shutil
import urllib
from subprocess import call
from multiprocessing import Pool
import zipfile

def getIdFromPath(path):
    """
    Return model id from path
    """
    parts = []
    (path, tail) = os.path.split(path)
    while path and tail:
        parts.append(tail)
        (path, tail) = os.path.split(path)
    parts.append(os.path.join(path, tail))
    res = list(map(os.path.normpath, parts))[::-1]
    res = [k for k in res if len(k) > 28]
    return res[0] if len(res) > 0 else ''

# Handles serialization of 1D numpy arrays to JSON
class NumpyAwareJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray) and obj.ndim == 1:
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def obj2stats(obj):
    """
    Computes statistics of OBJ vertices and returns as {num,min,max,centroid}
    """
    minVertex = np.array([np.Infinity, np.Infinity, np.Infinity])
    maxVertex = np.array([-np.Infinity, -np.Infinity, -np.Infinity])
    aggVertices = np.zeros(3)
    numVertices = 0
    with open(obj, 'r') as f:
        for line in f:
            if line.startswith('v '):
                v = np.fromstring(line[2:], sep=' ')
                aggVertices += v
                numVertices += 1
                minVertex = np.minimum(v, minVertex)
                maxVertex = np.maximum(v, maxVertex)
    centroid = aggVertices / numVertices
    info = {}
    info['id'] = getIdFromPath(obj)
    info['numVertices'] = numVertices
    info['min'] = minVertex
    info['max'] = maxVertex
    info['centroid'] = centroid
    return info

def normalizeOBJ(obj, out, stats=None):
    """
    Normalizes OBJ to be centered at origin and fit in unit cube
    """
    if os.path.isfile(out):
        return
    if not stats:
        stats = obj2stats(obj)
    diag = stats['max'] - stats['min']
    norm = 1 / np.linalg.norm(diag)
    c = stats['centroid']
    outmtl = obj + '.mtl'
    with open(obj, 'r') as f, open(out, 'w') as fo:
        for line in f:
            if line.startswith('v '):
                v = np.fromstring(line[2:], sep=' ')
                vNorm = (v - c) * norm
                vNormString = 'v %f %f %f\n' % (vNorm[0], vNorm[1], vNorm[2])
                fo.write(vNormString)
            elif line.startswith('mtllib '):
                fo.write('mtllib ' + os.path.basename(outmtl) + '\n')
            else:
                fo.write(line)
    outStats = open(os.path.splitext(out)[0] + '.json', 'w')
    j = json.dumps(stats, cls=NumpyAwareJSONEncoder)
    outStats.write(j + '\n')
    outStats.close()
    shutil.copy2(os.path.splitext(obj)[0] + '.mtl', outmtl)
    return stats

class BlenderRenderer:
    def __init__(self, args):
        self.obj = args.obj
        self.fp = bpy.path.abspath(f"//{args.results}")
        if not os.path.exists(self.fp):
            os.makedirs(self.fp)
        self.DEBUG = False
        self.VIEWS = args.views
        self.RESOLUTION = 800
        self.DEPTH_SCALE = 1.4
        self.COLOR_DEPTH = 8
        self.FORMAT = 'PNG'
        self.RANDOM_VIEWS = False
        self.UPPER_VIEWS = False
        self.CIRCLE_FIXED_START = (.3,0,0)

    def normalize(self):
        raw_obj_name = os.path.splitext(self.obj)[0]
        shutil.copy(raw_obj_name + '.mtl', raw_obj_name + '_normalized.mtl')
        self.obj_normalized = raw_obj_name + '_normalized.obj'
        normalizeOBJ(self.obj, self.obj_normalized)

    def listify_matrix(self, matrix):
        matrix_list = []
        for row in matrix:
            matrix_list.append(list(row))
        return matrix_list

    def setup_render(self):
        # Data to store in JSON file
        self.out_data = {
            'camera_angle_x': bpy.data.objects['Camera'].data.angle_x,
        }

        # Render Optimizations
        bpy.context.scene.render.use_persistent_data = True

        # Set up rendering of depth map.
        bpy.context.scene.use_nodes = True
        tree = bpy.context.scene.node_tree
        links = tree.links

        # Add passes for additionally dumping albedo and normals.
        #bpy.context.scene.view_layers["RenderLayer"].use_pass_normal = True
        bpy.context.scene.render.image_settings.file_format = str(self.FORMAT)
        bpy.context.scene.render.image_settings.color_depth = str(self.COLOR_DEPTH)

        if not self.DEBUG:
            # Create input render layer node.
            render_layers = tree.nodes.new('CompositorNodeRLayers')

            self.depth_file_output = tree.nodes.new(type="CompositorNodeOutputFile")
            self.depth_file_output.label = 'Depth Output'
            if self.FORMAT == 'OPEN_EXR':
                links.new(render_layers.outputs['Depth'], self.depth_file_output.inputs[0])
            else:
                # Remap as other types can not represent the full range of depth.
                map = tree.nodes.new(type="CompositorNodeMapValue")
                # Size is chosen kind of arbitrarily, try out until you're satisfied with resulting depth map.
                map.offset = [-0.7]
                map.size = [self.DEPTH_SCALE]
                map.use_min = True
                map.min = [0]
                links.new(render_layers.outputs['Depth'], map.inputs[0])

                links.new(map.outputs[0], self.depth_file_output.inputs[0])

            self.normal_file_output = tree.nodes.new(type="CompositorNodeOutputFile")
            self.normal_file_output.label = 'Normal Output'
            links.new(render_layers.outputs['Normal'], self.normal_file_output.inputs[0])

        # Background
        bpy.context.scene.render.dither_intensity = 0.0
        bpy.context.scene.render.film_transparent = True

        # Create collection for objects not to render with background

        # Delete default cube
        bpy.context.active_object.select_set(True)
        bpy.ops.object.delete()

        # Import textured mesh
        bpy.ops.object.select_all(action='DESELECT')

        bpy.ops.import_scene.obj(filepath=self.obj_normalized)

        obj = bpy.context.selected_objects[0]
        bpy.context.view_layer.objects.active = obj

        objs = [ob for ob in bpy.context.scene.objects if ob.type in ('EMPTY') and 'Empty' in ob.name]
        bpy.ops.object.delete({"selected_objects": objs})

    def parent_obj_to_camera(self, b_camera):
        origin = (0, 0, 0)
        b_empty = bpy.data.objects.new("Empty", None)
        b_empty.location = origin
        b_camera.parent = b_empty  # setup parenting

        scn = bpy.context.scene
        scn.collection.objects.link(b_empty)
        bpy.context.view_layer.objects.active = b_empty
        # scn.objects.active = b_empty
        return b_empty

    def render_360(self):
        scene = bpy.context.scene
        scene.render.resolution_x = self.RESOLUTION
        scene.render.resolution_y = self.RESOLUTION
        scene.render.resolution_percentage = 100

        cam = scene.objects['Camera']
        cam.location = (0, 4.0, 0.5)
        cam_constraint = cam.constraints.new(type='TRACK_TO')
        cam_constraint.track_axis = 'TRACK_NEGATIVE_Z'
        cam_constraint.up_axis = 'UP_Y'
        b_empty = self.parent_obj_to_camera(cam)
        cam_constraint.target = b_empty

        scene.render.image_settings.file_format = 'PNG'  # set output format to .png

        from math import radians

        stepsize = 360.0 / self.VIEWS
        rotation_mode = 'XYZ'

        if not self.DEBUG:
            for output_node in [self.depth_file_output, self.normal_file_output]:
                output_node.base_path = ''

        self.out_data['frames'] = []

        if not self.RANDOM_VIEWS:
            b_empty.rotation_euler = self.CIRCLE_FIXED_START

        for i in range(0, self.VIEWS):
            if self.DEBUG:
                i = np.random.randint(0,self.VIEWS)
                b_empty.rotation_euler[2] += radians(stepsize*i)
            if self.RANDOM_VIEWS:
                scene.render.filepath = self.fp + '/r_' + str(i)
                if self.UPPER_VIEWS:
                    rot = np.random.uniform(0, 1, size=3) * (1,0,2*np.pi)
                    rot[0] = np.abs(np.arccos(1 - 2 * rot[0]) - np.pi/2)
                    b_empty.rotation_euler = rot
                else:
                    b_empty.rotation_euler = np.random.uniform(0, 2*np.pi, size=3)
            else:
                print("Rotation {}, {}".format((stepsize * i), radians(stepsize * i)))
                scene.render.filepath = self.fp + '/r_' + str(i)

            # depth_file_output.file_slots[0].path = scene.render.filepath + "_depth_"
            # normal_file_output.file_slots[0].path = scene.render.filepath + "_normal_"

            if self.DEBUG:
                break
            else:
                bpy.ops.render.render(write_still=True)  # render still

            frame_data = {
                'file_path': scene.render.filepath,
                'rotation': radians(stepsize),
                'transform_matrix': self.listify_matrix(cam.matrix_world)
            }
            self.out_data['frames'].append(frame_data)

            if self.RANDOM_VIEWS:
                if self.UPPER_VIEWS:
                    rot = np.random.uniform(0, 1, size=3) * (1,0,2*np.pi)
                    rot[0] = np.abs(np.arccos(1 - 2 * rot[0]) - np.pi/2)
                    b_empty.rotation_euler = rot
                else:
                    b_empty.rotation_euler = np.random.uniform(0, 2*np.pi, size=3)
            else:
                b_empty.rotation_euler[2] += radians(stepsize)

        if not self.DEBUG:
            with open(self.fp + '/' + 'transforms.json', 'w') as out_file:
                json.dump(self.out_data, out_file, indent=4)

        bpy.ops.wm.quit_blender()


def main():
    parser = argparse.ArgumentParser(description='Normalized, aligns and renders 360 degree views of a 3D model.')
    parser.add_argument('obj', type=str, help='Path to the obj file.')
    parser.add_argument('--results', type=str, default="results", help='Path to the destination where rendered images and transforms.json should be saved.')
    parser.add_argument('--views', type=int, default=60, help='Number of views to be rendered.')
    argv = sys.argv[sys.argv.index("--") + 1:]
    args = parser.parse_args(argv)

    for dir in os.listdir('examples'):
        args.obj = os.path.join('examples', dir, dir+'.obj')
        args.results = os.path.join('dataset', dir)
        renderer = BlenderRenderer(args)
        renderer.normalize()
        renderer.setup_render()
        renderer.render_360()
    exit(0)

if __name__ == '__main__':
    main()