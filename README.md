1. Download Blender 2.93 from here: https://download.blender.org/release/

2. Follow installation instructions here: https://docs.blender.org/manual/en/latest/getting_started/installing/linux.html#

3. Make sure to export the blender binary path to your env path variable.

4. Clone this repository.

5. Place object files and mtl files in the same format as placed in the examples folder.

6. Run the following, the script generates a dataset for ALL folders within examples!

```blender --python blender_renderer.py -- examples/house_0/house_0.obj --views 60```