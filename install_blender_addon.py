import os
import shutil

addon_path = os.environ['BLENDER_USER_ADDON_PATH']
if addon_path[:-1] == '/':
  addon_path = addon_path[:-1]
assert addon_path.endswith('scripts/addons')

addon_folder = os.path.join(addon_path, 'taichi_elements')

if os.path.exists(addon_folder): # delete the old addon
  shutil.rmtree(addon_folder)
  
os.mkdir(addon_folder)
for f in os.listdir('.'):
  if f.endswith('.py'):
    shutil.copy(f, os.path.join(addon_folder, f))
