import os
import time
import shutil
import argparse

parser = argparse.ArgumentParser(description='Install the current Blender addon.')
parser.add_argument('-k', action='store_true', help='keep refreshing')

addon_path = os.environ['BLENDER_USER_ADDON_PATH']
if addon_path[:-1] == '/':
  addon_path = addon_path[:-1]
assert addon_path.endswith('scripts\\addons')

addon_folder = os.path.join(addon_path, 'taichi_elements')

def install():
  print("Installing...")
  if os.path.exists(addon_folder): # delete the old addon
    shutil.rmtree(addon_folder)
    
  os.mkdir(addon_folder)
  for f in os.listdir('.'):
    if f.endswith('.py'):
      shutil.copy(f, os.path.join(addon_folder, f))
  print("Done.")
  
args = parser.parse_args()
if args.k:
  while True:
    time.sleep(1)
    install()
else:
  print(f"This will remove everything under {addon_folder}.")
  print("Are you sure? [y/N]")
  if input() != 'y':
    print("exiting")
    exit()
  else:
    install()