import os
import time
import shutil
import argparse

parser = argparse.ArgumentParser(
    description='Install the current Blender addon.')
parser.add_argument('-k', action='store_true', help='keep refreshing')

addon_path = os.environ['BLENDER_USER_ADDON_PATH']
if addon_path[:-1] == '/':
    addon_path = addon_path[:-1]
assert addon_path.endswith(os.path.join('scripts', 'addons'))

addon_folder = os.path.join(addon_path, 'taichi_elements')
blender_addon_path = 'blender'
blender_addon_main_file_name = '__init__.py'
engine_path = 'engine'


def install():
    print("Installing...")
    if os.path.exists(addon_folder):  # delete the old addon
        shutil.rmtree(addon_folder)

    os.mkdir(addon_folder)
    for source_directory in (blender_addon_path, engine_path):
        for f in os.listdir(source_directory):
            if f.endswith('.py'):
                source_file_path = os.path.join(source_directory, f)
                shutil.copy(source_file_path, os.path.join(addon_folder, f))
    # copy main blender addon file
    shutil.copy(
        os.path.join(blender_addon_path, blender_addon_main_file_name),
        os.path.join(addon_folder, blender_addon_main_file_name)
    )
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
