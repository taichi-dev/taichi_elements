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
addon_engine_folder = os.path.join(addon_folder, 'engine')
taichi_elements_path = os.path.dirname(os.path.abspath(os.curdir))
blend_addon_path = os.path.join(taichi_elements_path, 'blender')
engine_path = os.path.join(taichi_elements_path, 'engine')
out_dirs = (addon_folder, addon_engine_folder)


def copy_file(src_dir, out_dir, f):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    source_file_path = os.path.join(src_dir, f)
    shutil.copy(source_file_path, os.path.join(out_dir, f))


def copy_files(src_dir, out_dir):
    for f in os.listdir(src_dir):
        if os.path.isdir(os.path.join(src_dir, f)):
            src_subdir = os.path.join(src_dir, f)
            out_subdir = os.path.join(out_dir, f)
            copy_files(src_subdir, out_subdir)
        elif f.endswith('.py'):
            copy_file(src_dir, out_dir, f)


def install():
    print("Installing...")
    if os.path.exists(addon_folder):  # delete the old addon
        shutil.rmtree(addon_folder)

    for dir_index, src_dir in enumerate((blend_addon_path, engine_path)):
        out_dir = out_dirs[dir_index]
        copy_files(src_dir, out_dir)

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
