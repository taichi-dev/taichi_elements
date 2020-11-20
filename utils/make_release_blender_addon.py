import os
import sys
import zipfile

# taichi elements repository path
repo_path = os.path.dirname(os.path.abspath(os.curdir))
os.chdir(repo_path)
# blender addon folder name
BLEND_FOLDER = 'blender'
# engine folder name
ENGINE_FOLDER = 'engine'
# out addon folder name
OUT_FOLDER = 'taichi_elements'
# blender addon path
blend_path = os.path.join(repo_path, BLEND_FOLDER)
# taichi elements engine path
engine_path = os.path.join(repo_path, ENGINE_FOLDER)
sys.path.append(repo_path)

from blender import bl_info

addon_ver = bl_info['version']
# release zip file name
fname = 'taichi_elements-' + '.'.join(map(str, addon_ver)) + '.zip'
# release dir name
dname = 'utils'
# release path
rpath = os.path.join(dname, fname)
# compress type
cmprss = zipfile.ZIP_DEFLATED


def write_files(zip_file, out_folder):
    for root, _, files in os.walk('.'):
        for file in files:
            if not file.endswith('.py'):
                continue
            # input file path
            in_path = os.path.join(root, file)[2:]
            # output file path
            out_path = os.path.join(out_folder, in_path)
            zip_file.write(in_path, out_path, compress_type=cmprss)


with zipfile.ZipFile(rpath, 'w') as z:
    z.write('LICENSE', '{0}/LICENSE'.format(OUT_FOLDER), compress_type=cmprss)

    os.chdir(os.path.join(repo_path, BLEND_FOLDER))
    write_files(z, OUT_FOLDER)

    os.chdir(os.path.join(repo_path, ENGINE_FOLDER))
    write_files(z, os.path.join(OUT_FOLDER, ENGINE_FOLDER))

print('Done.')
