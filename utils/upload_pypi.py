import os
import sys
import shutil
import platform

os.chdir(os.path.dirname(os.path.abspath(__file__)))

version = '0.0.1'

env_pypi_pwd = os.environ.get('PYPI_PWD', '')


def get_os_name():
    name = platform.platform()
    # in python 3.8, platform.platform() uses mac_ver() on macOS
    # it will return 'macOS-XXXX' instead of 'Darwin-XXXX'
    if name.lower().startswith('darwin') or name.lower().startswith('macos'):
        return 'osx'
    elif name.lower().startswith('windows'):
        return 'win'
    elif name.lower().startswith('linux'):
        return 'linux'
    assert False, "Unknown platform name %s" % name


if env_pypi_pwd == '':
    assert False, "Missing environment variable PYPI_PWD"


def get_python_executable():
    return '"' + sys.executable.replace('\\', '/') + '"'


with open('../setup.py') as fin:
    with open('setup.py', 'w') as fout:
        project_name = 'taichi_elements'
        print("project_name = '{}'".format(project_name), file=fout)
        print("version = '{}'".format(version), file=fout)
        for l in fin:
            print(l, file=fout, end='')

print("*** project_name = '{}'".format(project_name))

shutil.rmtree('./taichi_elements', ignore_errors=True)
shutil.rmtree('./build', ignore_errors=True)

os.makedirs('taichi_elements', exist_ok=True)

for f in os.listdir('../engine'):
    if f.endswith('.py'):
        shutil.copy(f'../engine/{f}', f'taichi_elements/{f}')

print("Using python executable", get_python_executable())

os.system('{} -m pip install --user --upgrade twine setuptools wheel'.format(
    get_python_executable()))

os.system('{} setup.py bdist_wheel'.format(get_python_executable()))

shutil.rmtree('./taichi_elements', ignore_errors=True)
shutil.rmtree('./build', ignore_errors=True)

os.system('{} -m twine upload dist/* --verbose -u yuanming-hu -p {}'.format(
    get_python_executable(),
    '%PYPI_PWD%' if get_os_name() == 'win' else '$PYPI_PWD'))
