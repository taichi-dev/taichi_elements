# necessary for Blender to detect this addon
bl_info = {
    'name': 'Taichi Elements',
    'description': 'High-Performance Multi-Material Continuum Physics Engine',
    'author': 'Taichi Elements Developers',
    'version': (0, 0, 0),
    'blender': (2, 82, 0),
    'location': 'Taichi Elements Window',
    'warning': 'Work in progress',
    'support': 'COMMUNITY',
    'wiki_url': 'https://taichi-elements.readthedocs.io/en/latest/',
    'tracker_url': 'https://github.com/taichi-dev/taichi_elements/issues',
    'category': 'Physics'
}

use_blender = False

try:
    # If inside blender, act as an addon
    import bpy

    use_blender = True
except:
    pass

if use_blender:
    import sys
    import os

    bundle_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'bundle')

    if os.path.exists(bundle_path):
        def register():
            print('Found Taichi-Elements/bundle at', bundle_path)
            if bundle_path not in sys.path:
                sys.path.insert(0, bundle_path)
            # import addon after path is inserted, so that `import taichi` works
            from . import addon
            addon.register()

        def unregister():
            from . import addon
            addon.unregister()
            if bundle_path in sys.path:
                sys.path.remove(bundle_path)

    else:
        print('Cannot find Taichi-Elements/bundle, assuming PyPI installation.')

        from . import addon

        def register():
            addon.register()

        def unregister():
            addon.unregister()

# Otherwise act as a PyPI package
