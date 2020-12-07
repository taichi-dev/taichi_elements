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
    from . import addon

    def register():
        addon.register()

    def unregister():
        addon.unregister()

# Otherwise act as a PyPI package
