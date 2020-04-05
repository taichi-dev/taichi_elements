# necessary for Blender to detect this addon
bl_info = {'name': 'Elements', 'blender': (2, 81, 0), 'category': 'Animation'}

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
