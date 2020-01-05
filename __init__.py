# necessary for Blender to detect this addon
bl_info = {
    'name': 'Elements',
    'blender': (2, 81, 0),
    'category': 'Animation'
}


try:
    # If inside blender, act as an addon
    import bpy

    from . import addon


    def register():
        addon.register()


    def unregister():
        addon.unregister()
except:
    pass

# Otherwise act as a PyPI package
