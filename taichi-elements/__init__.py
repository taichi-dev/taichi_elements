bl_info = {
    'name': 'Elements',
    'blender': (2, 81, 0),
    'category': 'Animation'
}


from . import addon


def register():
    addon.register()


def unregister():
    addon.unregister()
