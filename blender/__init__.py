bl_info = {
    'name': 'Taichi Elements',
    'description': 'High-Performance Multi-Material Continuum Physics Engine',
    'author': 'Taichi Elements Developers',
    'version': (0, 0, 0),
    'blender': (3, 4, 1),
    'location': 'Taichi Elements Window',
    'warning': 'Work in progress',
    'support': 'COMMUNITY',
    'wiki_url': 'https://taichi-elements.readthedocs.io/en/latest/',
    'tracker_url': 'https://github.com/taichi-dev/taichi_elements/issues',
    'category': 'Physics'
}


def register():
    from . import addon
    addon.register()


def unregister():
    from . import addon
    addon.unregister()
