import bpy

from . import tree, categories, nodes, sockets


addon_modules = [
    tree,
    sockets,
    nodes,
    categories
]


def register():
    for addon_module in addon_modules:
        addon_module.register()


def unregister():
    for addon_module in reversed(addon_modules):
        addon_module.unregister()
