import bpy

from . import tree, categories, nodes, sockets, operators


addon_modules = [
    tree,
    sockets,
    nodes,
    categories,
    operators
]


def register():
    for addon_module in addon_modules:
        addon_module.register()


def unregister():
    for addon_module in reversed(addon_modules):
        addon_module.unregister()
