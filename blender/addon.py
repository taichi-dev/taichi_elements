import bpy

from . import tree, categories, nodes, sockets, operators, handlers


addon_modules = [tree, sockets, nodes, categories, operators, handlers]


def register():
    bpy.types.Scene.elements_nodes = {}
    bpy.types.Scene.elements_sockets = {}
    bpy.types.Scene.elements_frame_start = bpy.props.IntProperty()
    bpy.types.Scene.elements_frame_end = bpy.props.IntProperty()
    for addon_module in addon_modules:
        addon_module.register()
    bpy.types.NODE_HT_header.append(operators.op_draw_func)


def unregister():
    bpy.types.NODE_HT_header.remove(operators.op_draw_func)
    for addon_module in reversed(addon_modules):
        addon_module.unregister()
    del bpy.types.Scene.elements_nodes
    del bpy.types.Scene.elements_sockets
    del bpy.types.Scene.elements_frame_end
    del bpy.types.Scene.elements_frame_start
