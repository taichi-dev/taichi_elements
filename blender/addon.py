import bpy

from . import tree
from . import categories
from . import nodes
from . import sockets
from . import operators
from . import handlers


addon_modules = [
    tree,
    sockets,
    nodes,
    categories,
    operators,
    handlers
]


def register():
    scn_type = bpy.types.Scene

    scn_type.elements_nodes = {}
    scn_type.elements_sockets = {}
    scn_type.elements_frame_start = bpy.props.IntProperty()
    scn_type.elements_frame_end = bpy.props.IntProperty()

    for addon_module in addon_modules:
        addon_module.register()

    bpy.types.NODE_HT_header.append(operators.op_draw_func)


def unregister():
    bpy.types.NODE_HT_header.remove(operators.op_draw_func)

    for addon_module in reversed(addon_modules):
        addon_module.unregister()

    scn_type = bpy.types.Scene

    del scn_type.elements_nodes
    del scn_type.elements_sockets
    del scn_type.elements_frame_end
    del scn_type.elements_frame_start
