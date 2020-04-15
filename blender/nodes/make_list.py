import bpy

from . import base
from ..categories import STRUCT


class ElementsMakeListNode(base.ElementsDynamicSocketsNode, base.BaseNode):
    bl_idname = 'elements_make_list_node'
    bl_label = 'Make List'

    text: bpy.props.StringProperty(default='Element')
    text_empty: bpy.props.StringProperty(default='Add Element')
    category = STRUCT
