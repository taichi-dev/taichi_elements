import bpy

from . import base
from ..categories import STRUCT


class ElementsMergeNode(base.ElementsDynamicSocketsNode, base.BaseNode):
    bl_idname = 'elements_merge_node'
    bl_label = 'Merge'

    text: bpy.props.StringProperty(default='List')
    text_empty: bpy.props.StringProperty(default='Merge Lists')
    category = STRUCT
