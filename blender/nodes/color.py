from . import base
from ..categories import INPUTS


class ElementsColorNode(base.BaseNode):
    bl_idname = 'elements_color_node'
    bl_label = 'Color'

    category = INPUTS

    def init(self, context):
        out = self.outputs.new('elements_color_socket', 'Color')
        out.text = ''
