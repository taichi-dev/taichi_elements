from . import base


class ElementsColorNode(base.BaseNode):
    bl_idname = 'elements_color_node'
    bl_label = 'Color'

    category = base.INPUTS

    def init(self, context):
        out = self.outputs.new('elements_color_socket', 'Color')
        out.text = ''
