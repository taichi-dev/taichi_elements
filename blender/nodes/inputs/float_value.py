from .. import base


class ElementsFloatNode(base.BaseNode):
    bl_idname = 'elements_float_node'
    bl_label = 'Float'

    category = base.INPUT

    def init(self, context):
        out = self.outputs.new('elements_float_socket', 'Float')
        out.text = ''
