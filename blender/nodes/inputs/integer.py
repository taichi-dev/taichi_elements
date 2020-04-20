from .. import base


class ElementsIntegerNode(base.BaseNode):
    bl_idname = 'elements_integer_node'
    bl_label = 'Integer'

    category = base.INPUT

    def init(self, context):
        out = self.outputs.new('elements_integer_socket', 'Integer')
        out.text = ''
