from . import base


class ElementsVectorNode(base.BaseNode):
    bl_idname = 'elements_vector_node'
    bl_label = 'Vector'

    category = base.INPUTS

    def init(self, context):
        out = self.outputs.new('elements_vector_socket', 'Vector')
        out.text = ''
