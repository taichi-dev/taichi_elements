from . import base


def get_out_value(socket):
    node = socket.node
    out = node.outputs['Vector']
    col = node.inputs['Color'].get_value()
    out.value[0] = col[0]
    out.value[1] = col[1]
    out.value[2] = col[2]
    return out.value


class ElementsColorToVectorNode(base.BaseNode):
    bl_idname = 'elements_color_to_vector_node'
    bl_label = 'Color to Vector'

    category = base.CONVERTER
    get_value = {'Vector': get_out_value, }

    def init(self, context):
        out = self.outputs.new('elements_vector_socket', 'Vector')
        out.text = 'Vector'
        out.hide_value = True

        col = self.inputs.new('elements_color_socket', 'Color')
        col.text = 'Color'
