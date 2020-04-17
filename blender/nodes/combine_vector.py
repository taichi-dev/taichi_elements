from . import base


def get_out_value(socket):
    node = socket.node
    out = node.outputs['Vector']
    x = node.inputs['X'].get_value()
    y = node.inputs['Y'].get_value()
    z = node.inputs['Z'].get_value()
    out.value[0] = x
    out.value[1] = y
    out.value[2] = z
    return out.value


class ElementsCombineVectorNode(base.BaseNode):
    bl_idname = 'elements_combine_vector_node'
    bl_label = 'Combine Vector'

    category = base.CONVERTER
    get_value = {'Vector': get_out_value, }

    def init(self, context):
        out = self.outputs.new('elements_vector_socket', 'Vector')
        out.text = 'Vector'
        out.hide_value = True

        # x, y, z components of vector
        x = self.inputs.new('elements_float_socket', 'X')
        x.text = 'X'

        y = self.inputs.new('elements_float_socket', 'Y')
        y.text = 'Y'

        z = self.inputs.new('elements_float_socket', 'Z')
        z.text = 'Z'
