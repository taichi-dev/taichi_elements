from . import base


def get_out_value_x(socket):
    node = socket.node
    vector = node.inputs['Vector'].get_value()
    x = node.outputs['X']
    x.value = vector[0]
    return x.value


def get_out_value_y(socket):
    node = socket.node
    vector = node.inputs['Vector'].get_value()
    y = node.outputs['Y']
    y.value = vector[1]
    return y.value


def get_out_value_z(socket):
    node = socket.node
    vector = node.inputs['Vector'].get_value()
    z = node.outputs['Z']
    z.value = vector[1]
    return z.value


class ElementsSeparateVectorNode(base.BaseNode):
    bl_idname = 'elements_separate_vector_node'
    bl_label = 'Separate Vector'

    category = base.CONVERTER
    get_value = {
        'X': get_out_value_x,
        'Y': get_out_value_y,
        'Z': get_out_value_z
    }

    def init(self, context):
        # x, y, z outputs
        x = self.outputs.new('elements_float_socket', 'X')
        x.text = 'X'
        x.hide_value = True

        y = self.outputs.new('elements_float_socket', 'Y')
        y.text = 'Y'
        y.hide_value = True

        z = self.outputs.new('elements_float_socket', 'Z')
        z.text = 'Z'
        z.hide_value = True

        # input vector
        vector_in = self.inputs.new('elements_vector_socket', 'Vector')
        vector_in.text = ''
