from .. import base


def get_out_value_x(socket):
    node = socket.node
    vectors = node.inputs['Vector'].get_value()
    x = node.outputs['X']
    # scene
    scn = bpy.context.scene
    key = '{0}.{1}'.format(node.name, x.name)
    res = []
    for vector in vectors:
        res.append(vector[0])
    scn.elements_sockets[key] = res


def get_out_value_y(socket):
    node = socket.node
    vectors = node.inputs['Vector'].get_value()
    y = node.outputs['Y']
    # scene
    scn = bpy.context.scene
    key = '{0}.{1}'.format(node.name, y.name)
    res = []
    for vector in vectors:
        res.append(vector[1])
    scn.elements_sockets[key] = res


def get_out_value_z(socket):
    node = socket.node
    vectors = node.inputs['Vector'].get_value()
    z = node.outputs['Z']
    # scene
    scn = bpy.context.scene
    key = '{0}.{1}'.format(node.name, z.name)
    res = []
    for vector in vectors:
        res.append(vector[2])
    scn.elements_sockets[key] = res


class ElementsSeparateXYZNode(base.BaseNode):
    bl_idname = 'elements_separate_xyz_node'
    bl_label = 'Separate XYZ'

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
