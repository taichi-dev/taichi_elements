import bpy

from .. import base


def get_out_value(socket):
    node = socket.node
    out = node.outputs['Vector']
    node = socket.node
    x = node.inputs['X'].get_value()
    y = node.inputs['Y'].get_value()
    z = node.inputs['Z'].get_value()
    # scene
    scn = bpy.context.scene
    key = '{0}.{1}'.format(node.name, out.name)
    res = []
    for x_val, y_val, z_val in zip(x, y, z):
        res.append((x_val, y_val, z_val))
    scn.elements_sockets[key] = res


class ElementsCombineXYZNode(base.BaseNode):
    bl_idname = 'elements_combine_xyz_node'
    bl_label = 'Combine XYZ'

    category = base.CONVERTER
    get_value = {'Vector': get_out_value, }

    def init(self, context):
        self.width = 170.0

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
