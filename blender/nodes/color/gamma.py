import bpy

from .. import base


def get_out_value(socket):
    node = socket.node
    out = node.outputs['Color']
    # input color
    in_col = node.inputs['Color'].get_value()
    # gamma
    gmm = node.inputs['Gamma'].get_value()[0]
    # scene
    scn = bpy.context.scene
    key = '{0}.{1}'.format(node.name, out.name)
    # result
    res = []

    for col in in_col:
        r = pow(col[0], gmm)
        g = pow(col[1], gmm)
        b = pow(col[2], gmm)
        res.append((r, g, b))

    scn.elements_sockets[key] = res


class ElementsGammaNode(base.BaseNode):
    bl_idname = 'elements_gamma_node'
    bl_label = 'Gamma'

    category = base.COLOR
    get_value = {'Color': get_out_value, }

    def init(self, context):
        self.width = 200.0

        out = self.outputs.new('elements_color_socket', 'Color')
        out.text = 'Color'
        out.hide_value = True

        col = self.inputs.new('elements_color_socket', 'Color')
        col.text = 'Color'

        gamma = self.inputs.new('elements_float_socket', 'Gamma')
        gamma.text = 'Gamma'
        gamma.default = 1.0
