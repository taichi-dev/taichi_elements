import bpy

from .. import base


def get_out_value(socket):
    node = socket.node
    out = node.outputs['Color']
    # input color
    in_col = node.inputs['Color'].get_value()
    # bright
    brg = node.inputs['Bright'].get_value()[0]
    # contrast
    cntr = node.inputs['Contrast'].get_value()[0]
    # scene
    scn = bpy.context.scene
    key = '{0}.{1}'.format(node.name, out.name)
    # result
    res = []

    for col in in_col:
        a = 1.0 + cntr
        b = brg - cntr * 0.5
        r = max(a * col[0] + b, 0.0)
        g = max(a * col[1] + b, 0.0)
        b = max(a * col[2] + b, 0.0)
        res.append((r, g, b))

    scn.elements_sockets[key] = res


class ElementsBrightContrastNode(base.BaseNode):
    bl_idname = 'elements_bright_contrast_node'
    bl_label = 'Bright/Contrast'

    category = base.COLOR
    get_value = {'Color': get_out_value, }

    def init(self, context):
        self.width = 170.0

        out = self.outputs.new('elements_color_socket', 'Color')
        out.text = 'Color'
        out.hide_value = True

        col = self.inputs.new('elements_color_socket', 'Color')
        col.text = 'Color'

        bright = self.inputs.new('elements_float_socket', 'Bright')
        bright.text = 'Bright'

        contrast = self.inputs.new('elements_float_socket', 'Contrast')
        contrast.text = 'Contrast'
