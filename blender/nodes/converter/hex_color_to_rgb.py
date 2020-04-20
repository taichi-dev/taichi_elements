import bpy

from .. import base


def get_out_value(socket):
    node = socket.node
    out = node.outputs['Color']
    hex_cols = node.inputs['Hex Color'].get_value()
    # scene
    scn = bpy.context.scene
    key = '{0}.{1}'.format(node.name, out.name)
    res = []
    for hex_col in hex_cols:
        red = ((hex_col >> 16) & 0xFF) / 0xFF
        green = ((hex_col >> 8) & 0xFF) / 0xFF
        blue = (hex_col & 0xFF) / 0xFF
        res.append((red, green, blue))
    scn.elements_sockets[key] = res


class ElementsHexColorToRGBNode(base.BaseNode):
    bl_idname = 'elements_hex_color_to_rgb_node'
    bl_label = 'Hex Color to RGB'

    category = base.CONVERTER
    get_value = {'Color': get_out_value, }

    def init(self, context):
        self.width = 180.0

        out = self.outputs.new('elements_color_socket', 'Color')
        out.text = 'Color'
        out.hide_value = True

        hex_col = self.inputs.new('elements_integer_socket', 'Hex Color')
        hex_col.text = 'Hex Color'
