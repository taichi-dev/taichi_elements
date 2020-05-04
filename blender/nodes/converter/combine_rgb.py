import bpy
import mathutils

from .. import base


def get_out_value(socket):
    node = socket.node
    out = node.outputs['Color']
    node = socket.node
    r = node.inputs['R'].get_value()
    g = node.inputs['G'].get_value()
    b = node.inputs['B'].get_value()
    key = '{0}.{1}'.format(node.name, out.name)
    res = []
    for r_val, g_val, b_val in zip(r, g, b):
        color = mathutils.Color((r_val, g_val, b_val))
        res.append(color)
    scn = bpy.context.scene
    scn.elements_sockets[key] = res


class ElementsCombineRGBNode(base.BaseNode):
    bl_idname = 'elements_combine_rgb_node'
    bl_label = 'Combine RGB'

    category = base.CONVERTER
    get_value = {'Color': get_out_value, }

    def create_input(self, name):
        inpt = self.inputs.new('elements_float_socket', name)
        inpt.text = name

    def init(self, context):
        self.width = 170.0

        out = self.outputs.new('elements_color_socket', 'Color')
        out.text = 'Color'
        out.hide_value = True

        # r, g, b components of color
        self.create_input('R')
        self.create_input('G')
        self.create_input('B')
