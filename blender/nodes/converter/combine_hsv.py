import bpy
import mathutils

from .. import base


def get_out_value(socket):
    node = socket.node
    out = node.outputs['Color']
    node = socket.node
    h = node.inputs['H'].get_value()
    s = node.inputs['S'].get_value()
    v = node.inputs['V'].get_value()
    key = '{0}.{1}'.format(node.name, out.name)
    res = []
    for h_val, s_val, v_val in zip(h, s, v):
        color = mathutils.Color()
        color.v = v_val
        color.s = s_val
        color.h = h_val
        res.append(color)
    scn = bpy.context.scene
    scn.elements_sockets[key] = res


class ElementsCombineHSVNode(base.BaseNode):
    bl_idname = 'elements_combine_hsv_node'
    bl_label = 'Combine HSV'

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

        # h, s, v components of color
        self.create_input('H')
        self.create_input('S')
        self.create_input('V')
