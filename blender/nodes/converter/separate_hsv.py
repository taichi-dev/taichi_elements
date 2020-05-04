import bpy

from .. import base


def get_out_value(socket, name):
    node = socket.node
    colors = node.inputs['Color'].get_value()
    component = node.outputs[name]
    res = []
    for color in colors:
        attr = name.lower()
        component_value = getattr(color, attr)
        res.append(component_value)
    scn = bpy.context.scene
    key = '{0}.{1}'.format(node.name, component.name)
    scn.elements_sockets[key] = res


def get_out_value_h(socket):
    get_out_value(socket, 'H')


def get_out_value_s(socket):
    get_out_value(socket, 'S')


def get_out_value_v(socket):
    get_out_value(socket, 'V')


class ElementsSeparateHSVNode(base.BaseNode):
    bl_idname = 'elements_separate_hsv_node'
    bl_label = 'Separate HSV'

    category = base.CONVERTER
    get_value = {
        'H': get_out_value_h,
        'S': get_out_value_s,
        'V': get_out_value_v
    }

    def create_output(self, name):
        output = self.outputs.new('elements_float_socket', name)
        output.text = name
        output.hide_value = True

    def init(self, context):
        # h, s, v outputs
        self.create_output('H')
        self.create_output('S')
        self.create_output('V')

        # input color
        color_in = self.inputs.new('elements_color_socket', 'Color')
        color_in.text = ''
