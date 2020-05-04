import bpy

from .. import base


# color component indices
indices = {
    'R': 0,
    'G': 1,
    'B': 2
}


def get_out_value(socket, name):
    node = socket.node
    colors = node.inputs['Color'].get_value()
    component = node.outputs[name]
    res = []
    index = indices[name]
    for color in colors:
        res.append(color[index])
    scn = bpy.context.scene
    key = '{0}.{1}'.format(node.name, component.name)
    scn.elements_sockets[key] = res


def get_out_value_r(socket):
    get_out_value(socket, 'R')


def get_out_value_g(socket):
    get_out_value(socket, 'G')


def get_out_value_b(socket):
    get_out_value(socket, 'B')


class ElementsSeparateRGBNode(base.BaseNode):
    bl_idname = 'elements_separate_rgb_node'
    bl_label = 'Separate RGB'

    category = base.CONVERTER
    get_value = {
        'R': get_out_value_r,
        'G': get_out_value_g,
        'B': get_out_value_b
    }

    def create_output(self, name):
        output = self.outputs.new('elements_float_socket', name)
        output.text = name
        output.hide_value = True

    def init(self, context):
        # r, g, b outputs
        self.create_output('R')
        self.create_output('G')
        self.create_output('B')

        # input color
        color_in = self.inputs.new('elements_color_socket', 'Color')
        color_in.text = ''
