import bpy

from .. import base


def mix_rgb(c1, c2, mode):
    if mode == 'ADD':
        res = (c1[0] + c2[0], c1[1] + c2[1], c1[2] + c2[2])
    elif mode == 'MULTIPLY':
        res = (c1[0] * c2[0], c1[1] * c2[1], c1[2] * c2[2])
    elif mode == 'SUBTRACT':
        res = (c1[0] - c2[0], c1[1] - c2[1], c1[2] - c2[2])
    elif mode == 'DIVIDE':
        res = [c1[0], c1[1], c1[2]]
        if c2[0] != 0.0:
            res[0] = c1[0] / c2[0]
        if c2[1] != 0.0:
            res[1] = c1[1] / c2[1]
        if c2[2] != 0.0:
            res[2] = c1[2] / c2[2]
    return res


def get_out_value(socket):
    node = socket.node
    out = node.outputs['Color']
    # input colors
    col1 = node.inputs['Color1'].get_value()
    col2 = node.inputs['Color2'].get_value()
    mode = node.mode
    # scene
    scn = bpy.context.scene
    key = '{0}.{1}'.format(node.name, out.name)
    # result
    res = []

    if len(col1) == len(col2):
        for c1, c2 in zip(col1, col2):
            # result value
            r_val = mix_rgb(c1, c2, mode)
            res.append(r_val)
    elif len(col1) == 1 and len(col2) > 1:
        c1 = col1[0]
        for c2 in col2:
            # result value
            r_val = mix_rgb(c1, c2, mode)
            res.append(r_val)
    elif len(col1) > 1 and len(col2) == 1:
        c2 = col2[0]
        for c1 in col1:
            # result value
            r_val = mix_rgb(c1, c2, mode)
            res.append(r_val)

    scn.elements_sockets[key] = res


class ElementsMixRGBNode(base.BaseNode):
    bl_idname = 'elements_mix_rgb_node'
    bl_label = 'Mix RGB'

    category = base.COLOR
    get_value = {'Color': get_out_value, }
    items = (
        ('ADD', 'Add', ''),
        ('MULTIPLY', 'Multiply', ''),
        ('SUBTRACT', 'Subtract', ''),
        ('DIVIDE', 'Divide', '')
    )
    mode: bpy.props.EnumProperty(name='Mode', items=items)

    def init(self, context):
        self.width = 160.0

        out = self.outputs.new('elements_color_socket', 'Color')
        out.text = 'Color'
        out.hide_value = True

        col1 = self.inputs.new('elements_color_socket', 'Color1')
        col1.text = 'Color1'

        col2 = self.inputs.new('elements_color_socket', 'Color2')
        col2.text = 'Color2'

    def draw_buttons(self, context, layout):
        layout.prop(self, 'mode', text='')
