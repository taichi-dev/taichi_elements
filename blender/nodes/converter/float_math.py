import bpy

from .. import base


def get_res_value(socket):
    node = socket.node
    out = node.outputs['Result']
    vals_1 = node.inputs['Float 1'].get_value()
    vals_2 = node.inputs['Float 2'].get_value()
    mode = node.mode
    res = []
    if type(vals_1) == list and type(vals_2) == float:
        val_2 = vals_2
        for val_1 in vals_1:
            if mode == 'ADD':
                res_val = val_1 + val_2
            elif mode == 'SUBTRACT':
                res_val = val_1 - val_2
            elif mode == 'MULTIPLY':
                res_val = val_1 * val_2
            elif mode == 'DIVIDE':
                res_val = val_1 / val_2
            res.append(res_val)
    elif type(vals_1) == float and type(vals_2) == list:
        val_1 = vals_1
        for val_2 in vals_2:
            if mode == 'ADD':
                res_val = val_1 + val_2
            elif mode == 'SUBTRACT':
                res_val = val_1 - val_2
            elif mode == 'MULTIPLY':
                res_val = val_1 * val_2
            elif mode == 'DIVIDE':
                res_val = val_1 / val_2
            res.append(res_val)
    elif type(vals_1) == list and type(vals_2) == list:
        for val_1, val_2 in zip(vals_1, vals_2):
            if mode == 'ADD':
                res_val = val_1 + val_2
            elif mode == 'SUBTRACT':
                res_val = val_1 - val_2
            elif mode == 'MULTIPLY':
                res_val = val_1 * val_2
            elif mode == 'DIVIDE':
                res_val = val_1 / val_2
            res.append(res_val)
    elif type(vals_1) == float and type(vals_2) == float:
        if mode == 'ADD':
            res_val = val_1 + val_2
        elif mode == 'SUBTRACT':
            res_val = val_1 - val_2
        elif mode == 'MULTIPLY':
            res_val = val_1 * val_2
        elif mode == 'DIVIDE':
            res_val = val_1 / val_2
        res.append(res_val)
    # scene
    scn = bpy.context.scene
    key = '{0}.{1}'.format(node.name, out.name)
    scn.elements_sockets[key] = res


class ElementsFloatMathNode(base.BaseNode):
    bl_idname = 'elements_float_math_node'
    bl_label = 'Float Math'

    category = base.CONVERTER
    get_value = {'Result': get_res_value, }
    items = [
        ('ADD', 'Add', ''),
        ('SUBTRACT', 'Subtract', ''),
        ('MULTIPLY', 'Multiply', ''),
        ('DIVIDE', 'Divide', '')
    ]
    mode: bpy.props.EnumProperty(items=items, name='Mode')

    def init(self, context):
        self.width = 200.0

        out = self.outputs.new('elements_float_socket', 'Result')
        out.text = 'Result'
        out.hide_value = True

        val_1 = self.inputs.new('elements_float_socket', 'Float 1')
        val_1.text = 'Float 1'

        val_2 = self.inputs.new('elements_float_socket', 'Float 2')
        val_2.text = 'Float 2'

    def draw_buttons(self, context, layout):
        layout.prop(self, 'mode')
