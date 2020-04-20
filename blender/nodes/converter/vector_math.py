import bpy
import mathutils

from .. import base


def get_val(mode, val_1, val_2):
    val_1 = mathutils.Vector(val_1)
    val_2 = mathutils.Vector(val_2)
    out = mathutils.Vector()
    if mode == 'ADD':
        out = val_1 + val_2
    elif mode == 'SUBTRACT':
        out = val_1 - val_2
    elif mode == 'MULTIPLY':
        out[0] = val_1[0] * val_2[0]
        out[1] = val_1[1] * val_2[1]
        out[2] = val_1[2] * val_2[2]
    elif mode == 'DIVIDE':
        out[0] = val_1[0] / val_2[0]
        out[1] = val_1[1] / val_2[1]
        out[2] = val_1[2] / val_2[2]
    return out


def get_res_value(socket):
    node = socket.node
    out = node.outputs['Result']
    vals_1 = node.inputs['Vector 1'].get_value()
    vals_2 = node.inputs['Vector 2'].get_value()
    mode = node.mode
    res = []

    if len(vals_1) == 1 and len(vals_2) > 1:
        val_1 = vals_1[0]
        for val_2 in vals_2:
            r = get_val(mode, val_1, val_2)
            res.append(r)
    elif len(vals_1) > 1 and len(vals_2) == 1:
        val_2 = vals_2[0]
        for val_1 in vals_1:
            r = get_val(mode, val_1, val_2)
            res.append(r)
    elif len(vals_1) > 1 and len(vals_2) > 1:
        for val_1, val_2 in zip(vals_1, vals_2):
            r = get_val(mode, val_1, val_2)
            res.append(r)

    # scene
    scn = bpy.context.scene
    key = '{0}.{1}'.format(node.name, out.name)
    scn.elements_sockets[key] = res


class ElementsVectorMathNode(base.BaseNode):
    bl_idname = 'elements_vector_math_node'
    bl_label = 'Vector Math'

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

        out = self.outputs.new('elements_vector_socket', 'Result')
        out.text = 'Result'
        out.hide_value = True

        val_1 = self.inputs.new('elements_vector_socket', 'Vector 1')
        val_1.text = 'Vector 1'

        val_2 = self.inputs.new('elements_vector_socket', 'Vector 2')
        val_2.text = 'Vector 2'

    def draw_buttons(self, context, layout):
        layout.prop(self, 'mode', text='')
