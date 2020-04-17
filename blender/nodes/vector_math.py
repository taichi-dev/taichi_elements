import bpy
import mathutils

from . import base


def get_res_value(socket):
    node = socket.node
    out = node.outputs['Result']
    val_1 = mathutils.Vector(node.inputs['Vector 1'].get_value())
    val_2 = mathutils.Vector(node.inputs['Vector 2'].get_value())
    mode = node.mode
    if mode == 'ADD':
        out.value = val_1 + val_2
    elif mode == 'SUBTRACT':
        out.value = val_1 - val_2
    elif mode == 'MULTIPLY':
        out.value[0] = val_1[0] * val_2[0]
        out.value[1] = val_1[1] * val_2[1]
        out.value[2] = val_1[2] * val_2[2]
    elif mode == 'DIVIDE':
        out.value[0] = val_1[0] / val_2[0]
        out.value[1] = val_1[1] / val_2[1]
        out.value[2] = val_1[2] / val_2[2]
    print(out.value[0], out.value[1], out.value[2])
    return out.value


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
        layout.prop(self, 'mode')
