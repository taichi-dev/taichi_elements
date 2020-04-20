from . import base


def get_out_value(socket):
    node = socket.node
    out = node.outputs['Float']
    integer = node.inputs['Integer'].get_value()
    out.value = float(integer)
    return out.value


class ElementsIntToFloatNode(base.BaseNode):
    bl_idname = 'elements_int_to_float_node'
    bl_label = 'Integer to Float'

    category = base.CONVERTER
    get_value = {'Float': get_out_value, }

    def init(self, context):
        out = self.outputs.new('elements_float_socket', 'Float')
        out.text = 'Float'
        out.hide_value = True

        integer = self.inputs.new('elements_integer_socket', 'Integer')
        integer.text = 'Integer'
