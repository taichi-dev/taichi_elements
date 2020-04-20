import bpy

from .. import base


def get_out_value(socket):
    node = socket.node
    out = node.outputs['Color']
    # input color
    in_col = node.inputs['Color'].get_value()
    # scene
    scn = bpy.context.scene
    key = '{0}.{1}'.format(node.name, out.name)
    # result
    res = []

    for col in in_col:
        r = 1.0 - col[0]
        g = 1.0 - col[1]
        b = 1.0 - col[2]
        res.append((r, g, b))

    scn.elements_sockets[key] = res


class ElementsInvertNode(base.BaseNode):
    bl_idname = 'elements_invert_node'
    bl_label = 'Invert'

    category = base.COLOR
    get_value = {'Color': get_out_value, }

    def init(self, context):
        self.width = 180.0

        out = self.outputs.new('elements_color_socket', 'Color')
        out.text = 'Color'
        out.hide_value = True

        col = self.inputs.new('elements_color_socket', 'Color')
        col.text = 'Color'
