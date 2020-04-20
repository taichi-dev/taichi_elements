import bpy

from .. import base


def get_out_value(socket):
    node = socket.node
    out = node.outputs['Vector']
    col = node.inputs['Color'].get_value()
    # scene
    scn = bpy.context.scene
    key = '{0}.{1}'.format(node.name, out.name)
    res = []
    for clr in col:
        res.append((clr[0], clr[1], clr[2]))
    scn.elements_sockets[key] = res


class ElementsColorToVectorNode(base.BaseNode):
    bl_idname = 'elements_color_to_vector_node'
    bl_label = 'Color to Vector'

    category = base.CONVERTER
    get_value = {'Vector': get_out_value, }

    def init(self, context):
        out = self.outputs.new('elements_vector_socket', 'Vector')
        out.text = 'Vector'
        out.hide_value = True

        col = self.inputs.new('elements_color_socket', 'Color')
        col.text = 'Color'
