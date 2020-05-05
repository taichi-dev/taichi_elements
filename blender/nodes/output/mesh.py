import bpy

from .. import base


def get_pos_value(socket):
    node = socket.node
    verts = node.inputs['Vertices']
    verts_key = '{0}.{1}'.format(node.name, verts.name)
    scn.elements_sockets[verts_key] = verts.get_value()


class ElementsMeshNode(base.BaseNode):
    bl_idname = 'elements_mesh_node'
    bl_label = 'Mesh'

    category = base.OUTPUT

    get_value = {
        'Position': get_pos_value,
    }

    required_nodes = {
        'Particles Object': [
            'elements_source_object_node',
        ],
    }

    def init(self, context):
        self.width = 180.0

        obj = self.inputs.new('elements_struct_socket', 'Mesh Object')
        obj.text = 'Mesh Object'

        verts = self.inputs.new('elements_vector_socket', 'Vertices')
        verts.text = 'Vertices'
        verts.hide_value = True
