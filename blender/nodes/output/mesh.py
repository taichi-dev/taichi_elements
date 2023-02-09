import bpy

from .. import base


def get_pos_value(socket):
    node = socket.node

    verts = node.inputs['Vertices']
    verts_key = '{0}.{1}'.format(node.name, verts.name)
    scn.elements_sockets[verts_key] = verts.get_value()

    vels = node.inputs['Velocity']
    vels_key = '{0}.{1}'.format(node.name, vels.name)
    scn.elements_sockets[vels_key] = vels.get_value()

    emit = node.inputs['Emitters']
    emit_key = '{0}.{1}'.format(node.name, emit.name)
    scn.elements_sockets[emit_key] = emit.get_value()


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

        verts = self.inputs.new('elements_vector_socket', 'Velocity')
        verts.text = 'Velocity'
        verts.hide_value = True

        emitters = self.inputs.new('elements_integer_socket', 'Emitters')
        emitters.text = 'Emitters'
        emitters.hide_value = True
