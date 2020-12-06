import os

import bpy

from .. import base
from ... import particles_io


def get_cache(socket):
    node = socket.node
    pos = node.outputs['Position']
    vel = node.outputs['Velocity']
    col = node.outputs['Hex Color']
    mat = node.outputs['Material']
    size = node.outputs['Size']
    folder = node.inputs['Folder'].get_value()[0]
    pos_key = '{0}.{1}'.format(node.name, pos.name)
    vel_key = '{0}.{1}'.format(node.name, vel.name)
    col_key = '{0}.{1}'.format(node.name, col.name)
    mat_key = '{0}.{1}'.format(node.name, mat.name)
    size_key = '{0}.{1}'.format(node.name, size.name)
    # scene
    scn = bpy.context.scene
    if not folder:
        scn.elements_sockets[pos_key] = ()
        scn.elements_sockets[vel_key] = ()
        scn.elements_sockets[col_key] = ()
        scn.elements_sockets[mat_key] = ()
        scn.elements_sockets[size_key] = (1.0, )    # TODO
        return
    caches = {}
    # particles file name
    name = 'particles_{0:0>6}.bin'.format(scn.frame_current)
    # absolute particles file path
    path = bpy.path.abspath(os.path.join(folder, name))

    if os.path.exists(path):
        particles_io.read_pars(path, caches, folder, socket.name)

    else:
        scn.elements_sockets[pos_key] = ()
        scn.elements_sockets[vel_key] = ()
        scn.elements_sockets[col_key] = ()
        scn.elements_sockets[mat_key] = ()
        scn.elements_sockets[size_key] = (1.0, )    # TODO
        return

    scn.elements_sockets[pos_key] = caches[folder][particles_io.POS]
    scn.elements_sockets[vel_key] = caches[folder][particles_io.VEL]
    scn.elements_sockets[col_key] = caches[folder][particles_io.COL]
    scn.elements_sockets[mat_key] = caches[folder][particles_io.MAT]
    scn.elements_sockets[size_key] = (1.0, )    # TODO


class ElementsCacheNode(base.BaseNode):
    bl_idname = 'elements_cache_node'
    bl_label = 'Cache'

    required_nodes = {'Particles': ['elements_simulation_node', ], }

    get_value = {
        'Position': get_cache,
        'Velocity': get_cache,
        'Hex Color': get_cache,
        'Material': get_cache,
        'Size': get_cache
    }

    category = base.COMPONENT

    def init(self, context):
        self.width = 200.0

        pars = self.inputs.new('elements_struct_socket', 'Particles')
        pars.text = 'Particles'

        folder = self.inputs.new('elements_folder_socket', 'Folder')
        folder.text = 'Folder'

        # particle position
        pos = self.outputs.new('elements_vector_socket', 'Position')
        pos.text = 'Position'
        pos.hide_value = True

        # particle velocity
        vel = self.outputs.new('elements_vector_socket', 'Velocity')
        vel.text = 'Velocity'
        vel.hide_value = True

        # particle color
        col = self.outputs.new('elements_integer_socket', 'Hex Color')
        col.text = 'Hex Color'
        col.hide_value = True

        # particle material id
        mat = self.outputs.new('elements_integer_socket', 'Material')
        mat.text = 'Material'
        mat.hide_value = True

        # particle size
        size = self.outputs.new('elements_float_socket', 'Size')
        size.text = 'Size'
        size.hide_value = True
