import bpy

from . import base


class ElementsCacheNode(base.BaseNode):
    bl_idname = 'elements_cache_node'
    bl_label = 'Disk Cache'

    required_nodes = {
        'Particles': [
            'elements_simulation_node',
        ],
        'Folder': [
            'elements_folder_node',
        ],
    }

    category = base.OUTPUT

    def init(self, context):
        self.width = 200.0

        pars = self.inputs.new('elements_struct_socket', 'Particles')
        pars.text = 'Particles'

        folder = self.inputs.new('elements_folder_socket', 'Folder')
        folder.text = 'Folder'
