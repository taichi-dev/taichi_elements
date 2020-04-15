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
    create_psys: bpy.props.BoolProperty(default=True, name='Particles System')
    create_pmesh: bpy.props.BoolProperty(default=False, name='Particles Mesh')

    category = base.OUTPUT

    def init(self, context):
        self.width = 200.0

        pars = self.inputs.new('elements_struct_socket', 'Particles')
        pars.text = 'Particles'

        folder = self.inputs.new('elements_folder_socket', 'Folder')
        folder.text = 'Folder'

    def draw_buttons(self, context, layout):
        layout.label(text='Create:')
        layout.prop(self, 'create_psys', expand=True)
        layout.prop(self, 'create_pmesh', expand=True)
