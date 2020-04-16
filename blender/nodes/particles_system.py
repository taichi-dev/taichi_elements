import bpy

from . import base


class ElementsParticlesSystemNode(base.BaseNode):
    bl_idname = 'elements_particles_system_node'
    bl_label = 'Particles System'

    category = base.IMPORT
    # particle system object name
    obj_name: bpy.props.StringProperty()

    def init(self, context):
        self.width = 250.0

        folder = self.inputs.new('elements_folder_socket', 'Folder')
        folder.text = 'Cache Folder'

    def draw_buttons(self, context, layout):
        layout.prop_search(self, 'obj_name', bpy.data, 'objects', text='')
