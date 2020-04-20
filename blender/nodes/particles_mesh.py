import bpy

from . import base


class ElementsParticlesMeshNode(base.BaseNode):
    bl_idname = 'elements_particles_mesh_node'
    bl_label = 'Particles Mesh'

    category = base.CREATE
    # mesh object name
    obj_name: bpy.props.StringProperty()

    def init(self, context):
        self.width = 250.0

        folder = self.inputs.new('elements_folder_socket', 'Folder')
        folder.text = 'Cache Folder'

    def draw_buttons(self, context, layout):
        layout.prop_search(self, 'obj_name', bpy.data, 'objects', text='')
