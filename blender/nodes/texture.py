import bpy

from . import base


class ElementsTextureNode(base.BaseNode):
    bl_idname = 'elements_texture_node'
    bl_label = 'Texture'

    name: bpy.props.StringProperty()
    category = base.SOURCE_DATA

    def init(self, context):
        self.width = 250.0

        out = self.outputs.new('elements_struct_socket', 'Texture')
        out.text = 'Texture'

    def draw_buttons(self, context, layout):
        layout.prop_search(self, 'name', bpy.data, 'textures', text='')
