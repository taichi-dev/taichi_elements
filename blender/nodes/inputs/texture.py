import bpy

from .. import base


class ElementsTextureNode(base.BaseNode):
    bl_idname = 'elements_texture_node'
    bl_label = 'Texture'

    tex_name: bpy.props.StringProperty()
    category = base.INPUT

    def init(self, context):
        self.width = 220.0

        out = self.outputs.new('elements_color_socket', 'Texture')
        out.text = 'Texture'
        out.hide_value = True

    def draw_buttons(self, context, layout):
        layout.prop_search(self, 'tex_name', bpy.data, 'textures', text='')
