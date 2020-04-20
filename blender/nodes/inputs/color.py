import bpy

from .. import base


class ElementsColorNode(base.BaseNode):
    bl_idname = 'elements_color_node'
    bl_label = 'Color'

    category = base.INPUT

    def init(self, context):
        out = self.outputs.new('elements_color_socket', 'Color')
        out.text = ''

    def draw_buttons(self, context, layout):
        layout.template_color_picker(
            self.outputs['Color'],
            'default',
            value_slider=True
        )
