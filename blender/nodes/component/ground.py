import bpy

from .. import base


class ElementsGroundNode(base.BaseNode):
    bl_idname = 'elements_ground_node'
    bl_label = 'Ground'

    category = base.COMPONENT
    items = [
        ('0', 'Sticky', ''),
        ('1', 'Slip', ''),
        ('2', 'Separate', '')
    ]
    surface: bpy.props.EnumProperty(items=items, default='0')

    def init(self, context):
        self.width = 180.0

        out = self.outputs.new('elements_struct_socket', 'Ground')
        out.text = 'Ground'

        pos = self.inputs.new('elements_vector_socket', 'Position')
        pos.text = 'Position'

        direct = self.inputs.new('elements_vector_socket', 'Direction')
        direct.text = 'Direction'

    def draw_buttons(self, context, layout):
        layout.prop(self, 'surface', text='Surface')
