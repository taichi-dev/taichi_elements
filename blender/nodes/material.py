import bpy

from . import base


class ElementsMaterialNode(base.BaseNode):
    bl_idname = 'elements_material_node'
    bl_label = 'Material'

    items = [
        ('WATER', 'Water', ''),
        ('SNOW', 'Snow', ''),
        ('ELASTIC', 'Elastic', ''),
        ('SAND', 'Sand', '')
    ]
    typ: bpy.props.EnumProperty(items=items, default='WATER')
    category = base.SOLVERS

    def init(self, context):
        out = self.outputs.new('elements_struct_socket', 'Material')
        out.text = 'Material Settings'

    def draw_buttons(self, context, layout):
        layout.prop(self, 'typ', text='')
