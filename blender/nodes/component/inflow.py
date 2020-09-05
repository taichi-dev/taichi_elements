import bpy

from .. import base


class ElementsInflowNode(base.BaseNode):
    bl_idname = 'elements_inflow_node'
    bl_label = 'Inflow'

    required_nodes = {
        'Source Object': [
            'elements_source_object_node',
        ],
        'Material': [
            'elements_material_node',
        ]
    }
    typ: bpy.props.StringProperty(default='INFLOW')

    category = base.COMPONENT

    def init(self, context):
        self.width = 200.0

        out = self.outputs.new('elements_struct_socket', 'Inflow')
        out.text = 'Inflow'

        enable = self.inputs.new('elements_float_socket', 'Enable')
        enable.text = 'Enable'
        enable.default = 1.0

        src_geom = self.inputs.new('elements_struct_socket', 'Source Object')
        src_geom.text = 'Source Object'

        mat = self.inputs.new('elements_struct_socket', 'Material')
        mat.text = 'Material'

        color = self.inputs.new('elements_color_socket', 'Color')
        color.text = 'Color'

        vel = self.inputs.new('elements_vector_socket', 'Velocity')
        vel.text = 'Velocity'
