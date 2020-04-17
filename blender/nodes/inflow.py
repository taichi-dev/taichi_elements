import bpy

from . import base


class ElementsInflowNode(base.BaseNode):
    bl_idname = 'elements_inflow_node'
    bl_label = 'Inflow'

    required_nodes = {
        'Source Geometry': [
            'elements_source_object_node',
        ],
        'Material': [
            'elements_material_node',
        ]
    }
    typ: bpy.props.StringProperty(default='INFLOW')

    category = base.SIMULATION_OBJECTS

    def init(self, context):
        out = self.outputs.new('elements_struct_socket', 'Inflow')
        out.text = 'Inflow'

        enable = self.inputs.new('elements_float_socket', 'Enable')
        enable.text = 'Enable'

        src_geom = self.inputs.new('elements_struct_socket', 'Source Geometry')
        src_geom.text = 'Source Geometry'

        mat = self.inputs.new('elements_struct_socket', 'Material')
        mat.text = 'Material'

        color = self.inputs.new('elements_color_socket', 'Color')
        color.text = 'Color'

        vel = self.inputs.new('elements_vector_socket', 'Velocity')
        vel.text = 'Velocity'
