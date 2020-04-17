import bpy

from . import base


class ElementsEmitterNode(base.BaseNode):
    bl_idname = 'elements_emitter_node'
    bl_label = 'Emitter'

    required_nodes = {
        'Emit Frame': [
            'elements_integer_node',
        ],
        'Source Geometry': [
            'elements_source_object_node',
        ],
        'Material': [
            'elements_material_node',
        ],
        'Color': ['elements_color_node'],
        'Velocity': ['elements_vector_node', 'elements_combine_vector_node']
    }
    typ: bpy.props.StringProperty(default='EMITTER')
    category = base.SIMULATION_OBJECTS

    def init(self, context):
        out = self.outputs.new('elements_struct_socket', 'Emitter')
        out.text = 'Emitter'

        emit_frame = self.inputs.new('elements_integer_socket', 'Emit Frame')
        emit_frame.text = 'Emit Frame'

        src_geom = self.inputs.new('elements_struct_socket', 'Source Geometry')
        src_geom.text = 'Source Geometry'

        mat = self.inputs.new('elements_struct_socket', 'Material')
        mat.text = 'Material'

        color = self.inputs.new('elements_color_socket', 'Color')
        color.text = 'Color'

        vel = self.inputs.new('elements_vector_socket', 'Velocity')
        vel.text = 'Velocity'
