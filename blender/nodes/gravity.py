from . import base
from ..categories import FORCE_FIELDS


class ElementsGravityNode(base.BaseNode):
    bl_idname = 'elements_gravity_node'
    bl_label = 'Gravity'

    required_nodes = {
        'Speed': ['elements_float_node', 'elements_integer_node'],
        'Direction': [],
    }

    category = FORCE_FIELDS

    def init(self, context):
        self.width = 175.0

        out = self.outputs.new('elements_struct_socket', 'Gravity')
        out.text = 'Gravity Force'

        # speed = self.inputs.new('elements_float_socket', 'Speed')
        # speed.text = 'Speed'
        # speed.value = 0.0

        direction = self.inputs.new('elements_vector_socket', 'Direction')
        direction.text = 'Direction'
        direction.value = (0.0, 0.0, -9.81)
