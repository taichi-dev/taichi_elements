from . import base


class ElementsGravityNode(base.BaseNode):
    bl_idname = 'elements_gravity_node'
    bl_label = 'Gravity'

    required_nodes = {'Direction': ['elements_vector_node', ], }

    category = base.FORCE_FIELDS

    def init(self, context):
        self.width = 175.0

        out = self.outputs.new('elements_struct_socket', 'Gravity')
        out.text = 'Gravity Force'

        direction = self.inputs.new('elements_vector_socket', 'Direction')
        direction.text = 'Direction'
        direction.value = (0.0, 0.0, -9.81)
