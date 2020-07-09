from .. import base


class ElementsHubNode(base.BaseNode):
    bl_idname = 'elements_hub_node'
    bl_label = 'Hub'

    required_nodes = {
        'Forces': [
            'elements_gravity_node', 'elements_make_list_node',
            'elements_merge_node'
        ],
        'Emitters': [
            'elements_emitter_node', 'elements_make_list_node',
            'elements_merge_node', 'elements_inflow_node'
        ],
        'Colliders': [
            'elements_ground_node', 'elements_make_list_node',
            'elements_merge_node'
        ],
    }

    category = base.COMPONENT

    def init(self, context):
        hub = self.outputs.new('elements_struct_socket', 'Hub Data')
        hub.text = 'Hub Data'

        forces = self.inputs.new('elements_struct_socket', 'Forces')
        forces.text = 'Forces'

        emitters = self.inputs.new('elements_struct_socket', 'Emitters')
        emitters.text = 'Emitters'

        colliders = self.inputs.new('elements_struct_socket', 'Colliders')
        colliders.text = 'Colliders'
