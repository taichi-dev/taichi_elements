from .. import base


class ElementsSimulationNode(base.BaseNode):
    bl_idname = 'elements_simulation_node'
    bl_label = 'Simulation'

    required_nodes = {
        'Solver': [
            'elements_mpm_solver_node',
        ],
        'Emitters': [
            'elements_emitter_node', 'elements_make_list_node',
            'elements_merge_node', 'elements_inflow_node'
        ],
        'Colliders': [
            'elements_ground_node', 'elements_make_list_node',
            'elements_merge_node'
        ],
        'Frame Start': [
            'elements_integer_node'
        ],
        'Frame End': [
            'elements_integer_node'
        ],
        'FPS': [
            'elements_integer_node'
        ]
    }

    category = base.COMPONENT

    def init(self, context):
        self.width = 200.0

        out = self.outputs.new('elements_struct_socket', 'Simulation Data')
        out.text = 'Particles'

        frame_start = self.inputs.new('elements_integer_socket', 'Frame Start')
        frame_start.text = 'Frame Start'
        frame_start.default = 0

        frame_end = self.inputs.new('elements_integer_socket', 'Frame End')
        frame_end.text = 'Frame End'
        frame_end.default = 50

        fps = self.inputs.new('elements_integer_socket', 'FPS')
        fps.text = 'FPS'
        fps.default = 30

        gravity = self.inputs.new('elements_vector_socket', 'Gravity')
        gravity.text = 'Gravity'
        gravity.default = (0.0, 0.0, -9.81)

        solver = self.inputs.new('elements_struct_socket', 'Solver')
        solver.text = 'Solver'

        emitters = self.inputs.new('elements_struct_socket', 'Emitters')
        emitters.text = 'Emitters'

        colliders = self.inputs.new('elements_struct_socket', 'Colliders')
        colliders.text = 'Colliders'
