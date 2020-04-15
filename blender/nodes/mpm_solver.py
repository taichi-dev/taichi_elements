from . import base
from ..categories import SOLVERS


class ElementsMpmSolverNode(base.BaseNode):
    bl_idname = 'elements_mpm_solver_node'
    bl_label = 'MPM Solver'

    required_nodes = {
        'Domain Object': [
            'elements_source_object_node',
        ],
        'Resolution': [
            'elements_integer_node',
        ],
        'Size': [
            'elements_float_node',
        ],
    }
    category = SOLVERS

    def init(self, context):
        self.width = 175.0

        out = self.outputs.new('elements_struct_socket', 'MPM Solver')
        out.text = 'Solver Settings'

        domain_obj = self.inputs.new('elements_struct_socket', 'Domain Object')
        domain_obj.text = 'Domain Object'

        res = self.inputs.new('elements_integer_socket', 'Resolution')
        res.text = 'Resolution'
        res.value = 64

        size = self.inputs.new('elements_float_socket', 'Size')
        size.text = 'Size'
        size.value = 10.0
