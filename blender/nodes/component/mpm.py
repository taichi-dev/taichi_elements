from .. import base


class ElementsMpmSolverNode(base.BaseNode):
    bl_idname = 'elements_mpm_solver_node'
    bl_label = 'MPM Solver'

    category = base.COMPONENT

    def init(self, context):
        self.width = 180.0

        out = self.outputs.new('elements_struct_socket', 'MPM Solver')
        out.text = 'Solver'

        res = self.inputs.new('elements_integer_socket', 'Resolution')
        res.text = 'Resolution'
        res.default = 32

        size = self.inputs.new('elements_float_socket', 'Size')
        size.text = 'Size'
        size.default = 1.0
