import bpy

from . import types


def get_simulation_nodes(operator, node_tree):
    simulation_nodes = []
    for node in node_tree.nodes:
        if node.bl_idname == 'elements_simulation_node':
            simulation_nodes.append(node)
    if len(simulation_nodes) != 1:
        operator.report(
            {'WARNING'},
            'The node tree must not contain more than 1 "Simulation" node.'
        )
        return
    else:
        return simulation_nodes[0]


def print_simulation_info(simulation_class, offset):
    offset += ' '
    for i in dir(simulation_class):
        v = getattr(simulation_class, i, None)
        if v and i[0] != '_':
            if type(v) in (types.List, types.Merge):
                print(offset, i, type(v))
                offset += ' '
                for e in v.elements:
                    print(offset, i, type(e))
                    print_simulation_info(e, offset)
            elif type(v) in types.elements_types:
                print(offset, i, type(v))
                print_simulation_info(v, offset)


class ELEMENTS_OT_SimulateParticles(bpy.types.Operator):
    bl_idname = "elements.simulate_particles"
    bl_label = "Simulate"

    def execute(self, context):
        self.node_tree = context.space_data.node_tree
        simulation_node = get_simulation_nodes(self, self.node_tree)
        if not simulation_node:
            return {'FINISHED'}

        simulation_class = simulation_node.get_class()

        print(79 * '=')
        print('simulation_class', type(simulation_class))
        print_simulation_info(simulation_class, '')
        print(79 * '=')

        return {'FINISHED'}


operator_classes = [
    ELEMENTS_OT_SimulateParticles,
]


def register():
    for operator_class in operator_classes:
        bpy.utils.register_class(operator_class)


def unregister():
    for operator_class in reversed(operator_classes):
        bpy.utils.unregister_class(operator_class)
