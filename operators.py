import bpy
from .mpm_solver import MPMSolver
import taichi as ti
import numpy as np

from . import node_types

# just for debugging
taichi_gui = None

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
            if type(v) in (node_types.List, node_types.Merge):
                print(offset, i, type(v))
                offset += ' '
                for e in v.elements:
                    print(offset, i, type(e))
                    print_simulation_info(e, offset)
            elif type(v) in node_types.elements_types:
                print(offset, i, type(v))
                for key in dir(v):
                    if key[0] != '_':
                        print(offset, '  ', key, '=', getattr(v, key))
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
        
        # TODO: list is not implemented
        
        res = simulation_class.solver.resolution
        size = simulation_class.solver.size
        ti.reset()
        print(f"Creating simulation of res {res}, size {size}")
        sim = MPMSolver((res, res, res), size=size)
        
        hub = simulation_class.hubs
        assert len(hub.forces) == 1, "Only one gravity supported"
        force = hub.forces[0].output_folder
        gravity = force[0], force[1], force[2]
        print('g =', gravity)
        sim.set_gravity(gravity)
        
        emitters = hub.emitters
        for emitter in emitters:
            obj = emitter.bpy_object.bpy_object
            # Note: rotation is not supported
            center_x = obj.matrix_world[0][3]
            center_y = obj.matrix_world[1][3]
            center_z = obj.matrix_world[2][3]
            scale_x = obj.matrix_world[0][0]
            scale_y = obj.matrix_world[1][1]
            scale_z = obj.matrix_world[2][2]
            print(obj.matrix_world)
            material = emitter.material.material_type
            if material == 'WATER':
                taichi_material = MPMSolver.material_water
            elif material == 'ELASTIC':
                taichi_material = MPMSolver.material_elastic
            elif material == 'SNOW':
                taichi_material = MPMSolver.material_snow
            else:
                assert False, material
            lower = (center_x - scale_x, center_y - scale_y, center_z - scale_z)
            cube_size = (2 * scale_x, 2 * scale_y, 2 * scale_z)
            print(lower)
            print(cube_size)
            sim.add_cube(lower_corner=lower, cube_size=cube_size, material=taichi_material)

        global taichi_gui
        write_to_disk = False
        if taichi_gui is None:
          taichi_gui = ti.GUI("Blender - Taichi Elements ", res=512, background_color=0x112F41)

        for frame in range(500):
            sim.step(1e-2)
            colors = np.array([0x068587, 0xED553B, 0xEEEEF0], dtype=np.uint32)
            np_x, np_v, np_material = sim.particle_info()
            print(np_x)
            np_x /= size
            
            # simple camera transform
            screen_x = ((np_x[:, 0] + np_x[:, 1]) / 2 ** 0.5) - 0.2
            screen_y = (np_x[:, 2])
            
            screen_pos = np.stack([screen_x, screen_y], axis=-1)
            
            taichi_gui.circles(screen_pos, radius=1.5, color=colors[np_material])
            taichi_gui.show(f'{frame:06d}.png' if write_to_disk else None)

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
