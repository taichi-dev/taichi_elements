import threading, struct, os

import bpy
from .mpm_solver import MPMSolver
import taichi as ti
import numpy as np

from . import node_types


def get_cache_folder(simulation_node):
    particles_socket = simulation_node.outputs['Simulation Data']
    if particles_socket.is_linked:
        for link in particles_socket.links:
            disk_cache_node = link.to_node
            folder = disk_cache_node.inputs['Folder'].get_value()
            folder = bpy.path.abspath(folder)
            return folder


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

    def __init__(self):
        self.timer = None
        self.thread = None
        self.is_runnig = False
        self.is_finishing = False
        self.event_type = 'DEFAULT'

    def run_simulation(self):
        for frame in range(100):
            if self.event_type == 'ESC':
                print('STOP SIMULATION')
                self.thread = None
                self.is_finishing = True
                self.cancel(bpy.context)
                return
            self.sim.step(1e-2)
            np_x, np_v, np_material = self.sim.particle_info()
            print(np_x)

            if not os.path.exists(self.cache_folder):
                os.makedirs(self.cache_folder)

            particles_file_path = os.path.join(
                self.cache_folder,
                'particles_{0:0>6}.bin'.format(frame)
            )
            data = bytearray()
            particles_count = len(np_x)
            data.extend(struct.pack('I', particles_count))
            print(particles_count)
            for particle_index in range(particles_count):
                data.extend(struct.pack('3f', *np_x[particle_index]))
                data.extend(struct.pack('3f', *np_v[particle_index]))

            with open(particles_file_path, 'wb') as file:
                file.write(data)

    def init_simulation(self):
        self.is_runnig = True
        self.scene.elements_nodes.clear()
        simulation_node = get_simulation_nodes(self, self.node_tree)
        if not simulation_node:
            return {'FINISHED'}

        simulation_node.get_class()
        self.cache_folder = get_cache_folder(simulation_node)

        if not self.cache_folder:
            self.report(
                {'WARNING'},
                'Cache folder not specified'
            )
            return {'FINISHED'}

        for i, j in self.scene.elements_nodes.items():
            print(i, j)

        simulation_class = self.scene.elements_nodes[simulation_node.name]

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
            source_geometry = emitter.source_geometry
            if not source_geometry:
                continue
            obj = emitter.source_geometry.bpy_object
            if not obj:
                continue
            # Note: rotation is not supported
            center_x = obj.matrix_world[0][3]
            center_y = obj.matrix_world[1][3]
            center_z = obj.matrix_world[2][3]
            scale_x = obj.matrix_world[0][0]
            scale_y = obj.matrix_world[1][1]
            scale_z = obj.matrix_world[2][2]
            if not emitter.material:
                continue
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

        self.size = size
        self.sim = sim
        self.run_simulation()

    def launch_simulation(self):
        self.thread = threading.Thread(
                target=self.init_simulation, 
                args=()
        )
        self.thread.start()

    def modal(self, context, event):
        if event.type == 'ESC':
            self.event_type = 'ESC'

        if not self.is_runnig:
            self.launch_simulation()

        if self.is_finishing:
            self.cancel(context)
            return {'FINISHED'}

        return {'PASS_THROUGH'}

    def execute(self, context):
        self.node_tree = context.space_data.node_tree
        self.scene = context.scene
        context.window_manager.modal_handler_add(self)
        self.timer = context.window_manager.event_timer_add(
            1.0, window=context.window
        )
        return {'RUNNING_MODAL'}

    def cancel(self, context):
        if self.timer:
            context.window_manager.event_timer_remove(self.timer)
            self.timer = None
        self.thread = None
        self.is_finishing = True


operator_classes = [
    ELEMENTS_OT_SimulateParticles,
]


def register():
    for operator_class in operator_classes:
        bpy.utils.register_class(operator_class)


def unregister():
    for operator_class in reversed(operator_classes):
        bpy.utils.unregister_class(operator_class)
