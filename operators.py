import threading
import struct
import os

import bpy, bmesh
from . import mpm_solver
import taichi as ti
import numpy as np


mpm_solver.USE_IN_BLENDER = True


def get_cache_folder(simulation_node):
    particles_socket = simulation_node.outputs['Simulation Data']
    if particles_socket.is_linked:
        for link in particles_socket.links:
            disk_cache_node = link.to_node
            folder = disk_cache_node.inputs['Folder'].get_value()
            par_sys = disk_cache_node.create_particles_system
            par_mesh = disk_cache_node.create_particles_mesh
            folder = bpy.path.abspath(folder)
            return folder, par_sys, par_mesh


def get_simulation_nodes(operator, node_tree):
    simulation_nodes = []
    for node in node_tree.nodes:
        if node.bl_idname == 'elements_simulation_node':
            simulation_nodes.append(node)
    simulation_nodes_count = len(simulation_nodes)
    if simulation_nodes_count != 1:
        if simulation_nodes_count > 1:
            operator.report({
                'WARNING'
            }, 'The node tree must not contain more than 1 "Simulation" node.')
            return
    else:
        return simulation_nodes[0]


def create_emitter(sim, emitter):
    source_geometry = emitter.source_geometry
    if not source_geometry:
        return
    obj_name = emitter.source_geometry.bpy_object_name
    obj = bpy.data.objects.get(obj_name)
    if not obj:
        return
    if obj.type != 'MESH':
        return
    if not emitter.material:
        return
    b_mesh = bmesh.new()
    b_mesh.from_mesh(obj.data)
    bmesh.ops.triangulate(b_mesh, faces=b_mesh.faces)
    triangles = []
    for face in b_mesh.faces:
        triangle = []
        for vertex in face.verts:
            v = obj.matrix_world @ vertex.co
            triangle.extend(v)
        triangles.append(triangle)

    triangles = np.array(triangles, dtype=np.float32)
    print('Object "{0}": {1} tris'.format(obj.name, len(triangles)))

    b_mesh.clear()
    material = emitter.material.material_type
    taichi_material = mpm_solver.MPMSolver.materials.get(material, None)
    if taichi_material is None:
        assert False, material
    red = int(emitter.color.r * 255) << 16
    green = int(emitter.color.g * 255) << 8
    blue = int(emitter.color.b * 255)
    color = red | green | blue
    sim.add_mesh(
        triangles=triangles,
        material=taichi_material,
        color=color
    )


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
        # self.frame_end + 1 - this means include the last frame in the range
        for frame in range(self.frame_start, self.frame_end + 1, 1):
            if self.event_type == 'ESC':
                print('STOP SIMULATION')
                self.thread = None
                self.is_finishing = True
                self.cancel(bpy.context)
                return
            print('Frame: {}'.format(frame))
            for emitter in self.emitters:
                if emitter.emit_frame == frame:
                    create_emitter(self.sim, emitter)
            # generate simulation state at t = 0
            particles = self.sim.particle_info()
            np_x = particles['position']
            np_v = particles['velocity']
            np_material = particles['material']
            np_color = particles['color']
            # and then start time stepping
            self.sim.step(1 / self.fps)
            print(np_x)

            if not os.path.exists(self.cache_folder):
                os.makedirs(self.cache_folder)

            particles_file_path = os.path.join(
                self.cache_folder, 'particles_{0:0>6}.bin'.format(frame))
            data = bytearray()
            particles_count = len(np_x)
            data.extend(struct.pack('I', particles_count))
            print(particles_count)
            for particle_index in range(particles_count):
                data.extend(struct.pack('3f', *np_x[particle_index]))
                data.extend(struct.pack('3f', *np_v[particle_index]))
                data.extend(struct.pack('I', np_color[particle_index]))

            write_obj = False
            if write_obj:
                with open(particles_file_path + '.obj', 'w') as f:
                    for i in range(particles_count):
                        x = np_x[i]
                        print(f'v {x[0]} {x[1]} {x[2]}', file=f)

            with open(particles_file_path, 'wb') as file:
                file.write(data)

    def init_simulation(self):
        self.is_runnig = True
        self.scene.elements_nodes.clear()
        simulation_node = get_simulation_nodes(self, self.node_tree)
        if not simulation_node:
            return {'FINISHED'}

        simulation_node.get_class()
        simulation_class = self.scene.elements_nodes[simulation_node.name]
        self.cache_folder, _, _ = get_cache_folder(simulation_node)

        if not self.cache_folder:
            self.report({'WARNING'}, 'Cache folder not specified')
            return {'FINISHED'}

        for i, j in self.scene.elements_nodes.items():
            print(i, j)

        simulation_class = self.scene.elements_nodes[simulation_node.name]

        self.frame_start = simulation_class.frame_start
        self.frame_end = simulation_class.frame_end
        self.fps = simulation_class.fps

        # TODO: list is not implemented

        res = simulation_class.solver.resolution
        size = simulation_class.solver.size
        ti.reset()
        print(f"Creating simulation of res {res}, size {size}")
        sim = mpm_solver.MPMSolver((res, res, res), size=size)

        hub = simulation_class.hubs
        assert len(hub.forces) == 1, "Only one gravity supported"
        gravity_direction = hub.forces[0].direction
        print('g =', gravity_direction)
        sim.set_gravity(tuple(gravity_direction))

        self.emitters = hub.emitters
        self.size = size
        self.sim = sim
        self.run_simulation()

    def launch_simulation(self):
        self.thread = threading.Thread(target=self.init_simulation, args=())
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
            1.0, window=context.window)
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
