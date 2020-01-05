import os, struct

import bpy

from . import operators


class FakeOperator:
    def report(self, message_type, message_text):
        print(message_type, message_text)


def get_node_tree():
    windows = bpy.context.window_manager.windows
    for window in windows:
        screen = window.screen
        for area in screen.areas:
            for space in area.spaces:
                if space.type == 'NODE_EDITOR':
                    if space.node_tree.bl_idname == 'elements_node_tree':
                        return space.node_tree


def get_particles():
    node_tree = get_node_tree()
    particles = []
    if not node_tree:
        return particles
    fake_operator = FakeOperator()
    simulation_node = operators.get_simulation_nodes(fake_operator, node_tree)
    cache_folder = operators.get_cache_folder(simulation_node)
    particles_file_name = 'particles_{0:0>6}.bin'.format(bpy.context.scene.frame_current)
    abs_particles_path = os.path.join(cache_folder, particles_file_name)
    if os.path.exists(abs_particles_path):
        with open(abs_particles_path, 'rb') as file:
            data = file.read()
        pos = 0
        particles_count = struct.unpack('I', data[pos : pos + 4])[0]
        pos += 4
        for particle_index in range(particles_count):
            particle_location = struct.unpack('3f', data[pos : pos + 12])
            pos += 12
            particles.append(particle_location)
    return particles


def update_particles_mesh(particles_object):
    particles_mesh_old = particles_object.data
    particles_mesh_old.name = 'temp'
    particles_mesh_new = bpy.data.meshes.new('elements_particles_mesh')
    particles_locations = get_particles()
    particles_mesh_new.from_pydata(particles_locations, (), ())
    particles_object.data = particles_mesh_new
    bpy.data.meshes.remove(particles_mesh_old)


def create_particles_object():
    particles_mesh = bpy.data.meshes.new('elements_particles_mesh')
    particles_object = bpy.data.objects.new(
        'elements_particles_object', particles_mesh
    )
    bpy.context.scene.collection.objects.link(particles_object)
    return particles_object


@bpy.app.handlers.persistent
def import_geometry(scene):
    particles_object = bpy.data.objects.get('elements_particles_object', None)
    if not particles_object:
        particles_object = create_particles_object()
    update_particles_mesh(particles_object)


def register():
    bpy.app.handlers.frame_change_pre.append(import_geometry)


def unregister():
    bpy.app.handlers.frame_change_pre.remove(import_geometry)
