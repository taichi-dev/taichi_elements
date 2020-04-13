import os, struct

import bpy, bmesh

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
                    if space.node_tree:
                        if space.node_tree.bl_idname == 'elements_node_tree':
                            return space.node_tree


def get_particles():
    node_tree = get_node_tree()
    particles = []
    velocities = []
    colors = []
    if not node_tree:
        return particles, velocities, colors, False, False
    fake_operator = FakeOperator()
    simulation_node = operators.get_simulation_nodes(fake_operator, node_tree)
    if not simulation_node:
        return particles, velocities, colors, False, False
    cache_folder, par_sys, par_mesh = operators.get_cache_folder(simulation_node)
    if not par_sys and not par_mesh:
        return particles, velocities, colors, False, False
    if not cache_folder:
        return particles, velocities, colors, False, False
    particles_file_name = 'particles_{0:0>6}.bin'.format(
        bpy.context.scene.frame_current)
    abs_particles_path = os.path.join(cache_folder, particles_file_name)
    if os.path.exists(abs_particles_path):
        with open(abs_particles_path, 'rb') as file:
            data = file.read()
        pos = 0
        particles_count = struct.unpack('I', data[pos:pos + 4])[0]
        pos += 4
        for particle_index in range(particles_count):
            particle_location = struct.unpack('3f', data[pos:pos + 12])
            pos += 12
            particles.extend(particle_location)
            particle_velocity = struct.unpack('3f', data[pos:pos + 12])
            pos += 12
            velocities.extend(particle_velocity)
            particle_color = struct.unpack('I', data[pos:pos + 4])[0]
            pos += 4
            colors.append(particle_color)
    return particles, velocities, colors, par_sys, par_mesh


def update_particles_mesh(particles_object, particles_locations):
    particles_mesh_old = particles_object.data
    particles_mesh_old.name = 'temp'
    particles_mesh_new = bpy.data.meshes.new('elements_particles_mesh')

    verts = []
    for particle_index in range(0, len(particles_locations), 3):
        verts.append((particles_locations[particle_index],
                      particles_locations[particle_index + 1],
                      particles_locations[particle_index + 2]))

    particles_mesh_new.from_pydata(verts, (), ())
    particles_object.data = particles_mesh_new
    bpy.data.meshes.remove(particles_mesh_old)


def create_particles_object():
    particles_mesh = bpy.data.meshes.new('elements_particles_mesh')
    particles_object = bpy.data.objects.new('elements_particles_object',
                                            particles_mesh)
    bpy.context.scene.collection.objects.link(particles_object)
    return particles_object


def create_particle_system_object():
    particle_system_mesh = bpy.data.meshes.new('elements_particle_system_mesh')
    particle_system_object = bpy.data.objects.new(
        'elements_particle_system_object', particle_system_mesh)
    bm = bmesh.new()
    bmesh.ops.create_cube(bm)
    bm.to_mesh(particle_system_mesh)
    bpy.context.scene.collection.objects.link(particle_system_object)
    particle_system_object.hide_viewport = False
    particle_system_object.hide_render = False
    particle_system_object.hide_select = False
    particle_sys_modifier = particle_system_object.modifiers.new(
        'Elements Particles', 'PARTICLE_SYSTEM')

    particle_system_object.show_instancer_for_render = False
    particle_system_object.show_instancer_for_viewport = False

    particle_system_object.particle_systems[0].settings.frame_start = 0
    particle_system_object.particle_systems[0].settings.frame_end = 0
    particle_system_object.particle_systems[0].settings.lifetime = 1000
    particle_system_object.particle_systems[0].settings.particle_size = 0.005
    particle_system_object.particle_systems[0].settings.display_size = 0.005
    particle_system_object.particle_systems[0].settings.color_maximum = 10.0
    particle_system_object.particle_systems[
        0].settings.display_color = 'VELOCITY'
    particle_system_object.particle_systems[0].settings.display_method = 'DOT'
    return particle_system_object


def update_particle_system_object(particle_system_object, particles_locations,
                                  particles_velocity, particles_color):
    particle_sys_modifier = particle_system_object.modifiers[
        'Elements Particles']

    particle_system_object.particle_systems[0].settings.count = len(
        particles_locations) // 3
    particle_system_object.particle_systems[0].settings.use_rotations = True
    particle_system_object.particle_systems[0].settings.rotation_mode = 'NONE'
    particle_system_object.particle_systems[0].settings.angular_velocity_mode = 'NONE'

    degp = bpy.context.evaluated_depsgraph_get()

    particle_systems = particle_system_object.evaluated_get(
        degp).particle_systems
    particle_system = particle_systems[0]
    particle_system.particles.foreach_set('location', particles_locations)
    particle_system.particles.foreach_set('velocity', particles_velocity)
    particles_color_float = []
    for particle_color in particles_color:
        particles_color_float.extend((
            ((particle_color >> 16) & 0xFF) / 0xFF,
            ((particle_color >> 8) & 0xFF) / 0xFF,
            (particle_color & 0xFF) / 0xFF
        ))
    particle_system.particles.foreach_set('angular_velocity', particles_color_float)
    particle_system_object.particle_systems[0].settings.frame_end = 0


@bpy.app.handlers.persistent
def import_simulation_data(scene):
    (
        particles_locations, particles_velocity, particles_color,
        create_particles_system, create_particles_mesh
    ) = get_particles()

    # create particles mesh
    if create_particles_mesh:
        particles_object = bpy.data.objects.get('elements_particles_object', None)
        if not particles_object:
            particles_object = create_particles_object()
        update_particles_mesh(particles_object, particles_locations)

    # create particles system
    if create_particles_system:
        particle_system_object = bpy.data.objects.get(
            'elements_particle_system_object', None)
        if not particle_system_object:
            particle_system_object = create_particle_system_object()
        update_particle_system_object(particle_system_object, particles_locations,
                                    particles_velocity, particles_color)


def register():
    bpy.app.handlers.frame_change_pre.append(import_simulation_data)
    bpy.app.handlers.render_init.append(import_simulation_data)


def unregister():
    bpy.app.handlers.render_init.remove(import_simulation_data)
    bpy.app.handlers.frame_change_pre.remove(import_simulation_data)