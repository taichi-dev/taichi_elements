# standart modules
import os
import struct

# blender modules
import bpy
import bmesh

# addon modules
from . import operators


# name of particles mesh
PAR_MESH_NAME = 'elements_particles_mesh'
# name of particles object
PAR_OBJ_NAME = 'elements_particles_object'
# name of particles system mesh
PSYS_MESH_NAME = 'elements_particle_system_mesh'
# name of particles system object
PSYS_OBJ_NAME = 'elements_particle_system_object'


class FakeOperator:
    def report(self, message_type, message_text):
        print(message_type, message_text)


# get elements node tree
def get_tree():
    windows = bpy.context.window_manager.windows
    for window in windows:
        screen = window.screen
        for area in screen.areas:
            for space in area.spaces:
                if space.type == 'NODE_EDITOR':
                    if space.node_tree:
                        if space.node_tree.bl_idname == 'elements_node_tree':
                            return space.node_tree


# get particles
def get_pars():
    # node tree
    tree = get_tree()
    # particles positions
    pos = []
    # particles velocities
    vel = []
    # particles colors
    col = []
    # create particle system
    sys = False
    # create particles mesh
    mesh = False

    if not tree:
        return pos, vel, col, sys, mesh

    # fake operator
    op = FakeOperator()
    # simulation node
    sim = operators.get_sim_node(op, tree)

    if not sim:
        return pos, vel, col, sys, mesh

    # cache folder, create particle system, create particles mesh
    folder, sys, mesh = operators.get_cache_folder(sim)

    if not sys and not mesh:
        return pos, vel, col, sys, mesh

    if not folder:
        return pos, vel, col, sys, mesh

    # particles file name
    name = 'particles_{0:0>6}.bin'.format(bpy.context.scene.frame_current)
    # absolute particles file path
    path = os.path.join(folder, name)

    if os.path.exists(path):

        with open(path, 'rb') as file:
            # particles file data
            data = file.read()

        # read offset in file
        offs = 0
        # particles count
        count = struct.unpack('I', data[offs : offs + 4])[0]
        offs += 4

        for index in range(count):
            # particle position
            p_pos = struct.unpack('3f', data[offs : offs + 12])
            offs += 12
            pos.extend(p_pos)

            # particle velocity
            p_vel = struct.unpack('3f', data[offs : offs + 12])
            offs += 12
            vel.extend(p_vel)

            # particle color
            p_col = struct.unpack('I', data[offs : offs + 4])[0]
            offs += 4
            col.append(p_col)

    return pos, vel, col, sys, mesh


# update particles mesh
# Function params:
# obj - particles object, pos - particles positions
def update_pmesh(obj, pos):
    # old particles mesh
    me_old = obj.data
    me_old.name = 'temp'
    # new particles mesh
    me_new = bpy.data.meshes.new(PAR_MESH_NAME)
    verts = []

    # i - particle index
    for i in range(0, len(pos), 3):
        verts.append((pos[i], pos[i + 1], pos[i + 2]))

    me_new.from_pydata(verts, (), ())
    obj.data = me_new
    bpy.data.meshes.remove(me_old)


# create particles object
def create_pobj():
    # particles mesh
    par_me = bpy.data.meshes.new(PAR_MESH_NAME)
    # particles object
    par_obj = bpy.data.objects.new(PAR_OBJ_NAME, par_me)
    bpy.context.scene.collection.objects.link(par_obj)
    return par_obj


# create particle system object
def create_psys_obj():
    # particle system mesh
    psys_me = bpy.data.meshes.new(PSYS_MESH_NAME)
    # particle system object
    psys_obj = bpy.data.objects.new(PSYS_OBJ_NAME, psys_me)
    psys_obj.modifiers.new('Elements Particles', 'PARTICLE_SYSTEM')
    # create geometry
    bm = bmesh.new()
    bmesh.ops.create_cube(bm)
    bm.to_mesh(psys_me)
    bpy.context.scene.collection.objects.link(psys_obj)
    # set obj settings
    psys_obj.hide_viewport = False
    psys_obj.hide_render = False
    psys_obj.hide_select = False
    psys_obj.show_instancer_for_render = False
    psys_obj.show_instancer_for_viewport = False
    # particle system settings
    psys_stgs = psys_obj.particle_systems[0].settings
    # set particle system settings
    psys_stgs.frame_start = 0
    psys_stgs.frame_end = 0
    psys_stgs.lifetime = 1000
    psys_stgs.particle_size = 0.005
    psys_stgs.display_size = 0.005
    psys_stgs.color_maximum = 10.0
    psys_stgs.display_color = 'VELOCITY'
    psys_stgs.display_method = 'DOT'
    return psys_obj


# update particle system object
# Function params:
# psys_obj - particle system object
# p_pos - particles positions
# p_vel - particles velocities
# p_col - particles colors
def upd_psys_obj(psys_obj, p_pos, p_vel, p_col):
    # particle system settings
    psys_stgs = psys_obj.particle_systems[0].settings
    psys_stgs.count = len(p_pos) // 3
    psys_stgs.use_rotations = True
    psys_stgs.rotation_mode = 'NONE'
    psys_stgs.angular_velocity_mode = 'NONE'
    # blender dependency graph
    degp = bpy.context.evaluated_depsgraph_get()
    # particle system
    psys = psys_obj.evaluated_get(degp).particle_systems[0]
    psys.particles.foreach_set('location', p_pos)
    psys.particles.foreach_set('velocity', p_vel)
    # particles color in float format
    p_col_flt = []

    # col - particle color
    for col in p_col:
        red = ((col >> 16) & 0xFF) / 0xFF
        green = ((col >> 8) & 0xFF) / 0xFF
        blue = (col & 0xFF) / 0xFF
        p_col_flt.extend((red, green, blue))

    psys.particles.foreach_set('angular_velocity', p_col_flt)
    psys_stgs.frame_end = 0


# import simulation data
@bpy.app.handlers.persistent
def imp_sim_data(scene):
    # p_pos - particles positions
    # p_vel - particles velocities
    # p_col - particles colors
    # use_psys - create particle system
    # use_pmesh - create particles mesh
    p_pos, p_vel, p_col, use_psys, use_pmesh = get_pars()

    # create particles mesh
    if use_pmesh:
        # particles object
        p_obj = bpy.data.objects.get(PAR_OBJ_NAME, None)
        if not p_obj:
            p_obj = create_pobj()
        update_pmesh(p_obj, p_pos)

    # create particles system
    if use_psys:
        # particle system object
        psys_obj = bpy.data.objects.get(PSYS_OBJ_NAME, None)
        if not psys_obj:
            psys_obj = create_psys_obj()
        upd_psys_obj(psys_obj, p_pos, p_vel, p_col)


def register():
    bpy.app.handlers.frame_change_pre.append(imp_sim_data)
    bpy.app.handlers.render_init.append(imp_sim_data)


def unregister():
    bpy.app.handlers.render_init.remove(imp_sim_data)
    bpy.app.handlers.frame_change_pre.remove(imp_sim_data)
