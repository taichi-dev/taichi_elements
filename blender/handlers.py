# standart modules
import os
import struct

# blender modules
import bpy
import bmesh

# addon modules
from . import operators


# name of particles object
PAR_OBJ_NAME = 'elements_particles'
# name of particles system object
PSYS_OBJ_NAME = 'elements_particle_system'


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


# read particles
def read_pars(imp_nd_obj, caches, imps, imp_type):
    # particles positions
    pos = []
    # particles velocities
    vel = []
    # particles colors
    col = []
    # scene
    scn = bpy.context.scene
    # particles file name
    name = 'particles_{0:0>6}.bin'.format(scn.frame_current)
    # cache folder
    folder = imp_nd_obj.cache_folder
    imps.append((folder, imp_nd_obj.obj_name, imp_type))

    if caches.get(folder, None):
        return

    # absolute particles file path
    path = bpy.path.abspath(os.path.join(folder, name))

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

    caches[folder] = (pos, vel, col)


# get particles
def get_pars():
    # cache folders data
    caches = {}
    # importers
    imps = []

    # node tree
    tree = get_tree()

    if not tree:
        return caches, imps

    # simulation nodes tree object
    tree = operators.get_tree_obj(tree)
    # scene
    scn = bpy.context.scene

    # imp_nd - import node
    for imp_nd, (_, imp_type) in tree.imp_nds.items():
        # import node object
        imp_nd_obj = scn.elements_nodes[imp_nd]
        read_pars(imp_nd_obj, caches, imps, imp_type)

    return caches, imps


# update particles mesh
# Function params:
# obj - particles object, pos - particles positions
def update_pmesh(obj, pos, mesh_name):
    # old particles mesh
    me_old = obj.data
    me_old.name = 'temp'
    # new particles mesh
    me_new = bpy.data.meshes.new(mesh_name)
    verts = []

    # i - particle index
    for i in range(0, len(pos), 3):
        verts.append((pos[i], pos[i + 1], pos[i + 2]))

    me_new.from_pydata(verts, (), ())
    obj.data = me_new
    bpy.data.meshes.remove(me_old)


# create particles object
def create_pobj(mesh_name):
    if not mesh_name:
        mesh_name = PAR_OBJ_NAME
    # particles mesh
    par_me = bpy.data.meshes.new(mesh_name)
    # particles object
    par_obj = bpy.data.objects.new(mesh_name, par_me)
    bpy.context.scene.collection.objects.link(par_obj)
    return par_obj


# set particles system settings
def set_psys_settings(psys_obj):
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


# create particle system object
def create_psys_obj(obj_name):
    # particle system mesh
    psys_me = bpy.data.meshes.new(obj_name)
    # particle system object
    psys_obj = bpy.data.objects.new(PSYS_OBJ_NAME, psys_me)
    psys_obj.modifiers.new('Elements Particles', 'PARTICLE_SYSTEM')
    # create geometry
    bm = bmesh.new()
    bmesh.ops.create_cube(bm)
    bm.to_mesh(psys_me)
    bpy.context.scene.collection.objects.link(psys_obj)
    set_psys_settings(psys_obj)
    return psys_obj


# update particle system object
# Function params:
# psys_obj - particle system object
# p_pos - particles positions
# p_vel - particles velocities
# p_col - particles colors
def upd_psys_obj(psys_obj, p_pos, p_vel, p_col):
    if not len(psys_obj.particle_systems):
        psys_obj.modifiers.new('Elements Particles', 'PARTICLE_SYSTEM')
        set_psys_settings(psys_obj)
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
    scene.elements_nodes.clear()
    # caches data, importers
    caches, imps = get_pars()

    for folder, obj_name, imp_type in imps:
        if imp_type == 'PAR_MESH':
            if not obj_name:
                obj_name = PAR_OBJ_NAME
            # particles object
            p_obj = bpy.data.objects.get(obj_name, None)
            if not p_obj:
                p_obj = create_pobj(obj_name)
            # particles positions
            p_pos = caches[folder][0]
            update_pmesh(p_obj, p_pos, obj_name)
        elif imp_type == 'PAR_SYS':
            if not obj_name:
                obj_name = PSYS_OBJ_NAME
            # particle system object
            psys_obj = bpy.data.objects.get(obj_name, None)
            if not psys_obj:
                psys_obj = create_psys_obj(obj_name)
            p_pos = caches[folder][0]
            p_vel = caches[folder][1]
            p_col = caches[folder][2]
            upd_psys_obj(psys_obj, p_pos, p_vel, p_col)


def register():
    bpy.app.handlers.frame_change_pre.append(imp_sim_data)
    bpy.app.handlers.render_init.append(imp_sim_data)


def unregister():
    bpy.app.handlers.render_init.remove(imp_sim_data)
    bpy.app.handlers.frame_change_pre.remove(imp_sim_data)
