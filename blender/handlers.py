# standart modules
import os

# blender modules
import bpy
import bmesh

# addon modules
from . import operators, particles_io


# name of particles object
PAR_OBJ_NAME = 'elements_particles'
# name of particles system object
PSYS_OBJ_NAME = 'elements_particle_system'


# get elements node tree
def get_trees():
    trees = []
    for node_group in bpy.data.node_groups:
        if node_group.bl_idname == 'elements_node_tree':
            trees.append(node_group)
    return trees


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
    # scene current frame
    cur_frm = bpy.context.scene.frame_current
    # set particle system settings
    psys_stgs.frame_start = cur_frm
    psys_stgs.frame_end = cur_frm
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
# p_mat - particles materials ids
def upd_psys_obj(psys_obj, p_pos, p_vel, p_ang, p_life):
    if not len(psys_obj.particle_systems):
        psys_obj.modifiers.new('Elements Particles', 'PARTICLE_SYSTEM')
        set_psys_settings(psys_obj)
    # particle system settings
    psys_stgs = psys_obj.particle_systems[0].settings
    psys_stgs.count = len(p_pos)
    psys_stgs.use_rotations = True
    psys_stgs.rotation_mode = 'NONE'
    psys_stgs.angular_velocity_mode = 'NONE'
    # blender dependency graph
    degp = bpy.context.evaluated_depsgraph_get()
    # particle system
    psys = psys_obj.evaluated_get(degp).particle_systems[0]
    pos = []
    for p in p_pos:
        pos.extend((p[0], p[1], p[2]))
    psys.particles.foreach_set('location', pos)
    if len(p_vel) != 1:
        vel = []
        for v in p_vel:
            vel.extend((v[0], v[1], v[2]))
        psys.particles.foreach_set('velocity', vel)
    if len(p_ang) != 1:
        ang = []
        for a in p_ang:
            ang.extend((a[0], a[1], a[2]))
        psys.particles.foreach_set('angular_velocity', ang)
    # scene current frame
    cur_frm = bpy.context.scene.frame_current
    psys_stgs.frame_end = cur_frm
    psys_stgs.frame_start = cur_frm
    # used when navigating a timeline in the opposite direction
    psys_stgs.frame_end = cur_frm
    degp.update()
    if len(p_life) != 1:
        psys.particles.foreach_set('lifetime', p_life)


# get outputs nodes
def get_outs_nds():
    outs_nds = []
    # node tree
    trees = get_trees()

    if not trees:
        return

    for tree in trees:
        for node in tree.nodes:
            if node.bl_idname == 'elements_particles_system_node':
                outs_nds.append(node)

    return outs_nds


def create_psys(node):
    node.get_class()
    scn = bpy.context.scene
    # node object
    nd_obj, frm = scn.elements_nodes[node.name]
    obj_name = nd_obj.obj_name
    if not obj_name:
        obj_name = PSYS_OBJ_NAME
    # particle system object
    psys_obj = bpy.data.objects.get(obj_name, None)
    if not psys_obj:
        psys_obj = create_psys_obj(obj_name)
    p_pos = nd_obj.position
    p_vel = nd_obj.velocity
    p_ang = nd_obj.angular_velocity
    p_life = nd_obj.lifetime
    upd_psys_obj(psys_obj, p_pos, p_vel, p_ang, p_life)


# import simulation data
@bpy.app.handlers.persistent
def imp_sim_data(scene):
    # outputs nodes
    outs = get_outs_nds()
    for node in outs:
        create_psys(node)


def register():
    bpy.app.handlers.frame_change_pre.append(imp_sim_data)
    bpy.app.handlers.render_init.append(imp_sim_data)


def unregister():
    bpy.app.handlers.render_init.remove(imp_sim_data)
    bpy.app.handlers.frame_change_pre.remove(imp_sim_data)
