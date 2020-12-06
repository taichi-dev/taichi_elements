# standart modules
import threading
import struct
import os

# blender modules
import bpy
import bmesh

# addon modules
import taichi as ti
import numpy as np
from .engine import mpm_solver
from . import types, particles_io


WARN_SIM_NODE = 'The node tree must not contain more than 1 "Simulation" node.'
mpm_solver.USE_IN_BLENDER = True
IMPORT_NODES = (
    'elements_particles_mesh_node',
    'elements_particles_system_node'
)


# sim_node - simulation node
def get_cache_folder(sim_node):
    # particles socket
    par_s = sim_node.outputs['Simulation Data']
    if par_s.is_linked:
        for link in par_s.links:
            # disk cache node
            disk = link.to_node
            folder_raw = disk.inputs['Folder'].get_value()[0]
            folder = bpy.path.abspath(folder_raw)
            return folder


# get simulation nodes tree object
def get_tree_obj(node_tree):
    # simulation nodes tree object
    tree = types.Tree()

    for node in node_tree.nodes:
        if node.bl_idname == 'elements_simulation_node':
            tree.sim_nds[node.name] = node
        elif node.bl_idname in IMPORT_NODES:
            if node.bl_idname == 'elements_particles_system_node':
                import_type = 'PAR_SYS'
            elif node.bl_idname == 'elements_particles_mesh_node':
                import_type = 'PAR_MESH'
            node.get_class()
            tree.imp_nds[node.name] = node, import_type
        elif node.bl_idname == 'elements_cache_node':
            tree.cache_nds[node.name] = node

    return tree


def create_emitter(solv, emitter, vel):
    # source object
    src_obj = emitter.source_object

    if not src_obj:
        return

    obj_name = src_obj.obj_name
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
    # emitter triangles
    tris = []

    for face in b_mesh.faces:
         # triangle
        tri = []
        # v - bmesh vertex
        for v in face.verts:
            # final vertex coordinate
            v_co = obj.matrix_world @ v.co
            tri.extend(v_co)
        tris.append(tri)

    b_mesh.clear()
    tris = np.array(tris, dtype=np.float32)
    # material type
    mat = emitter.material.typ
    # taichi material
    ti_mat = mpm_solver.MPMSolver.materials.get(mat, None)

    if ti_mat is None:
        assert False, mat

    # emitter particles color
    red = int(emitter.color[0].r * 255) << 16
    green = int(emitter.color[0].g * 255) << 8
    blue = int(emitter.color[0].b * 255)
    color = red | green | blue
    # add emitter
    solv.add_mesh(triangles=tris, material=ti_mat, color=color, velocity=vel)


class ELEMENTS_OT_SimulateParticles(bpy.types.Operator):
    bl_idname = "elements.simulate_particles"
    bl_label = "Simulate"

    def __init__(self):
        self.timer = None
        self.thread = None
        self.is_runnig = False
        self.is_finishing = False
        self.event_type = 'DEFAULT'

    def create_emitters(self, frame):
        for emitter in self.emitters:
            if len(emitter.velocity) == 1:
                vel = emitter.velocity[0]
            else:
                vel = emitter.velocity[frame]
            if emitter.typ == 'EMITTER':
                if emitter.emit_frame[0] == frame:
                    create_emitter(self.solv, emitter, vel)
            elif emitter.typ == 'INFLOW':
                if type(emitter.enable) == float:
                    enable = emitter.enable
                else:
                    if len(emitter.enable) == 1:
                        index = 0
                    else:
                        index = frame
                    enable = bool(int(round(emitter.enable[index], 0)))
                if enable:
                    create_emitter(self.solv, emitter, vel)

    def save_particles(self, frame, np_x, np_v, np_color, np_material):
        if not os.path.exists(self.cache_folder):
            os.makedirs(self.cache_folder)

        # file name
        fname = 'particles_{0:0>6}'.format(frame)
        # particle file path
        pars_fpath = os.path.join(self.cache_folder, fname)
        # particles data
        par_data = {
            particles_io.POS: np_x,
            particles_io.VEL: np_v,
            particles_io.COL: np_color,
            particles_io.MAT: np_material
        }
        data = particles_io.write_pars_v1(par_data, pars_fpath, fname)

        with open(pars_fpath + '.bin', 'wb') as file:
            file.write(data)

        write_obj = False

        if write_obj:
            with open(pars_fpath + '.obj', 'w') as f:
                for i in range(pars_cnt):
                    x = np_x[i]
                    print(f'v {x[0]} {x[1]} {x[2]}', file=f)

    def run_sim(self):
        # self.frame_end + 1 - this means include the last frame in the range
        for frame in range(self.frame_start, self.frame_end + 1, 1):
            if self.event_type == 'ESC':
                print('STOP SIMULATION')
                self.thread = None
                self.is_finishing = True
                self.cancel(bpy.context)
                return
            print('Frame: {}'.format(frame))

            self.create_emitters(frame)

            # generate simulation state at t = 0
            # particles
            pars = self.solv.particle_info()
            np_x = pars['position']
            np_v = pars['velocity']
            np_material = pars['material']
            np_color = pars['color']
            # and then start time stepping
            self.solv.step(1 / self.fps)
            print(np_x)

            self.save_particles(frame, np_x, np_v, np_color, np_material)

    def init_sim(self):
        # simulation nodes
        sim = []
        for node in self.node_tree.nodes:
            if node.bl_idname == 'elements_simulation_node':
                sim.append(node)

        if len(sim) != 1:
            self.report({'WARNING'}, WARN_SIM_NODE)
        else:
            inputs = sim[0].inputs
            self.scene.elements_frame_start = inputs['Frame Start'].get_value()[0]
            self.scene.elements_frame_end = inputs['Frame End'].get_value()[0]

        self.is_runnig = True
        self.scene.elements_nodes.clear()
        tree = get_tree_obj(self.node_tree)
        # simulation nodes count
        sim_nodes_cnt = len(tree.sim_nds)
    
        if sim_nodes_cnt != 1:
            if sim_nodes_cnt > 1:
                self.report({'WARNING'}, WARN_SIM_NODE)
                return

        sim = list(tree.sim_nds.values())[0]

        if not sim:
            return {'FINISHED'}

        sim.get_class()
        # simulation class
        cls, _ = self.scene.elements_nodes[sim.name]
        self.cache_folder = get_cache_folder(sim)

        if not self.cache_folder:
            self.report({'WARNING'}, 'Cache folder not specified')
            return {'FINISHED'}

        self.frame_start = cls.frame_start[0]
        self.frame_end = cls.frame_end[0]
        self.fps = cls.fps[0]

        # TODO: list is not implemented

        res = cls.solver.resolution[0]
        size = cls.solver.size[0]
        ti.reset()
        print(f"Creating simulation of res {res}, size {size}")
        solv = mpm_solver.MPMSolver((res, res, res), size=size, unbounded=True)

        solv.set_gravity(tuple(cls.gravity[0]))

        self.emitters = cls.emitters
        for collider in cls.colliders:
            solv.add_surface_collider(
                (*collider.position),
                (*collider.direction),
                surface=int(collider.surface)
            )
        self.size = size
        self.solv = solv
        self.run_sim()

    def launch_sim(self):
        self.thread = threading.Thread(target=self.init_sim, args=())
        self.thread.start()

    def modal(self, context, event):
        if event.type == 'ESC':
            self.event_type = 'ESC'

        if not self.is_runnig:
            self.launch_sim()

        if self.is_finishing:
            self.cancel(context)
            return {'FINISHED'}

        return {'PASS_THROUGH'}

    def execute(self, context):
        self.node_tree = context.space_data.node_tree
        self.scene = context.scene
        context.window_manager.modal_handler_add(self)
        win = context.window
        self.timer = context.window_manager.event_timer_add(1.0, window=win)
        return {'RUNNING_MODAL'}

    def cancel(self, context):
        if self.timer:
            context.window_manager.event_timer_remove(self.timer)
            self.timer = None
        self.thread = None
        self.is_finishing = True


# operators draw function
def op_draw_func(self, context):
    if context.space_data.node_tree:
        if context.space_data.node_tree.bl_idname == 'elements_node_tree':
            self.layout.operator('elements.simulate_particles')
            self.layout.operator('elements.stable_render_animation')


class ELEMENTS_OT_StableRenderAnimation(bpy.types.Operator):
    bl_idname = 'elements.stable_render_animation'
    bl_label = 'Render'
    bl_description = 'Stable Render Animation'

    @classmethod
    def poll(cls, context):
        # space data
        spc_data = context.space_data
        if spc_data.node_tree:
            return spc_data.node_tree.bl_idname == 'elements_node_tree'

    def execute(self, context):
        scn = context.scene
        rend = scn.render
        rend.image_settings.file_format = 'PNG'
        # output folder
        out = rend.filepath

        for frm in range(scn.frame_start, scn.frame_end + 1):
            file_name = '{0:0>4}.png'.format(frm)
            file_path = os.path.join(bpy.path.abspath(out), file_name)
            if rend.use_overwrite or not os.path.exists(file_path): 
                print('Render Frame:', frm)
                scn.frame_set(frm)
                bpy.ops.render.render(animation=False)
                for image in bpy.data.images:
                    if image.type == 'RENDER_RESULT':
                        image.save_render(file_path, scene=scn)
                        bpy.data.images.remove(image)

        return {'FINISHED'}


operator_classes = [
    ELEMENTS_OT_SimulateParticles,
    ELEMENTS_OT_StableRenderAnimation
]


def register():
    for operator_class in operator_classes:
        bpy.utils.register_class(operator_class)


def unregister():
    for operator_class in reversed(operator_classes):
        bpy.utils.unregister_class(operator_class)
