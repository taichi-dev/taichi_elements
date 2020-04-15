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
from . import mpm_solver


WARN_SIM_NODE = 'The node tree must not contain more than 1 "Simulation" node.'
mpm_solver.USE_IN_BLENDER = True


# sim_node - simulation node
def get_cache_folder(sim_node):
    # particles socket
    par_s = sim_node.outputs['Simulation Data']
    if par_s.is_linked:
        for link in par_s.links:
            # disk cache node
            disk = link.to_node
            folder_raw = disk.inputs['Folder'].get_value()
            folder = bpy.path.abspath(folder_raw)
            par_sys = disk.create_psys
            par_mesh = disk.create_pmesh
            return folder, par_sys, par_mesh


# get simulation node
def get_sim_node(operator, node_tree):
    # simulation nodes
    sim_nodes = []

    for node in node_tree.nodes:
        if node.bl_idname == 'elements_simulation_node':
            sim_nodes.append(node)

    # simulation nodes count
    sim_nodes_cnt = len(sim_nodes)

    if sim_nodes_cnt != 1:
        if sim_nodes_cnt > 1:
            operator.report({'WARNING'}, WARN_SIM_NODE)
            return
    else:
        return sim_nodes[0]


def create_emitter(solv, emitter):
    # source geometry
    src_geom = emitter.source_geometry

    if not src_geom:
        return

    obj_name = src_geom.name
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
    red = int(emitter.color.r * 255) << 16
    green = int(emitter.color.g * 255) << 8
    blue = int(emitter.color.b * 255)
    color = red | green | blue
    # add emitter
    solv.add_mesh(triangles=tris, material=ti_mat, color=color)


class ELEMENTS_OT_SimulateParticles(bpy.types.Operator):
    bl_idname = "elements.simulate_particles"
    bl_label = "Simulate"

    def __init__(self):
        self.timer = None
        self.thread = None
        self.is_runnig = False
        self.is_finishing = False
        self.event_type = 'DEFAULT'

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

            # create emitters
            for emitter in self.emitters:
                if emitter.typ == 'EMITTER':
                    if emitter.emit_frame == frame:
                        create_emitter(self.solv, emitter)
                elif emitter.typ == 'INFLOW':
                    enable = emitter.enable_fcurve
                    action = bpy.data.actions.get(enable.act, None)
                    if action is None:
                        create_emitter(self.solv, emitter)
                        continue
                    if len(action.fcurves) > enable.index:
                        fcurve = action.fcurves[enable.index]
                        enable_value = bool(int(fcurve.evaluate(frame)))
                        if enable_value:
                            create_emitter(self.solv, emitter)
                    else:
                        create_emitter(self.solv, emitter)

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

            if not os.path.exists(self.cache_folder):
                os.makedirs(self.cache_folder)

            # file name
            fname = 'particles_{0:0>6}.bin'.format(frame)
            # particle file path
            pars_fpath = os.path.join(self.cache_folder, fname)
            data = bytearray()
            # particles count
            pars_cnt = len(np_x)
            data.extend(struct.pack('I', pars_cnt))
            print('Particles count:', pars_cnt)

            # par_i - particles index
            for par_i in range(pars_cnt):
                data.extend(struct.pack('3f', *np_x[par_i]))
                data.extend(struct.pack('3f', *np_v[par_i]))
                data.extend(struct.pack('I', np_color[par_i]))

            write_obj = False

            if write_obj:
                with open(pars_fpath + '.obj', 'w') as f:
                    for i in range(pars_cnt):
                        x = np_x[i]
                        print(f'v {x[0]} {x[1]} {x[2]}', file=f)

            with open(pars_fpath, 'wb') as file:
                file.write(data)

    def init_sim(self):
        self.is_runnig = True
        self.scene.elements_nodes.clear()
        # simulation node
        sim = get_sim_node(self, self.node_tree)

        if not sim:
            return {'FINISHED'}

        sim.get_class()
        # simulation class
        cls = self.scene.elements_nodes[sim.name]
        self.cache_folder, _, _ = get_cache_folder(sim)

        if not self.cache_folder:
            self.report({'WARNING'}, 'Cache folder not specified')
            return {'FINISHED'}

        self.frame_start = cls.frame_start
        self.frame_end = cls.frame_end
        self.fps = cls.fps

        # TODO: list is not implemented

        res = cls.solver.resolution
        size = cls.solver.size
        ti.reset()
        print(f"Creating simulation of res {res}, size {size}")
        solv = mpm_solver.MPMSolver((res, res, res), size=size)

        hub = cls.hubs
        assert len(hub.forces) == 1, "Only one gravity supported"
        gravity_direction = hub.forces[0].direction
        solv.set_gravity(tuple(gravity_direction))

        self.emitters = hub.emitters
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


# render operator draw function
def rend_op_draw_func(self, context):
    if context.space_data.node_tree:
        if context.space_data.node_tree.bl_idname == 'elements_node_tree':
            self.layout.operator('elements.stable_render_animation')


class ELEMENTS_OT_StableRenderAnimation(bpy.types.Operator):
    bl_idname = "elements.stable_render_animation"
    bl_label = "Stable Render Animation"

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
