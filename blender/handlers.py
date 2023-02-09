# blender modules
import bpy

# site package modules
import numpy


# name of particles object
OBJ_NAME = 'taichi_elements_particles'


def update_pmesh(obj, pos, vel, emitters, colors):
    me = obj.data

    pos_count = len(pos) // 3
    verts_count = len(me.vertices)

    if not pos_count and not verts_count:
        return

    if not pos_count:
        me.clear_geometry()
        return

    if not verts_count:
        pos = pos.reshape((pos.shape[0] // 3, 3))
        me.from_pydata(pos, (), ())

    elif pos_count == verts_count:
        me.vertices.foreach_set('co', pos)

    elif pos_count > verts_count:
        diff = pos_count - verts_count
        me.vertices.add(diff)
        me.vertices.foreach_set('co', pos)

    elif pos_count < verts_count:
        me.clear_geometry()
        me.vertices.add(pos_count)
        me.vertices.foreach_set('co', pos)

    for attr in me.attributes:
        me.attributes.remove(attr)

    if len(emitters) > 1:
        emt_attr = me.attributes.new('ti_emitter', 'INT', 'POINT')
        emt_attr.data.foreach_set('value', emitters)

    if len(colors) > 1:
        col_attr = me.attributes.get('ti_color')
        if not col_attr:
            col_attr = me.attributes.new('ti_color', 'BYTE_COLOR', 'POINT')
        colors_float = []
        for col in colors:
            red = (col >> 16 & 0xff) / 0xff
            green = (col >> 8 & 0xff) / 0xff
            blue = (col & 0xff) / 0xff
            alpha = 1.0
            colors_float.extend((red, green, blue, alpha))
        col_attr.data.foreach_set('color', colors_float)
    else:
        col_attr = me.attributes.get('ti_color')
        if col_attr:
            me.attributes.remove(col_attr)

    if len(vel) == len(pos):
        vel_attr = me.attributes.new('ti_velocity', 'FLOAT_VECTOR', 'POINT')
        vel_attr.data.foreach_set('vector', vel)

    me.update()


# create particles object
def create_pobj(name):
    if not name:
        name = OBJ_NAME

    par_me = bpy.data.meshes.new(name)

    par_obj = bpy.data.objects.new(name, par_me)
    bpy.context.scene.collection.objects.link(par_obj)

    return par_obj


def get_output_nodes():
    mesh_nodes = []

    for tree in bpy.data.node_groups:
        if tree.bl_idname == 'elements_node_tree':
            for node in tree.nodes:
                if node.bl_idname == 'elements_mesh_node':
                    mesh_nodes.append(node)

    return mesh_nodes


def create_mesh(node):
    node.get_class()
    scn = bpy.context.scene

    node_obj, _ = scn.elements_nodes[node.name]
    obj_struct = node_obj.mesh_object

    if obj_struct:
        obj_name = obj_struct.obj_name
    else:
        obj_name = OBJ_NAME

    me_obj = bpy.data.objects.get(obj_name, None)
    if not me_obj:
        me_obj = create_pobj(obj_name)

    verts = node_obj.vertices
    vels = node_obj.velocity
    emitters = node_obj.emitters
    colors = node_obj.color

    if type(verts) == numpy.ndarray:
        update_pmesh(me_obj, verts, vels, emitters, colors)
    else:
        update_pmesh(me_obj, (), (), (), ())


CURRENT_FRAME = None
IMPORTED_COUNT = 0


# import simulation data
@bpy.app.handlers.persistent
def imp_sim_data(scene):
    global CURRENT_FRAME
    global IMPORTED_COUNT
    if scene.frame_current == CURRENT_FRAME:
        if IMPORTED_COUNT:
            return
    else:
        IMPORTED_COUNT = 0
    CURRENT_FRAME = scene.frame_current
    IMPORTED_COUNT += 1

    output_nodes = get_output_nodes()
    for node in output_nodes:
        create_mesh(node)


handlers = (
    bpy.app.handlers.frame_change_pre,
    bpy.app.handlers.render_init
)


def register():
    for handler in handlers:
        handler.append(imp_sim_data)


def unregister():
    for handler in reversed(handlers):
        handler.remove(imp_sim_data)
