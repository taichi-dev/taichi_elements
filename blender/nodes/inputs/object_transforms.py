import bpy

from .. import base


def get_loc(socket):
    get_res(socket, 'Location')


def get_euler(socket):
    get_res(socket, 'Rotation Euler')


def get_scale(socket):
    get_res(socket, 'Scale')


def get_dir(socket):
    get_res(socket, 'Direction')


def get_res(socket, mode):
    node = socket.node
    out = node.outputs[mode]
    # scene
    scn = bpy.context.scene
    key = '{0}.{1}'.format(node.name, out.name)
    # input obj
    obj = node.inputs['Obj'].get_value()
    obj, _ = scn.elements_nodes[obj]
    obj_name = obj.obj_name
    obj = bpy.data.objects.get(obj_name)
    # result
    res = []
    if obj:
        # r - result
        if mode == 'Location':
            r = obj.location
        elif mode == 'Rotation Euler':
            r = obj.rotation_euler
        elif mode == 'Scale':
            r = obj.scale
        elif mode == 'Direction':
            matrix = obj.rotation_euler.to_matrix().to_3x3()
            r = (matrix[0][2], matrix[1][2], matrix[2][2])
        res.append((r[0], r[1], r[2]))
    scn.elements_sockets[key] = res


class ElementsObjectTransformsNode(base.BaseNode):
    bl_idname = 'elements_object_transforms_node'
    bl_label = 'Object Transforms'

    required_nodes = {
        'Obj': [
            'elements_source_object_node',
        ],
    }

    category = base.INPUT

    get_value = {
        'Location': get_loc,
        'Rotation Euler': get_euler,
        'Scale': get_scale,
        'Direction': get_dir
    }

    def init(self, context):
        self.width = 180.0

        obj = self.inputs.new('elements_struct_socket', 'Obj')
        obj.text = 'Object'

        loc = self.outputs.new('elements_vector_socket', 'Location')
        loc.text = 'Location'
        loc.hide_value = True

        euler = self.outputs.new('elements_vector_socket', 'Rotation Euler')
        euler.text = 'Rotation Euler'
        euler.hide_value = True

        scale = self.outputs.new('elements_vector_socket', 'Scale')
        scale.text = 'Scale'
        scale.hide_value = True

        direct = self.outputs.new('elements_vector_socket', 'Direction')
        direct.text = 'Direction'
        direct.hide_value = True
