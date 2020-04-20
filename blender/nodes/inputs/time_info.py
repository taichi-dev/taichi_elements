import bpy

from .. import base


def get_f_st_value(socket):
    node = socket.node
    out = node.outputs['Frame Start']
    scn = bpy.context.scene
    key = '{0}.{1}'.format(node.name, out.name)
    scn.elements_sockets[key] = [scn.frame_start, ]


def get_f_en_value(socket):
    node = socket.node
    out = node.outputs['Frame End']
    scn = bpy.context.scene
    key = '{0}.{1}'.format(node.name, out.name)
    scn.elements_sockets[key] = [scn.frame_end, ]


def get_f_cur_value(socket):
    node = socket.node
    out = node.outputs['Frame Current']
    scn = bpy.context.scene
    key = '{0}.{1}'.format(node.name, out.name)
    scn.elements_sockets[key] = [scn.frame_current, ]


def get_fps_value(socket):
    node = socket.node
    out = node.outputs['FPS']
    scn = bpy.context.scene
    key = '{0}.{1}'.format(node.name, out.name)
    scn.elements_sockets[key] = [scn.render.fps, ]


class ElementsTimeInfoNode(base.BaseNode):
    bl_idname = 'elements_time_info_node'
    bl_label = 'Time Info'

    category = base.INPUT
    get_value = {
        'Frame Start': get_f_st_value,
        'Frame End': get_f_en_value,
        'Frame Current': get_f_cur_value,
        'FPS': get_fps_value
    }

    def init(self, context):
        f_st = self.outputs.new('elements_integer_socket', 'Frame Start')
        f_st.text = 'Frame Start'
        f_st.hide_value = True

        f_en = self.outputs.new('elements_integer_socket', 'Frame End')
        f_en.text = 'Frame End'
        f_en.hide_value = True

        f_cur = self.outputs.new('elements_integer_socket', 'Frame Current')
        f_cur.text = 'Frame Current'
        f_cur.hide_value = True

        fps = self.outputs.new('elements_integer_socket', 'FPS')
        fps.text = 'FPS'
        fps.hide_value = True
