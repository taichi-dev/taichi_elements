import bpy

from .. import base
from ... import utils


def get_out_value(socket):
    frm_strt, frm_end = utils.get_frame_info()
    node = socket.node
    action = bpy.data.actions.get(node.act, None)
    if action is None:
        return [1.0, ]
    if len(action.fcurves) > node.index:
        values = []
        scn = bpy.context.scene
        for frame in range(frm_strt, frm_end + 1):
            fcurve = action.fcurves[node.index]
            value = fcurve.evaluate(frame)
            values.append(value)
        out = node.outputs['FCurve Values']
        key = '{0}.{1}'.format(node.name, out.name)
        scn.elements_sockets[key] = values
        return values
    else:
        return [1.0, ]


class ElementsFCurveNode(base.BaseNode):
    bl_idname = 'elements_fcurve_node'
    bl_label = 'FCurve'

    # action name
    act: bpy.props.StringProperty()
    # fcurve index
    index: bpy.props.IntProperty(min=0, name='FCurve Index')
    category = base.INPUT
    get_value = {'FCurve Values': get_out_value, }

    def init(self, context):
        self.width = 220.0

        output_socket = self.outputs.new(
            'elements_float_socket',
            'FCurve Values'
        )
        output_socket.hide_value = True
        output_socket.text = 'FCurve Values'

    def draw_buttons(self, context, layout):
        layout.prop_search(self, 'act', bpy.data, 'actions', text='Action')
        action = bpy.data.actions.get(self.act, None)
        if action:
            layout.prop(self, 'index')
            if len(action.fcurves) > self.index:
                fcurve = action.fcurves[self.index]
                layout.label(text=fcurve.data_path)
