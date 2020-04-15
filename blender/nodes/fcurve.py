import bpy

from . import base
from ..categories import SOURCE_DATA


class ElementsFCurveNode(base.BaseNode):
    bl_idname = 'elements_fcurve_node'
    bl_label = 'FCurve'

    # action name
    act: bpy.props.StringProperty()
    # fcurve index
    index: bpy.props.IntProperty(min=0, name='FCurve Index')
    category = SOURCE_DATA

    def init(self, context):
        output_socket = self.outputs.new(
            'elements_struct_socket',
            'FCurve Values'
        )
        output_socket.text = 'FCurve Values'

    def draw_buttons(self, context, layout):
        layout.prop_search(self, 'act', bpy.data, 'actions', text='Action')
        action = bpy.data.actions.get(self.act, None)
        if action:
            layout.prop(self, 'index')
            if len(action.fcurves) > self.index:
                fcurve = action.fcurves[self.index]
                layout.label(text=fcurve.data_path)
