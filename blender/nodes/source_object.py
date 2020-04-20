import bpy

from . import base


class ElementsSourceObjectNode(base.BaseNode):
    bl_idname = 'elements_source_object_node'
    bl_label = 'Source Object'

    obj_name: bpy.props.StringProperty()
    category = base.SOURCE_DATA

    def init(self, context):
        self.width = 180.0

        out = self.outputs.new('elements_struct_socket', 'Object')
        out.text = 'Source Object'

    def draw_buttons(self, context, layout):
        layout.prop_search(self, 'obj_name', bpy.data, 'objects', text='')
