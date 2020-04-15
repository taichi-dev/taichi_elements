import bpy

from . import base


class ElementsSourceObjectNode(base.BaseNode):
    bl_idname = 'elements_source_object_node'
    bl_label = 'Source Object'

    name: bpy.props.StringProperty()
    category = base.SOURCE_DATA

    def init(self, context):
        out = self.outputs.new('elements_struct_socket', 'Object')
        out.text = 'Source Geometry'

    def draw_buttons(self, context, layout):
        layout.prop_search(self, 'name', bpy.data, 'objects', text='')
