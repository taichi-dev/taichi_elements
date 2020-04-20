import bpy

from .. import base


class ElementsSourceObjectNode(base.BaseNode):
    bl_idname = 'elements_source_object_node'
    bl_label = 'Object'

    obj_name: bpy.props.StringProperty()
    category = base.INPUT

    def init(self, context):
        self.width = 180.0

        out = self.outputs.new('elements_struct_socket', 'Object')
        out.text = 'Object'

    def draw_buttons(self, context, layout):
        layout.prop_search(self, 'obj_name', bpy.data, 'objects', text='')
