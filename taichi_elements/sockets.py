import bpy


socket_colors = {
    'SOLVER': (0.0, 1.0, 0.0, 1.0),
    'NUMBER': (0.3, 0.3, 0.3, 1.0),
    'STRUCT': (1.0, 0.0, 1.0, 1.0)
}


class ElementsSolverSocket(bpy.types.NodeSocket):
    bl_idname = 'elements_solver_socket'

    text: bpy.props.StringProperty(default='Solver')

    def draw_color(self, context, node):
        return socket_colors['SOLVER']

    def draw(self, context, layout, node, text):
        layout.label(text=self.text)


class ElementsMaterialSocket(bpy.types.NodeSocket):
    bl_idname = 'elements_material_socket'

    items = [
        ('0', 'Water', ''),
        ('1', 'Snow', ''),
        ('2', 'Elastic', '')
    ]
    value: bpy.props.EnumProperty(
        items=items, default='0'
    )

    def draw_color(self, context, node):
        return socket_colors['STRUCT']

    def draw(self, context, layout, node, text):
        layout.prop(self, 'value', text='')


class ElementsNumberSocket(bpy.types.NodeSocket):
    bl_idname = 'elements_integer_socket'

    def draw(self, context, layout, node, text):
        if not len(self.links) or self.is_output:
            if self.text:
                row = layout.split(factor=0.6)
                row.label(text=self.text)
                row.prop(self, 'value', text='')
            else:
                row = layout.split(factor=1.0)
                row.prop(self, 'value', text='')
        else:
            layout.label(text=self.text)


class ElementsIntegerSocket(ElementsNumberSocket):
    bl_idname = 'elements_integer_socket'

    value: bpy.props.IntProperty(default=0)
    text: bpy.props.StringProperty(default='Integer')

    def draw_color(self, context, node):
        return socket_colors['NUMBER']


class ElementsFloatSocket(ElementsNumberSocket):
    bl_idname = 'elements_float_socket'

    value: bpy.props.FloatProperty(default=0.0)
    text: bpy.props.StringProperty(default='Float')

    def draw_color(self, context, node):
        return socket_colors['NUMBER']


class Elements3dVectorFloatSocket(ElementsNumberSocket):
    bl_idname = 'elements_3d_vector_float_socket'

    value: bpy.props.FloatVectorProperty(default=(0.0, 0.0, 0.0), size=3)
    text: bpy.props.StringProperty(default='Float')

    def draw_color(self, context, node):
        return socket_colors['NUMBER']


class ElementsStructSocket(bpy.types.NodeSocket):
    bl_idname = 'elements_struct_socket'

    text: bpy.props.StringProperty(default='Value')

    def draw_color(self, context, node):
        return socket_colors['STRUCT']

    def draw(self, context, layout, node, text):
        layout.label(text=self.text)


socket_classes = [
    ElementsIntegerSocket,
    ElementsFloatSocket,
    Elements3dVectorFloatSocket,
    ElementsMaterialSocket,
    ElementsSolverSocket,
    ElementsStructSocket
]


def register():
    for socket_class in socket_classes:
        bpy.utils.register_class(socket_class)


def unregister():
    for socket_class in reversed(socket_classes):
        bpy.utils.unregister_class(socket_class)
