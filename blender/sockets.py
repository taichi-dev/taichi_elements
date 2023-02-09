import bpy


# get reroute input socket
def get_socket(socket):
    node = socket.node
    for input_socket in node.inputs:
        for link in input_socket.links:
            from_node = link.from_node
            from_socket = link.from_socket
            if from_node:
                if from_node.bl_idname == 'NodeReroute':
                    from_socket = get_socket(from_socket)
                return from_socket


def get_socket_value(socket):
    node = socket.node
    result = [socket.default, ]

    if hasattr(node, 'get_value'):
        get_value_func = node.get_value.get(socket.name, None)

        if get_value_func:
            get_value_func(socket)
            key = '{0}.{1}'.format(node.name, socket.name)
            scn = bpy.context.scene
            value = scn.elements_sockets.get(key, None)

            if value is None:
                raise BaseException('Cannot find socket value: {}'.format(key))

            result = value

    return result


class ElementsBaseSocket(bpy.types.NodeSocket):
    bl_idname = 'elements_base_socket'

    split_factor = 0.5

    def get_value(self):
        result = [self.default, ]

        if len(self.links) and not self.is_output:
            from_socket = self.links[0].from_socket

            if from_socket.node.bl_idname == 'NodeReroute':
                from_socket = get_socket(from_socket)

            if from_socket:
                if from_socket.bl_idname == self.bl_idname:
                    if hasattr(from_socket, 'get_value'):
                        result = get_socket_value(from_socket)

        return result


    def draw_color(self, context, node):
        return self.color

    def draw(self, context, layout, node, text):
        draw_value = True

        if len(self.links):
            draw_value = False

        if self.is_output:
            draw_value = True

        if self.hide_value:
            draw_value = False

        if draw_value:
            if self.text:
                row = layout.split(factor=self.split_factor)
                row.label(text=self.text)
                row.prop(self, 'default', text='')
            else:
                row = layout.split(factor=1.0)
                row.prop(self, 'default', text='')
        else:
            layout.label(text=self.text)


class ElementsIntegerSocket(ElementsBaseSocket):
    bl_idname = 'elements_integer_socket'

    default: bpy.props.IntProperty(default=0)
    text: bpy.props.StringProperty(default='Integer')
    color = (0.25, 0.25, 0.25, 1.0)


class ElementsFloatSocket(ElementsBaseSocket):
    bl_idname = 'elements_float_socket'

    default: bpy.props.FloatProperty(default=0.0, precision=4)
    text: bpy.props.StringProperty(default='Float')
    color = (0.5, 0.5, 0.5, 1.0)


class ElementsVectorSocket(ElementsBaseSocket):
    bl_idname = 'elements_vector_socket'

    default: bpy.props.FloatVectorProperty(
        default=(0.0, 0.0, 0.0),
        size=3,
        precision=4
    )
    text: bpy.props.StringProperty(default='Float')
    color = (0.25, 0.25, 0.8, 1.0)


class ElementsFolderSocket(ElementsBaseSocket):
    bl_idname = 'elements_folder_socket'

    default: bpy.props.StringProperty(subtype='DIR_PATH')
    text: bpy.props.StringProperty(default='Folder')
    color = (1.0, 0.5, 0.0, 1.0)
    split_factor = 0.35


class ElementsColorSocket(ElementsBaseSocket):
    bl_idname = 'elements_color_socket'

    default: bpy.props.FloatVectorProperty(
        subtype='COLOR',
        min=0.0,
        max=1.0,
        size=3,
        default=(0.8, 0.8, 0.8)
    )
    text: bpy.props.StringProperty(default='Float')
    color = (0.8, 0.8, 0.0, 1.0)


class ElementsAddSocket(ElementsBaseSocket):
    bl_idname = 'elements_add_socket'

    text: bpy.props.StringProperty(default='')
    color = (0.0, 0.0, 0.0, 0.25)

    def draw(self, context, layout, node, text):
        layout.label(text=self.text)


class ElementsStructSocket(ElementsBaseSocket):
    bl_idname = 'elements_struct_socket'

    text: bpy.props.StringProperty(default='Value')
    color = (0.0, 1.0, 0.0, 1.0)

    def get_value(self):
        if not self.is_output and len(self.links):
            from_socket = self.links[0].from_socket
            if from_socket.node.bl_idname == 'NodeReroute':
                from_socket = get_socket(from_socket)
            if from_socket.bl_idname == self.bl_idname:
                return from_socket.links[0].from_node.get_class()
            else:
                return None
        else:
            return None

    def draw(self, context, layout, node, text):
        layout.label(text=self.text)


socket_classes = [
    ElementsIntegerSocket,
    ElementsFloatSocket,
    ElementsVectorSocket,
    ElementsColorSocket,
    ElementsStructSocket,
    ElementsAddSocket,
    ElementsFolderSocket
]


def register():
    for socket_class in socket_classes:
        bpy.utils.register_class(socket_class)


def unregister():
    for socket_class in reversed(socket_classes):
        bpy.utils.unregister_class(socket_class)
