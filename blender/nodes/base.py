import bpy


# node category names
INPUT = 'Input'
OUTPUT = 'Output'
COMPONENT = 'Component'
CONVERTER = 'Converter'
COLOR = 'Color'
STRUCT = 'Struct'
LAYOUT = 'Layout'

HAS_NO_ATTRIBUTE = 'Has no attribute'


def create_class(node):
    node_class_name = ''.join(
        map(lambda part: part.capitalize(), node.bl_label.split(' ')))
    node_class_attributes = {}
    node_params_attributes = {}
    node_elements = []
    scene = bpy.context.scene

    for input_socket_name, input_socket in node.inputs.items():
        if input_socket.bl_idname != 'elements_add_socket':
            attribute_name = '_'.join(
                map(lambda part: part.lower(), input_socket.text.split(' ')))
            if input_socket.bl_idname == 'elements_struct_socket':
                if node.bl_idname == 'elements_make_list_node':
                    node_elements.append(input_socket.get_value())
                elif node.bl_idname == 'elements_merge_node':
                    node_name = input_socket.get_value()
                    node_class, frm = scene.elements_nodes.get(
                        node_name, (None, None)
                    )
                    if node_class:
                        node_elements.extend(node_class.elements)
                else:
                    input_object = input_socket.get_value()
                    node_class_attributes[attribute_name] = input_object
            else:
                input_value = input_socket.get_value()
                node_params_attributes[attribute_name] = input_value

    if hasattr(node, '__annotations__'):
        for annotation_name in node.__annotations__.keys():
            node_params_attributes[annotation_name] = getattr(
                node, annotation_name)

    def node_init_function(self):
        self.is_list = False
        self.params = {}
        self.inputs = {}
        self.elements = []
        self.offset = 0

        for attribute_name, attribute_value in node_class_attributes.items():
            self.inputs[attribute_name] = attribute_value

        for attribute_name, attribute_value in node_params_attributes.items():
            self.params[attribute_name] = attribute_value

        if len(node_elements):
            self.is_list = True
            for element in node_elements:
                self.elements.append(element)
        else:
            self.elements.append(self)

    def get_attribute_function(self, name):
        params = object.__getattribute__(self, 'params')
        inputs = object.__getattribute__(self, 'inputs')
        elements = object.__getattribute__(self, 'elements')
        is_list = object.__getattribute__(self, 'is_list')
        if name == 'params':
            return params
        elif name == 'inputs':
            return inputs
        elif name == 'elements':
            return elements
        elif name == 'is_list':
            return is_list
        else:
            if self.is_list:
                attribute, frm = bpy.context.scene.elements_nodes.get(
                    name, (None, None)
                )
                return attribute
            else:
                attribute = params.get(name, HAS_NO_ATTRIBUTE)
                if attribute == HAS_NO_ATTRIBUTE:
                    attribute_name = inputs.get(name, HAS_NO_ATTRIBUTE)
                    if attribute_name == HAS_NO_ATTRIBUTE:
                        raise BaseException(
                            'Cannot find attribute: {}'.format(name))
                    if attribute_name is None:
                        return attribute_name
                    attribute, frm = bpy.context.scene.elements_nodes.get(
                        attribute_name, (None, None)
                    )
                return attribute

    def node_len_function(self):
        return len(self.elements)

    def node_next_function(self):
        if self.offset < len(self.elements):
            item = self.elements[self.offset]
            self.offset += 1
            return item
        else:
            self.offset = 0
            raise StopIteration

    def node_getitem_function(self, item):
        if self.is_list:
            return getattr(self, self.elements.__getitem__(item))
        else:
            return self.elements.__getitem__(item)

    def node_dir_function(self):
        # class attributes
        attrs = []
        attrs.extend(self.params.keys())
        attrs.extend(self.inputs.keys())
        return attrs

    node_class = type(
        node_class_name, (), {
            '__init__': node_init_function,
            '__getattribute__': get_attribute_function,
            '__len__': node_len_function,
            '__next__': node_next_function,
            '__getitem__': node_getitem_function,
            '__dir__': node_dir_function
        })

    return node_class()


def find_node_class(node):
    scene = bpy.context.scene
    node_class, frame = scene.elements_nodes.get(node.name, (None, None))
    frm_cur = scene.frame_current
    if not node_class or frm_cur != frame:
        node_class = create_class(node)
        scene.elements_nodes[node.name] = node_class, frm_cur
    return node.name


def get_reroute_input(node):
    if len(node.inputs):
        # reroute links
        re_links = node.inputs[0].links
        if len(re_links):
            from_node = re_links[0].from_node
            node_id = from_node.bl_idname
            if node_id == 'NodeReroute':
                from_node = get_reroute_input(from_node)
            return from_node
        else:
            return
    else:
        return


class BaseNode(bpy.types.Node):
    @classmethod
    def poll(cls, node_tree):
        return node_tree.bl_idname == 'elements_node_tree'

    def get_class(self):
        return find_node_class(self)

    def update(self):
        for input_socket in self.inputs:
            if len(input_socket.links):
                for link in input_socket.links:
                    if input_socket.bl_idname == 'elements_struct_socket':
                        if hasattr(self, 'required_nodes'):
                            socket_nodes = self.required_nodes.get(
                                input_socket.name, None
                            )
                            # linked node
                            node = link.from_node
                            # linked node id name
                            node_id = node.bl_idname
                            if node_id == 'NodeReroute':
                                # reroute from node
                                re_node = get_reroute_input(node)
                                if re_node is None:
                                    return
                                else:
                                    node_id = re_node.bl_idname
                            if not node_id in socket_nodes:
                                bpy.context.space_data.node_tree.links.remove(link)
                    else:
                        from_socket = link.from_socket
                        if from_socket.bl_idname != input_socket.bl_idname:
                            bpy.context.space_data.node_tree.links.remove(link)


class ElementsDynamicSocketsNode():
    def add_linked_socket(self, links):
        empty_input_socket = self.inputs.new('elements_struct_socket',
                                             'Element')
        empty_input_socket.text = self.text
        node_tree = bpy.context.space_data.node_tree
        if len(links):
            node_tree.links.new(links[0].from_socket, empty_input_socket)

    def add_empty_socket(self):
        empty_input_socket = self.inputs.new('elements_add_socket', 'Add')
        empty_input_socket.text = self.text_empty

    def init(self, context):
        self.add_empty_socket()
        output_socket = self.outputs.new('elements_struct_socket',
                                         'Set Elements')
        output_socket.text = 'Elements'

    def update(self):
        for input_socket in self.inputs:
            if input_socket.bl_idname == 'elements_struct_socket':
                if not input_socket.is_linked:
                    self.inputs.remove(input_socket)
        for input_socket in self.inputs:
            if input_socket.bl_idname == 'elements_add_socket':
                if input_socket.is_linked:
                    self.add_linked_socket(input_socket.links)
                    self.inputs.remove(input_socket)
                    self.add_empty_socket()
