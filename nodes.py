import inspect

import bpy

CATEGORY_SOLVERS_NAME = 'Solvers'
CATEGORY_SIMULATION_OBJECTS_NAME = 'Simulation Objects'
CATEGORY_SOURCE_DATA_NAME = 'Source Data'
CATEGORY_INPUTS_NAME = 'Inputs'
CATEGORY_FORCE_FIELDS_NAME = 'Force Fields'
CATEGORY_STRUCT_NAME = 'Struct'
CATEGORY_OUTPUT_NAME = 'Output'
CATEGORY_LAYOUT_NAME = 'Layout'


def create_class(node):
    node_class_name = ''.join(
        map(lambda part: part.capitalize(), node.bl_label.split(' ')))
    node_class_attributes = {}
    node_params_attributes = {}
    node_elements = []

    for input_socket_name, input_socket in node.inputs.items():
        if input_socket.bl_idname != 'elements_add_socket':
            attribute_name = '_'.join(
                map(lambda part: part.lower(), input_socket.text.split(' ')))
            if input_socket.bl_idname == 'elements_struct_socket':
                if node.bl_idname == 'elements_make_list_node':
                    node_elements.append(input_socket.get_value())
                elif node.bl_idname == 'elements_merge_node':
                    node_elements.extend(input_socket.get_value().elements)
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
                attribute = bpy.context.scene.elements_nodes.get(name, None)
                return attribute
            else:
                attribute = params.get(name, None)
                if not attribute:
                    attribute_name = inputs.get(name, None)
                    if attribute_name is None:
                        raise BaseException(
                            'Cannot find attibute: {}'.format(name))
                    attribute = bpy.context.scene.elements_nodes.get(
                        attribute_name, None)
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

    node_class = type(
        node_class_name, (), {
            '__init__': node_init_function,
            '__getattribute__': get_attribute_function,
            '__len__': node_len_function,
            '__next__': node_next_function,
            '__getitem__': node_getitem_function
        })

    return node_class()


def find_node_class(node):
    scene = bpy.context.scene
    node_class = scene.elements_nodes.get(node.name, None)
    if not node_class:
        node_class = create_class(node)
        scene.elements_nodes[node.name] = node_class
    return node.name


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
                    if hasattr(self, 'required_nodes'):
                        socket_nodes = self.required_nodes.get(
                            input_socket.name, None)
                        if not link.from_node.bl_idname in socket_nodes:
                            bpy.context.space_data.node_tree.links.remove(link)


class ElementsMpmSolverNode(BaseNode):
    bl_idname = 'elements_mpm_solver_node'
    bl_label = 'MPM Solver'

    required_nodes = {
        'Domain Object': [
            'elements_source_object_node',
        ],
        'Resolution': [
            'elements_integer_node',
        ],
        'Size': [
            'elements_float_node',
        ],
    }
    category = CATEGORY_SOLVERS_NAME

    def init(self, context):
        self.width = 175.0

        solver_output_socket = self.outputs.new('elements_struct_socket',
                                                'MPM Solver')
        solver_output_socket.text = 'Solver Settings'

        domain_object_socket = self.inputs.new('elements_struct_socket',
                                               'Domain Object')
        domain_object_socket.text = 'Domain Object'

        resolution = self.inputs.new('elements_integer_socket', 'Resolution')
        resolution.text = 'Resolution'
        resolution.value = 128
        size = self.inputs.new('elements_float_socket', 'Size')
        size.text = 'Size'
        size.value = 10.0


class ElementsMaterialNode(BaseNode):
    bl_idname = 'elements_material_node'
    bl_label = 'Material'

    items = [('WATER', 'Water', ''), ('SNOW', 'Snow', ''),
             ('ELASTIC', 'Elastic', ''), ('SAND', 'Sand', '')]
    material_type: bpy.props.EnumProperty(items=items, default='WATER')
    category = CATEGORY_SOLVERS_NAME

    def init(self, context):
        material_output_socket = self.outputs.new('elements_struct_socket',
                                                  'Material')
        material_output_socket.text = 'Material Settings'

    def draw_buttons(self, context, layout):
        layout.prop(self, 'material_type', text='')


class ElementsIntegerNode(BaseNode):
    bl_idname = 'elements_integer_node'
    bl_label = 'Integer'

    category = CATEGORY_INPUTS_NAME

    def init(self, context):
        integer_socket = self.outputs.new('elements_integer_socket', 'Integer')
        integer_socket.text = ''


class ElementsFloatNode(BaseNode):
    bl_idname = 'elements_float_node'
    bl_label = 'Float'

    category = CATEGORY_INPUTS_NAME

    def init(self, context):
        float_socket = self.outputs.new('elements_float_socket', 'Float')
        float_socket.text = ''


class ElementsEmitterNode(BaseNode):
    bl_idname = 'elements_emitter_node'
    bl_label = 'Emitter'

    required_nodes = {
        'Emit Time': [
            'elements_integer_node',
        ],
        'Source Geometry': [
            'elements_source_object_node',
        ],
        'Material': [
            'elements_material_node',
        ]
    }

    category = CATEGORY_SIMULATION_OBJECTS_NAME

    def init(self, context):
        emitter_output_socket = self.outputs.new('elements_struct_socket',
                                                 'Emitter')
        emitter_output_socket.text = 'Emitter'

        emit_time_socket = self.inputs.new('elements_integer_socket',
                                           'Emit Time')
        emit_time_socket.text = 'Emit Time'

        source_geometry_socket = self.inputs.new('elements_struct_socket',
                                                 'Source Geometry')
        source_geometry_socket.text = 'Source Geometry'

        material_socket = self.inputs.new('elements_struct_socket', 'Material')
        material_socket.text = 'Material'


class ElementsSimulationNode(BaseNode):
    bl_idname = 'elements_simulation_node'
    bl_label = 'Simulation'

    required_nodes = {
        'Solver': [
            'elements_mpm_solver_node',
        ],
        'Hubs': [
            'elements_hub_node', 'elements_make_list_node',
            'elements_merge_node'
        ],
        'Frame Start': [
            'elements_integer_node'
        ],
        'Frame End': [
            'elements_integer_node'
        ],
        'FPS': [
            'elements_integer_node'
        ]
    }

    category = CATEGORY_SIMULATION_OBJECTS_NAME

    def init(self, context):
        simulation_data_socket = self.outputs.new('elements_struct_socket',
                                                  'Simulation Data')
        simulation_data_socket.text = 'Particles'

        frame_start = self.inputs.new('elements_integer_socket', 'Frame Start')
        frame_start.text = 'Frame Start'
        frame_start.value = 0

        frame_end = self.inputs.new('elements_integer_socket', 'Frame End')
        frame_end.text = 'Frame End'
        frame_end.value = 50

        fps = self.inputs.new('elements_integer_socket', 'FPS')
        fps.text = 'FPS'
        fps.value = 30

        solver_socket = self.inputs.new('elements_struct_socket', 'Solver')
        solver_socket.text = 'Solver'

        hubs_socket = self.inputs.new('elements_struct_socket', 'Hubs')
        hubs_socket.text = 'Hubs'

    def draw_buttons(self, context, layout):
        layout.operator('elements.simulate_particles')


class ElementsHubNode(BaseNode):
    bl_idname = 'elements_hub_node'
    bl_label = 'Hub'

    required_nodes = {
        'Forces': [
            'elements_gravity_node', 'elements_make_list_node',
            'elements_merge_node'
        ],
        'Emitters': [
            'elements_emitter_node', 'elements_make_list_node',
            'elements_merge_node'
        ],
    }

    category = CATEGORY_SIMULATION_OBJECTS_NAME

    def init(self, context):
        hub_socket = self.outputs.new('elements_struct_socket', 'Hub Data')
        hub_socket.text = 'Hub Data'

        forces_socket = self.inputs.new('elements_struct_socket', 'Forces')
        forces_socket.text = 'Forces'

        emitters_socket = self.inputs.new('elements_struct_socket', 'Emitters')
        emitters_socket.text = 'Emitters'


class ElementsSourceObjectNode(BaseNode):
    bl_idname = 'elements_source_object_node'
    bl_label = 'Source Object'

    bpy_object_name: bpy.props.StringProperty()
    category = CATEGORY_SOURCE_DATA_NAME

    def init(self, context):
        object_output_socket = self.outputs.new('elements_struct_socket',
                                                'Object')
        object_output_socket.text = 'Source Geometry'

    def draw_buttons(self, context, layout):
        layout.prop_search(self,
                           'bpy_object_name',
                           bpy.data,
                           'objects',
                           text='')


class ElementsCacheNode(BaseNode):
    bl_idname = 'elements_cache_node'
    bl_label = 'Disk Cache'

    required_nodes = {
        'Particles': [
            'elements_simulation_node',
        ],
        'Folder': [
            'elements_folder_node',
        ],
    }

    category = CATEGORY_OUTPUT_NAME

    def init(self, context):
        self.width = 200.0

        particles_input_socket = self.inputs.new('elements_struct_socket',
                                                 'Particles')
        particles_input_socket.text = 'Particles'

        cache_folder_input_socket = self.inputs.new('elements_folder_socket',
                                                    'Folder')
        cache_folder_input_socket.text = 'Folder'


class ElementsFolderNode(BaseNode):
    bl_idname = 'elements_folder_node'
    bl_label = 'Folder'

    category = CATEGORY_INPUTS_NAME

    def init(self, context):
        self.width = 250.0

        cache_folder_output_socket = self.outputs.new('elements_folder_socket',
                                                      'Folder')
        cache_folder_output_socket.text = ''


class ElementsGravityNode(BaseNode):
    bl_idname = 'elements_gravity_node'
    bl_label = 'Gravity'

    required_nodes = {
        'Speed': ['elements_float_node', 'elements_integer_node'],
        'Direction': [],
    }

    category = CATEGORY_FORCE_FIELDS_NAME

    def init(self, context):
        self.width = 175.0

        gravity_output = self.outputs.new('elements_struct_socket', 'Gravity')
        gravity_output.text = 'Gravity Force'
        speed_socket = self.inputs.new('elements_float_socket', 'Speed')
        speed_socket.text = 'Speed'
        speed_socket.value = 9.8

        direction_socket = self.inputs.new('elements_3d_vector_float_socket',
                                           'Direction')
        direction_socket.text = 'Direction'
        direction_socket.value = (0.0, 0.0, -1.0)


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
        empty_input_socket.text = self.text_empty_socket

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


class ElementsTextureNode(BaseNode):
    bl_idname = 'elements_texture_node'
    bl_label = 'Texture'

    texture_name: bpy.props.StringProperty()
    category = CATEGORY_SOURCE_DATA_NAME

    def init(self, context):
        self.width = 250.0

        texture_output = self.outputs.new('elements_struct_socket', 'Texture')
        texture_output.text = 'Texture'

    def draw_buttons(self, context, layout):
        layout.prop_search(self, 'texture_name', bpy.data, 'textures', text='')


class ElementsMakeListNode(ElementsDynamicSocketsNode, BaseNode):
    bl_idname = 'elements_make_list_node'
    bl_label = 'Make List'

    text: bpy.props.StringProperty(default='Element')
    text_empty_socket: bpy.props.StringProperty(default='Add Element')
    category = CATEGORY_STRUCT_NAME


class ElementsMergeNode(ElementsDynamicSocketsNode, BaseNode):
    bl_idname = 'elements_merge_node'
    bl_label = 'Merge'

    text: bpy.props.StringProperty(default='List')
    text_empty_socket: bpy.props.StringProperty(default='Merge Lists')
    category = CATEGORY_STRUCT_NAME


node_classes = []
glabal_variables = globals().copy()
for variable_name, variable_object in glabal_variables.items():
    if hasattr(variable_object, '__mro__'):
        object_mro = inspect.getmro(variable_object)
        if BaseNode in object_mro and variable_object != BaseNode:
            node_classes.append(variable_object)


def register():
    for node_class in node_classes:
        bpy.utils.register_class(node_class)


def unregister():
    for node_class in reversed(node_classes):
        bpy.utils.unregister_class(node_class)
