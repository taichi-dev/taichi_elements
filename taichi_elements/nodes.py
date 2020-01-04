import bpy

from . import types


class BaseNode(bpy.types.Node):
    @classmethod
    def poll(cls, node_tree):
        return node_tree.bl_idname == 'elements_node_tree'

    def get_class(self):
        return None

    def update(self):
        for input_socket in self.inputs:
            if len(input_socket.links):
                for link in input_socket.links:
                    socket_nodes = self.required_nodes.get(input_socket.name, None)
                    if not link.from_node.bl_idname in socket_nodes:
                        bpy.context.space_data.node_tree.links.remove(link)


class ElementsMpmSolverNode(BaseNode):
    bl_idname = 'elements_mpm_solver_node'
    bl_label = 'MPM Solver'

    required_nodes = {
        'Resolution': ['elements_integer_node', ],
    }

    def init(self, context):
        self.width = 175.0

        solver_output_socket = self.outputs.new(
            'elements_struct_socket',
            'MPM Solver'
        )
        solver_output_socket.text = 'Solver Settings'

        resolution = self.inputs.new(
            'elements_integer_socket',
            'Resolution'
        )
        resolution.text = 'Resolution'
        resolution.value = 128

    def get_class(self):
        simulation_class = types.MpmSolverSettings()
        simulation_class.resolution = self.inputs['Resolution'].get_value()
        return simulation_class


class ElementsMaterialNode(BaseNode):
    bl_idname = 'elements_material_node'
    bl_label = 'Material'

    items = [
        ('WATER', 'Water', ''),
        ('SNOW', 'Snow', ''),
        ('ELASTIC', 'Elastic', '')
    ]
    material: bpy.props.EnumProperty(
        items=items, default='WATER'
    )

    def init(self, context):
        material_output_socket = self.outputs.new(
            'elements_struct_socket',
            'Material'
        )
        material_output_socket.text = 'Material Settings'

    def draw_buttons(self, context, layout):
        layout.prop(self, 'material', text='')

    def get_class(self):
        simulation_class = types.Material()
        simulation_class.material_type = self.material
        return simulation_class


class ElementsIntegerNode(BaseNode):
    bl_idname = 'elements_integer_node'
    bl_label = 'Integer'

    def init(self, context):
        integer_socket = self.outputs.new(
            'elements_integer_socket',
            'Integer'
        )
        integer_socket.text = ''


class ElementsFloatNode(BaseNode):
    bl_idname = 'elements_float_node'
    bl_label = 'Float'

    def init(self, context):
        float_socket = self.outputs.new(
            'elements_float_socket',
            'Float'
        )
        float_socket.text = ''


class ElementsEmitterNode(BaseNode):
    bl_idname = 'elements_emitter_node'
    bl_label = 'Emitter'

    required_nodes = {
        'Emit Time': ['elements_integer_node', ],
        'Source Geometry': ['elements_source_object_node', ],
        'Material': ['elements_material_node', ]
    }

    def init(self, context):
        emitter_output_socket = self.outputs.new(
            'elements_struct_socket',
            'Emitter'
        )
        emitter_output_socket.text = 'Emitter'

        emit_time_socket = self.inputs.new(
            'elements_integer_socket',
            'Emit Time'
        )
        emit_time_socket.text = 'Emit Time'

        source_geometry_socket = self.inputs.new(
            'elements_struct_socket',
            'Source Geometry'
        )
        source_geometry_socket.text = 'Source Geometry'

        material_socket = self.inputs.new(
            'elements_struct_socket',
            'Material'
        )
        material_socket.text = 'Material'

    def get_class(self):
        simulation_class = types.Emitter()
        simulation_class.emit_time = self.inputs['Emit Time'].get_value()
        simulation_class.bpy_object = self.inputs['Source Geometry'].get_value()
        simulation_class.material = self.inputs['Material'].get_value()
        return simulation_class


class ElementsSimulationNode(BaseNode):
    bl_idname = 'elements_simulation_node'
    bl_label = 'Simulation'

    required_nodes = {
        'Solver': ['elements_mpm_solver_node', ],
        'Hubs': [
            'elements_hub_node',
            'elements_make_list_node',
            'elements_merge_node'
        ]
    }

    def init(self, context):
        simulation_data_socket = self.outputs.new(
            'elements_struct_socket',
            'Simulation Data'
        )
        simulation_data_socket.text = 'Particles'

        solver_socket = self.inputs.new(
            'elements_struct_socket',
            'Solver'
        )
        solver_socket.text = 'Solver'

        hubs_socket = self.inputs.new(
            'elements_struct_socket',
            'Hubs'
        )
        hubs_socket.text = 'Hubs'

    def draw_buttons(self, context, layout):
        layout.operator('elements.simulate_particles')

    def get_class(self):
        simulation_class = types.Simulation()
        simulation_class.solver = self.inputs['Solver'].get_value()
        simulation_class.hubs = self.inputs['Hubs'].get_value()
        return simulation_class

    def get_output_class(self):
        simulation_class = types.Particles()
        return simulation_class


class ElementsHubNode(BaseNode):
    bl_idname = 'elements_hub_node'
    bl_label = 'Hub'

    required_nodes = {
        'Forces': [
            'elements_gravity_node',
            'elements_make_list_node',
            'elements_merge_node'
        ],
        'Emitters': [
            'elements_emitter_node',
            'elements_make_list_node',
            'elements_merge_node'
        ],
    }

    def init(self, context):
        hub_socket = self.outputs.new(
            'elements_struct_socket',
            'Hub Data'
        )
        hub_socket.text = 'Hub Data'

        forces_socket = self.inputs.new(
            'elements_struct_socket',
            'Forces'
        )
        forces_socket.text = 'Forces'

        emitters_socket = self.inputs.new(
            'elements_struct_socket',
            'Emitters'
        )
        emitters_socket.text = 'Emitters'

    def get_class(self):
        simulation_class = types.Hub()
        simulation_class.forces = self.inputs['Forces'].get_value()
        simulation_class.emitters = self.inputs['Emitters'].get_value()
        return simulation_class


class ElementsSourceObjectNode(BaseNode):
    bl_idname = 'elements_source_object_node'
    bl_label = 'Source Object'

    object_name: bpy.props.StringProperty()

    def init(self, context):
        object_output_socket = self.outputs.new(
            'elements_struct_socket',
            'Object'
        )
        object_output_socket.text = 'Source Geometry'

    def draw_buttons(self, context, layout):
        layout.prop_search(self, 'object_name', bpy.data, 'objects', text='')

    def get_class(self):
        simulation_class = types.SourceObject()
        simulation_class.bpy_object = bpy.data.objects.get(self.object_name, None)
        return simulation_class


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

    def init(self, context):
        self.width = 200.0

        particles_input_socket = self.inputs.new(
            'elements_struct_socket',
            'Particles'
        )
        particles_input_socket.text = 'Particles'

        cache_folder_input_socket = self.inputs.new(
            'elements_folder_socket',
            'Folder'
        )
        cache_folder_input_socket.text = 'Folder'

    def get_class(self):
        simulation_class = types.DiskCache()
        simulation_class.output_folder = self.inputs['Particles'].from_node.get_output_class()
        simulation_class.output_folder = self.inputs['Folder'].get_value()
        return simulation_class


class ElementsFolderNode(BaseNode):
    bl_idname = 'elements_folder_node'
    bl_label = 'Folder'

    def init(self, context):
        self.width = 250.0

        cache_folder_output_socket = self.outputs.new(
            'elements_folder_socket',
            'Folder'
        )
        cache_folder_output_socket.text = ''


class ElementsGravityNode(BaseNode):
    bl_idname = 'elements_gravity_node'
    bl_label = 'Gravity'

    required_nodes = {
        'Speed': [
            'elements_float_node',
            'elements_integer_node'
        ],
        'Direction': [],
    }

    def init(self, context):
        self.width = 175.0

        gravity_output = self.outputs.new(
            'elements_struct_socket',
            'Gravity'
        )
        gravity_output.text = 'Gravity Force'
        speed_socket = self.inputs.new(
            'elements_float_socket',
            'Speed'
        )
        speed_socket.text = 'Speed'
        speed_socket.value = 9.8

        direction_socket = self.inputs.new(
            'elements_3d_vector_float_socket',
            'Direction'
        )
        direction_socket.text = 'Direction'
        direction_socket.value = (0.0, 0.0, -1.0)

    def get_class(self):
        simulation_class = types.GravityForceField()
        simulation_class.output_folder = self.inputs['Speed'].get_value()
        simulation_class.output_folder = self.inputs['Direction'].get_value()
        return simulation_class


class ElementsDynamicSocketsNode(BaseNode):
    def add_linked_socket(self, links):
        empty_input_socket = self.inputs.new(
            'elements_struct_socket',
            'Element'
        )
        empty_input_socket.text = self.text
        node_tree = bpy.context.space_data.node_tree
        if len(links):
            node_tree.links.new(links[0].from_socket, empty_input_socket)

    def add_empty_socket(self):
        empty_input_socket = self.inputs.new(
            'elements_add_socket',
            'Add'
        )
        empty_input_socket.text = self.text_empty_socket

    def init(self, context):
        self.add_empty_socket()
        output_socket = self.outputs.new(
            'elements_struct_socket',
            'Set Elements'
        )
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


class ElementsMakeListNode(ElementsDynamicSocketsNode):
    bl_idname = 'elements_make_list_node'
    bl_label = 'Make List'

    text: bpy.props.StringProperty(default='Element')
    text_empty_socket: bpy.props.StringProperty(default='Add Element')

    def get_class(self):
        simulation_class = types.List()
        for element in self.inputs:
            if element.bl_idname != 'elements_add_socket':
                element_class = element.get_value()
                simulation_class.elements.append(element_class)
        return simulation_class


class ElementsMergeNode(ElementsDynamicSocketsNode):
    bl_idname = 'elements_merge_node'
    bl_label = 'Merge'

    text: bpy.props.StringProperty(default='List')
    text_empty_socket: bpy.props.StringProperty(default='Merge Lists')

    def get_class(self):
        simulation_class = types.Merge()
        for element in self.inputs:
            if element.bl_idname != 'elements_add_socket':
                element_class = element.get_value()
                if hasattr(element_class, 'elements'):
                    simulation_class.elements.extend(element_class.elements)
        return simulation_class


node_classes = [
    ElementsMpmSolverNode,
    ElementsMaterialNode,
    ElementsEmitterNode,
    ElementsSimulationNode,
    ElementsHubNode,
    ElementsSourceObjectNode,
    ElementsIntegerNode,
    ElementsFloatNode,
    ElementsGravityNode,
    ElementsMakeListNode,
    ElementsMergeNode,
    ElementsCacheNode,
    ElementsFolderNode
]


def register():
    for node_class in node_classes:
        bpy.utils.register_class(node_class)


def unregister():
    for node_class in reversed(node_classes):
        bpy.utils.unregister_class(node_class)
