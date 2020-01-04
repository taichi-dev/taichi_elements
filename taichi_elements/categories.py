import bpy, nodeitems_utils


class ElementsNodeCategory(nodeitems_utils.NodeCategory):
    @classmethod
    def poll(cls, context):
        return context.space_data.tree_type == 'elements_node_tree'


# Solvers Category
items = [
    nodeitems_utils.NodeItem('elements_mpm_solver_node'),
    nodeitems_utils.NodeItem('elements_material_node')
]
elements_solvers_category = ElementsNodeCategory(
    'elements_solvers_category',
    'Solvers',
    items=items
)
# Simulation Objects Category
items = [
    nodeitems_utils.NodeItem('elements_emitter_node'),
    nodeitems_utils.NodeItem('elements_hub_node'),
    nodeitems_utils.NodeItem('elements_simulation_node')
]
elements_simulation_objects_category = ElementsNodeCategory(
    'elements_simulation_objects_category',
    'Simulation Objects',
    items=items
)
# Source Data Category
items = [
    nodeitems_utils.NodeItem('elements_source_object_node'),
]
elements_source_data_category = ElementsNodeCategory(
    'elements_source_data_category',
    'Source Data',
    items=items
)
# Input Category
items = [
    nodeitems_utils.NodeItem('elements_integer_node'),
    nodeitems_utils.NodeItem('elements_float_node'),
    nodeitems_utils.NodeItem('elements_folder_node')
]
elements_input_category = ElementsNodeCategory(
    'elements_input_category',
    'Input',
    items=items
)
# Force Fields Category
items = [
    nodeitems_utils.NodeItem('elements_gravity_node'),
]
elements_force_fields_category = ElementsNodeCategory(
    'elements_force_fields_category',
    'Force Fields',
    items=items
)
# Struct Category
items = [
    nodeitems_utils.NodeItem('elements_make_list_node'),
    nodeitems_utils.NodeItem('elements_merge_node')
]
elements_struct_category = ElementsNodeCategory(
    'elements_struct_category',
    'Struct',
    items=items
)
# Output Category
items = [
    nodeitems_utils.NodeItem('elements_cache_node'),
]
elements_output_category = ElementsNodeCategory(
    'elements_output_category',
    'Output',
    items=items
)
# Layout Category
items = [
    nodeitems_utils.NodeItem('NodeFrame'),
    nodeitems_utils.NodeItem('NodeReroute')
]
elements_layout_category = ElementsNodeCategory(
    'elements_layout_category',
    'Layout',
    items=items
)

node_categories = [
    elements_solvers_category,
    elements_simulation_objects_category,
    elements_source_data_category,
    elements_input_category,
    elements_force_fields_category,
    elements_struct_category,
    elements_output_category,
    elements_layout_category
]


def register():
    nodeitems_utils.register_node_categories(
        'elements_node_tree', node_categories
    )


def unregister():
    nodeitems_utils.unregister_node_categories('elements_node_tree')
