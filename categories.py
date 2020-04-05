import bpy, nodeitems_utils

from . import nodes


class ElementsNodeCategory(nodeitems_utils.NodeCategory):
    @classmethod
    def poll(cls, context):
        return context.space_data.tree_type == 'elements_node_tree'


node_categories_data = {}
for node in nodes.node_classes:
    if not node_categories_data.get(node.category, None):
        node_categories_data[node.category] = []
    node_categories_data[node.category].append(node.bl_idname)

node_categories = []
for category_name, nodes_ids in node_categories_data.items():
    category_items = []
    for node_id in nodes_ids:
        category_items.append(nodeitems_utils.NodeItem(node_id))
    category = ElementsNodeCategory(category_name.lower().replace(' ', '_'),
                                    category_name,
                                    items=category_items)
    node_categories.append(category)


def register():
    nodeitems_utils.register_node_categories('elements_node_tree',
                                             node_categories)


def unregister():
    nodeitems_utils.unregister_node_categories('elements_node_tree')
