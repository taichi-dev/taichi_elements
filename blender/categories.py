import bpy
import nodeitems_utils

from . import nodes


class ElementsNodeCategory(nodeitems_utils.NodeCategory):
    @classmethod
    def poll(cls, context):
        return context.space_data.tree_type == 'elements_node_tree'


# get categories data
def get_categs_data():
    # node categories data
    data = {}

    for node in nodes.node_classes:
        if not data.get(node.category, None):
            data[node.category] = []
        data[node.category].append(node.bl_idname)

    # key - category name, values - node identifier
    data['Layout'] = ['NodeFrame', 'NodeReroute']
    return data


def get_categories():
    data = get_categs_data()
    # node categories
    categories = []

    for name, ids in data.items():
        # category items
        items = []
        for node_id in ids:
            items.append(nodeitems_utils.NodeItem(node_id))
        category_id = name.lower().replace(' ', '_')
        category = ElementsNodeCategory(category_id, name, items=items)
        categories.append(category)

    return categories


def register():
    categories = get_categories()
    nodeitems_utils.register_node_categories('elements_node_tree', categories)


def unregister():
    nodeitems_utils.unregister_node_categories('elements_node_tree')
