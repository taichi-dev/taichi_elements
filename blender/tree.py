import bpy


class ElementsNodeTree(bpy.types.NodeTree):
    bl_idname = 'elements_node_tree'
    bl_label = 'Taichi Elements'
    bl_icon = 'PHYSICS'

    @classmethod
    def poll(cls, context):
        return True


def register():
    bpy.utils.register_class(ElementsNodeTree)


def unregister():
    bpy.utils.unregister_class(ElementsNodeTree)
