from . import base


class ElementsFolderNode(base.BaseNode):
    bl_idname = 'elements_folder_node'
    bl_label = 'Folder'

    category = base.INPUTS

    def init(self, context):
        self.width = 250.0

        out = self.outputs.new('elements_folder_socket', 'Folder')
        out.text = ''
