from . import base
from ..categories import INPUTS


class ElementsFolderNode(base.BaseNode):
    bl_idname = 'elements_folder_node'
    bl_label = 'Folder'

    category = INPUTS

    def init(self, context):
        self.width = 250.0

        out = self.outputs.new('elements_folder_socket', 'Folder')
        out.text = ''
