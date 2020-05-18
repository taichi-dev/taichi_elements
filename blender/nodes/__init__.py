import inspect

import bpy

from .base import *
# categories
from .inputs import *
from .output import *
from .component import *
from .color import *
from .converter import *
from .struct import *


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
