import inspect

import bpy

from .base import *
from .color import *
from .cache import *
from .emitter import *
from .fcurve import *
from .float_value import *
from .folder import *
from .particles_mesh import *
from .particles_system import *
from .combine_vector import *
from .color_to_vector import *
from .hex_color_to_rgb import *
from .int_to_float import *
from .seratate_vector import *
from .float_math import *
from .vector_math import *
from .gravity import *
from .hub import *
from .inflow import *
from .integer import *
from .make_list import *
from .material import *
from .merge import *
from .mpm_solver import *
from .simulation import *
from .source_object import *
from .texture import *
from .vector import *


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
