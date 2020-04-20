import inspect

import bpy

from .base import *
# input
from .integer import *
from .float_value import *
from .vector import *
from .color import *
from .folder import *
from .obj import *
from .texture import *
from .fcurve import *
from .time_info import *

# element
from .simulation import *
from .mpm_solver import *
from .material import *
from .emitter import *
from .inflow import *
from .gravity import *
from .hub import *
from .cache import *

# output
from .particles_system import *

# converter
from .int_to_float import *
from .color_to_vector import *
from .hex_color_to_rgb import *
from .float_math import *
from .vector_math import *
from .combine_vector import *
from .seratate_vector import *

# color
from .bright_contrast import *
from .gamma import *
from .invert import *
from .mix_rgb import *

# struct
from .make_list import *
from .merge import *


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
