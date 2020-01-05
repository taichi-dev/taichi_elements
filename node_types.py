import bpy


def get_node_class(node_name):
    if node_name is None:
        return None
    scene = bpy.context.scene
    node_class = scene.elements_nodes[node_name]
    return node_class


class BaseSimulationObject:
    def __len__(self):
        return 1

    def __next__(self):
        if self.offset == 0:
            self.offset += 1
            return self
        else:
            self.offset = 0
            raise StopIteration

    def __getitem__(self, item):
        return [self, ].__getitem__(item)


class Emitter(BaseSimulationObject):
    @property
    def source_geometry(self):
        return get_node_class(self._source_geometry)

    @property
    def material(self):
        return get_node_class(self._material)

    @source_geometry.setter
    def source_geometry(self, value):
        self._source_geometry = value

    @material.setter
    def material(self, value):
        self._material = value

    def __init__(self):
        self.offset = 0
        self.emit_time = None
        self._source_geometry = None
        self._material = None


class SourceObject:
    def __init__(self):
        self.bpy_object = None


class Texture:
    def __init__(self):
        self.bpy_texture = None


class Material:
    def __init__(self):
        self.material_type = None


class Hub(BaseSimulationObject):
    @property
    def forces(self):
        return get_node_class(self._forces)

    @property
    def emitters(self):
        return get_node_class(self._emitters)

    @forces.setter
    def forces(self, value):
        self._forces = value

    @emitters.setter
    def emitters(self, value):
        self._emitters = value

    def __init__(self):
        self._forces = None
        self._emitters = None


class Simulation:
    @property
    def solver(self):
        return get_node_class(self._solver)

    @property
    def hubs(self):
        return get_node_class(self._hubs)

    @solver.setter
    def solver(self, value):
        self._solver = value

    @hubs.setter
    def hubs(self, value):
        self._hubs = value

    def __init__(self):
        self._hubs = None
        self._solver = None


class MpmSolverSettings:
    def __init__(self):
        self.resolution = None
        self.domain_object = None


class GravityForceField(BaseSimulationObject):
    def __init__(self):
        self.speed = None
        self.direction = None


class DiskCache:
    def __init__(self):
        self.output_folder = None


class Particles:
    def __init__(self):
        pass


class List:
    def __init__(self):
        self.elements = []
        self.offset = 0
      
    def __len__(self):
        return len(self.elements)
    
    def __next__(self):
        if self.offset < len(self.elements):
            item = get_node_class(self.elements[self.offset])
            self.offset += 1
            return item
        else:
            self.offset = 0
            raise StopIteration

    def __getitem__(self, item):
        return get_node_class(self.elements.__getitem__(item))


class Merge:
    def __init__(self):
        self.elements = []


elements_types = [
    Emitter,
    SourceObject,
    Material,
    Hub,
    Simulation,
    MpmSolverSettings,
    GravityForceField,
    DiskCache,
    Particles,
    List,
    Merge
]
