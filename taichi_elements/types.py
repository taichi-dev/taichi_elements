class Emitter:
    def __init__(self):
        self.emit_time = None
        self.source_geometry = None
        self.material = None


class SourceObject:
    def __init__(self):
        self.bpy_object = None


class Material:
    def __init__(self):
        self.material_type = None


class Hub:
    def __init__(self):
        self.forces = None
        self.emitters = None


class Simulation:
    def __init__(self):
        self.hubs = None
        self.solver = None


class MpmSolverSettings:
    def __init__(self):
        self.resolution = None


class GravityForceField:
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
