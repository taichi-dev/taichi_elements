Blender nodes for Taichi elements
=================================

.. contents:: Contents
   :depth: 3

Node System
-----------
(TODO)

Node Categories
---------------
(TODO)

Node Sockets
------------
(TODO)

Nodes
-----





MPM Solver
~~~~~~~~~~
.. tip:: Location: ``Add > Solvers > MPM Solver``

Description
"""""""""""
This node tells the simulation to use the MPM method (currently the Material Point Method is the only available simulation method). This node stores the settings of the MPM solver.

Parameters
""""""""""
`It has no parameters.`

Inputs
""""""
**Domain Object** - this socket is temporarily not working.

**Resolution** - domain resolution in voxels. The simulation will use a cubic domain. For example, if the Resolution value is 64, then the domain resolution will be 64 x 64 x 64.

**Size** - domain size in meters. The domain is created in such a way that its left, back, bottom corner (in the direction -X, -Y, -Z) is at coordinates 0, 0, 0. And if Size is 10.0, then the right, front, top corner will have a coordinate 10, 10, 10.

Outputs
"""""""
**Solver Settings** - it is a socket, which is a set of MPM solver parameters.





----------------------------

Material
~~~~~~~~
.. tip:: Location: ``Add > Solvers > Material``

Description
"""""""""""
This node stores information about the properties of the material. Using this node, you can specify what physical characteristics the emitter particles will have. Be it the material of water, snow, sand, etc.

Parameters
""""""""""
**Material Type** - This parameter specifies what the material will be for Emitters. The following options are available: water, sand, snow, elastic.

Inputs
""""""
`It has no inputs.`

Outputs
"""""""
**Material Settings** - This output is material settings. At the moment, from the settings there is only the type of material.





----------------------------

Integer
~~~~~~~
.. tip:: Location: ``Add > Inputs > Integer``

Description
"""""""""""
This is a simple input node that provides an integer value.

Parameters
""""""""""
`It has no parameters.`

Inputs
""""""
`It has no inputs.`

Outputs
"""""""
**Integer Value** - an integer value that can be connected to any other integer socket.





----------------------------

Float
~~~~~
.. tip:: Location: ``Add > Inputs > Float``

Description
"""""""""""
This node represents a floating point number.

Parameters
""""""""""
`It has no parameters.`

Inputs
""""""
`It has no inputs.`

Outputs
"""""""
**Float Value** - this socket provides a floating point number that can be connected to any float socket.





----------------------------

Folder
~~~~~~
.. tip:: Location: ``Add > Inputs > Folder``

Description
"""""""""""
Using this node, you can specify the folder.

Parameters
""""""""""
`It has no parameters.`

Inputs
""""""
`It has no inputs.`

Outputs
"""""""
**String Value** - this output is a string that indicates the folder. 





----------------------------

Emitter
~~~~~~~
.. tip:: Location: ``Add > Simulation Objects > Emitter``

Description
"""""""""""
Using this node, you can add an emitter to the simulation. Emitter is a mesh object that emits particles from its volume once.

Parameters
""""""""""
`It has no parameters.`

Inputs
""""""
**Emit Frame** - indicates the frame number in the timeline in which particles will be emitted.

**Source Geometry** - indicates a mesh object that will emit particles from its volume.

**Material** - this socket accepts material parameters (water, snow, sand, elastic).

**Color** - particle color.

Outputs
"""""""
**Emitter** - this socket is a structure that stores the settings of the emitter.





----------------------------

Inflow
~~~~~~
.. tip:: Location: ``Add > Simulation Objects > Inflow``

Description
"""""""""""
This type of object emits particles like a faucet. An Inflow object can continuously emit particles, and can also stop the emission of particles, and then continue to emit particles.

Parameters
""""""""""
`It has no parameters.`

Inputs
""""""
**Enable FCurve** - This socket accepts an input animation curve that will indicate the enable value for the inflow object. If you want particles to be emitted throughout the simulation, you can omit the animation curve, but the socket must be connected to the FCurve node. But if you need to turn on and off the inflow object during the simulation, you need to specify the animation curve in the FCurve node. At a value of 0.0, inflow will not emit particles, and at a value of 1.0, continuous emission of particles will occur.

**Source Geometry** - indicates a mesh object that will emit particles from its volume.

**Material** - this socket accepts material parameters (water, snow, sand, elastic).

**Color** - particle color.

Outputs
"""""""
**Inflow** - this socket is a structure that stores the settings of the inflow object.





----------------------------

Simulation
~~~~~~~~~~
.. tip:: Location: ``Add > Simulation Objects > Simulation``

Description
"""""""""""
This node is a simulation in general. The simulation is launched using the Simulate operator of this node.

Parameters
""""""""""
`It has no parameters.`

Inputs
""""""
**Frame Start** - frame on the timeline from which the simulation begins. If you plan to create a particle system from the simulation cache, it is better to set this value to 2, since 0 and 1 frames cannot be displayed correctly (this is a limitation of the blender particle system).

**Frame End** - frame number on the timeline in which the simulation ends.

**FPS** - simulation frame rate.

**Solver** - this socket accepts solver settings as input. At the moment, only MPM Solver can be connected.

**Hubs** - This socket accepts a hub or a list of hubs as an input. The hub node is described below.

Outputs
"""""""
**Particles** - this output represents particle simulation data.





----------------------------

Hub
~~~
.. tip:: Location: ``Add > Simulation Objects > Hub``

Description
"""""""""""
This node is the connecting link between emitters and force fields. At the moment, only one force field is supported in the simulation. In the future, the use of different force fields for individual emitters is possible.

Parameters
""""""""""
`It has no parameters.`

Inputs
""""""
**Forces** - input for force fields. At the moment, it is possible to connect only one force field. In the future, the capabilities of the simulator and this node will expand, so that it is possible to connect several force fields.

**Emitters** - socket to connect emitter or emitter list. Those emitters that are not connected to any hub object will not participate in the simulation.

Outputs
"""""""
**Hub Data** - these are the settings of the hub object.
