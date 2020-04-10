Blender nodes for Taichi elements
=================================

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





----------------------------

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
