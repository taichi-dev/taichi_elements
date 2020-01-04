# Elements
High-Performance Continuum Physics Engine (work in progress). 

The solver is beening developed using Taichi, therefore it is cross-platform and supports multithreaded CPUs and massively parallel GPUs. 

The short-term plan is
 - To build a reusable multimaterial (water/elastic/snow/sand/mud) simulator
 - To integrate the simulator into Blender

## How to run
 - Install [Taichi](https://github.com/yuanming-hu/taichi) with pip
 - `python3 demo_2d.py` and you will see
 <img src="https://github.com/yuanming-hu/public_files/raw/master/graphics/elements/demo_2d.gif">
 - `python3 demo_3d.py` and you will see a 3D simulation visualized in 2D.
