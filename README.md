# Taichi Elements [![Build Status](https://travis-ci.com/taichi-dev/taichi_elements.svg?branch=master)](https://travis-ci.com/taichi-dev/taichi_elements) [![Code Coverage](https://codecov.io/gh/taichi-dev/taichi_elements/branch/master/graph/badge.svg)](https://codecov.io/gh/taichi-dev/taichi_elements)
*Taichi elements* is a high-performance multi-material continuum physics engine (work in progress). Features:

- Cross-platform: Windows, Linux, and OS X
- Supports multi-threaded CPUs and massively parallel GPUs
- Supports multiple materials, including water, elastic objects, snow, and sand
- Supports (virtually) infinitely large simulation domains
- Supports [sparse grids](https://docs.taichi.graphics/lang/articles/advanced/sparse)
- Highly efficient and scalable, especially on GPUs

# Using `taichi_elements` in Python
 - Install [taichi](https://github.com/taichi-dev/taichi) with `pip`: `python3 -m pip install taichi`
 - Execute `python3 download_ply.py` to download model files used by the demos
 - Execute `python3 demo/demo_2d.py` and you will see

<img src="https://github.com/yuanming-hu/public_files/raw/master/graphics/elements/demo_2d.gif">

 - Execute `python3 demo_3d.py` and you will see a 3D simulation visualized in 2D
 - Execute `python3 demo_3d_ggui.py` and you will see a 3D simulation rendered with [GGUI](https://docs.taichi.graphics/lang/articles/misc/ggui). Note that GGUI requires Vulkan so please make sure your platform supports that.

<img src="https://github.com/taichi-dev/public_files/raw/master/taichi_elements/demo_3d_ggui.gif" style="zoom:40%;" />

 - Execute `python3 demo/demo_2d_sparse_active_blocks.py` to get a visual understanding of Taichi sparse computation

<img src="https://raw.githubusercontent.com/taichi-dev/public_files/master/taichi_elements/sparse_mpm_active_blocks.gif" height="600px">

## To simulate and render an example 3D scene with Python
- Make sure you have a modern NVIDIA GPU (e.g. GTX 1080)
- Execute `python3 download_ply.py` to download model files
- Run `python3 demo/demo_3d_letters.py` (wait for at least 10 frames)
   - A binary particle folder with a timestamp in its time (e.g., `sim_2020-07-27_20-55-48`) will be created under the current folder.
- Example:

 ```bash
python3 render_particles.py \
     -i ./path/to/particles \
     -b 0 -e 400 -s 1 \
     -o ./output \
     --gpu-memory 20 \
     -M 460 \
     --shutter-time 0.0 \
     -r 128
 ```
 - Images are in the `output/` folder. For example, 100 million MPM particles simulated in 8 hours on a V100 GPU:

[[Watch on YouTube]](https://youtu.be/klMDVUzFFnk)
<img src="https://raw.githubusercontent.com/taichi-dev/public_files/master/taichi_elements/100Mparticles.jpg" height="600px">

- Here is a one-billion-particle simulation on a GPU with `80` GB memory. Each particle takes 40 bytes, thanks to [quantization](https://github.com/taichi-dev/quantaichi).

https://user-images.githubusercontent.com/2309174/162767279-2ced8a2f-38bd-42d6-9bb8-a827144464ff.mp4

# Using `taichi_elements` in Blender

## Installing Taichi for Blender
(Not sure if it is the standard approach, but it works for now.)
 - Find the Python3 executable bundled in Blender. Open a console in Blender and type in
 ```python
 import sys
 print(sys.exec_prefix)
 ```
  The output looks like `/XXX/blender-2.81a-linux-glibc217-x86_64/2.81`, which means python3 is located at `/XXX/blender-2.81a-linux-glibc217-x86_64/2.81/python/bin/python3.7`
 - Install [pip](https://pip.pypa.io/en/stable/installing/) using that Python executable
 - Install Taichi: `./python3.7m -m pip install --upgrade taichi` (Note: as of Oct 8 2021, Taichi version is `v0.8.1`. Please use the latest version.)

## Installing taichi_elements (experimental) for Blender
 - Set the environment variable `BLENDER_USER_ADDON_PATH`, e.g. `/home/XXX/.config/blender/2.81/scripts/addons`
 - Go to `utils` folder
 - Execute `python3 install_blender_addon.py` to install the addon
   - If you are doing development and wish to **k**eep refreshing the installed addon, add an argument `-k`.
 - Restart Blender to reload the addon
