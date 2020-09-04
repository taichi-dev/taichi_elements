#!/bin/bash

set -e
mkdir -p build/Taichi-Elements/bundle
python3 -m pip install taichi -t build/Taichi-Elements/bundle
cp -r blender/* build/Taichi-Elements
cp -r engine build/Taichi-Elements
rm -f build/Taichi-Elements.zip
cd build && zip -r Taichi-Elements.zip Taichi-Elements
