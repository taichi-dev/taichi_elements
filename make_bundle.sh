#!/bin/bash

set -e
rm -rf build/Taichi-Elements
mkdir -p build/Taichi-Elements/bundle
python3 -m pip install --no-deps -r blender/requirements.txt -t build/Taichi-Elements/bundle
cp -r blender/* build/Taichi-Elements
cp -r engine build/Taichi-Elements
rm -rf build/Taichi-Elements/bundle/include
rm -rf build/Taichi-Elements/bundle/*.dist-info
rm -rf build/Taichi-Elements/bundle/bin
rm -f build/Taichi-Elements.zip
cd build && zip -r Taichi-Elements.zip Taichi-Elements
