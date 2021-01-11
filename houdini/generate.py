from plyfile import PlyData, PlyElement
import numpy as np

with open('f.ply', 'wb') as f:
    num_particles = 10000000
    header = f"""ply
format binary_little_endian 1.0
comment Created by taichi
element vertex {num_particles}
property float x
property float y
property float z
end_header
"""
    f.write(str.encode(header))
    
    pos = np.random.randn(num_particles, 3).astype(np.float32)
    
    f.write(pos.tobytes())
    
    

