import struct, os, time

import numpy

import bpy


# particles format version 0
PARS_FMT_VER_0 = 0
PARS_FMT_VER_1 = 1
# particles format support versions
PARS_FMT_SUPP = (PARS_FMT_VER_0, PARS_FMT_VER_1)

# particle attributes

# position
POS = 0
# velocity
VEL = 1
# color
COL = 2
# material id
MAT = 3

# numpy attributes type
attr_types = {
    POS: numpy.float32,
    VEL: numpy.float32,
    COL: numpy.int32,
    MAT: numpy.int32
}

# attributes names
attr_names = {
    POS: 'pos',
    VEL: 'vel',
    COL: 'col',
    MAT: 'mat'
}


def write_pars_v0(par_data):
    data = bytearray()
    # particles format version
    data.extend(struct.pack('I', PARS_FMT_VER_0))
    # particles count
    pars_cnt = len(par_data[POS])
    data.extend(struct.pack('I', pars_cnt))
    print('Particles count:', pars_cnt)

    # par_i - particles index
    for par_i in range(pars_cnt):
        data.extend(struct.pack('3f', *par_data[POS][par_i]))
        data.extend(struct.pack('3f', *par_data[VEL][par_i]))
        data.extend(struct.pack('I', par_data[COL][par_i]))
        data.extend(struct.pack('I', par_data[MAT][par_i]))

    return data


def write_pars_v1(par_data, fpath, fname):
    data = bytearray()
    # particles format version
    data.extend(struct.pack('I', PARS_FMT_VER_1))
    # particles count
    pars_cnt = par_data[POS].shape[0]
    data.extend(struct.pack('I', pars_cnt))
    print('Particles count:', pars_cnt)

    for attr_id in range(4):
       fname_str = '{}_{}.bin'.format(fname, attr_names[attr_id])
       fname_byte = bytes(fname_str, 'utf-8')
       length = len(fname_byte)
       data.extend(struct.pack('I', length))
       data.extend(struct.pack('{}s'.format(length), fname_byte))

    for attr_id in range(4):
        attr_array = par_data[attr_id]
        attr_array.tofile('{}_{}.bin'.format(fpath, attr_names[attr_id]))

    return data


def read_pars_v0(data, caches, offs, folder):
    # particles positions
    pos = []
    # particles velocities
    vel = []
    # particles colors
    col = []
    # particles materials
    mat = []
    # particles count
    count = struct.unpack('I', data[offs : offs + 4])[0]
    offs += 4

    for index in range(count):
        # particle position
        p_pos = struct.unpack('3f', data[offs : offs + 12])
        offs += 12
        pos.append(p_pos)

        # particle velocity
        p_vel = struct.unpack('3f', data[offs : offs + 12])
        offs += 12
        vel.append(p_vel)

        # particle color
        p_col = struct.unpack('I', data[offs : offs + 4])[0]
        offs += 4
        col.append(p_col)

        # particle material
        p_mat = struct.unpack('I', data[offs : offs + 4])[0]
        offs += 4
        mat.append(p_mat)

    caches[folder] = {POS: pos, VEL: vel, COL: col, MAT: mat}


def read_pars_v1(data, caches, offs, folder):
    # particles count
    count = struct.unpack('I', data[offs : offs + 4])[0]
    offs += 4
    caches[folder] = {}

    for attr_id in range(4):
        file_name_len = struct.unpack('I', data[offs : offs + 4])[0]
        offs += 4

        file_name_bytes = struct.unpack('{}s'.format(file_name_len), data[offs : offs + file_name_len])[0]
        file_name = str(file_name_bytes, 'utf-8')
        offs += file_name_len

        file_path = bpy.path.abspath(os.path.join(folder, file_name))
        caches[folder][attr_id] = numpy.fromfile(file_path, dtype=attr_types[attr_id])


# read particles
def read_pars(file_path, caches, folder, attr_name):
    start_time = time.time()

    with open(file_path, 'rb') as file:
        data = file.read()
    # read offset in file
    offs = 0
    # particles format version
    ver = struct.unpack('I', data[offs : offs + 4])[0]
    offs += 4

    if not ver in PARS_FMT_SUPP:
        msg = 'Unsupported particles format version: {0}'.format(ver)
        raise BaseException(msg)

    if ver == PARS_FMT_VER_0:
        read_pars_v0(data, caches, offs, folder)
    elif ver == PARS_FMT_VER_1:
        read_pars_v1(data, caches, offs, folder)

    end_time = time.time()
    total_time = end_time - start_time
    print('read particles {}: {:.4f} seconds'.format(attr_name.lower(), total_time))
