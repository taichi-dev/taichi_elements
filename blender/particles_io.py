import struct


# particles format version 0
PARS_FMT_VER_0 = 0
# particles format support versions
PARS_FMT_SUPP = (PARS_FMT_VER_0, )

# particle attributes

# position
POS = 0
# velocity
VEL = 1
# color
COL = 2
# material id
MAT = 3


def write_pars_v0(par_data):
    data = bytearray()
    # particles format version
    data.extend(struct.pack('I', 0))
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
        pos.extend(p_pos)

        # particle velocity
        p_vel = struct.unpack('3f', data[offs : offs + 12])
        offs += 12
        vel.extend(p_vel)

        # particle color
        p_col = struct.unpack('I', data[offs : offs + 4])[0]
        offs += 4
        col.append(p_col)

        # particle material
        p_mat = struct.unpack('I', data[offs : offs + 4])[0]
        offs += 4
        mat.append(p_mat)

    caches[folder] = {POS: pos, VEL: vel, COL: col, MAT: mat}


# read particles
def read_pars(data, caches, folder):
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
