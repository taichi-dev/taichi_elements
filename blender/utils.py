import bpy


def get_frame_info():
    scn = bpy.context.scene
    frm_strt = scn.elements_frame_start
    frm_end = scn.elements_frame_end
    return frm_strt, frm_end
