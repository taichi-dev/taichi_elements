import sys
import os
from datetime import datetime

demo_dir = os.path.dirname(os.path.abspath(__file__))
taichi_elements_path = os.path.dirname(demo_dir)
sys.path.append(taichi_elements_path)


def create_output_folder(prefix):
    folder = prefix + '_' + datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    os.mkdir(folder)
    return folder
