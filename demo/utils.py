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


class Tee(object):
    def __init__(self, fn, mode):
        self.file = open(fn, mode)
        self.stdout = sys.stdout
        sys.stdout = self

    def __del__(self):
        sys.stdout = self.stdout
        self.file.close()

    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)

    def flush(self):
        self.file.flush()
