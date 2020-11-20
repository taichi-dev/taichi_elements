import sys
import os

tests_dir = os.path.dirname(os.path.abspath(__file__))
taichi_elements_path = os.path.dirname(tests_dir)
sys.path.append(taichi_elements_path)
