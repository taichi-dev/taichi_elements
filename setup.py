project_name = 'taichi_elements'

version = '0.0.1'
import setuptools

classifiers = [
    'Development Status :: 2 - Pre-Alpha',
    'Topic :: Multimedia :: Graphics',
    'Topic :: Games/Entertainment :: Simulation',
    'Intended Audience :: Science/Research',
    'Intended Audience :: Developers',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
]

packages = setuptools.find_packages()

setuptools.setup(name=project_name,
                 packages=packages,
                 version=version,
                 description='Taichi Elements',
                 author='Taichi Elements Developers',
                 author_email='yuanmhu@gmail.com',
                 url='https://github.com/taichi-dev/taichi_elements',
                 install_requires=[
                     'taichi>=0.6.27',
                 ],
                 keywords=['graphics', 'simulation'],
                 license='MIT',
                 include_package_data=True,
                 classifiers=classifiers)
