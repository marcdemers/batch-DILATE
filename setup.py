import io
import os
from distutils.core import setup

from setuptools import find_packages

dir = os.path.dirname(__file__)

with io.open(os.path.join(dir, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(name='batchdilate',
      version='0.2',
      description='An extension of the differentiable DILATE loss for multi-step timeseries predictions.',
      long_description=long_description,
      long_description_content_type='text/markdown',
      url='https://github.com/marcdemers/batch-DILATE',
      author='Marc Demers',
      author_email='demers.marc@gmail.com',
      license='MIT',
      classifiers=[
          'Intended Audience :: Science/Research',
          'Topic :: Scientific/Engineering',
          'Topic :: Scientific/Engineering :: Mathematics',
          'Programming Language :: Python :: 3.3',
          'Programming Language :: Python :: 3.4',
          'Programming Language :: Python :: 3.5',
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7',
          'Programming Language :: Python :: Implementation',
      ],
      install_requires=['numpy>=1.17', 'numba>=0.47.0', 'torch>=1.1'],
      packages=find_packages()
      )
