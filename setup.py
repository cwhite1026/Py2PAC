#My attempt at a setup... I have no idea how this is supposed to work.

from setuptools import setup
import os
import sys
from mock import Mock as MagicMock
    
class Mock(MagicMock):
    @classmethod
    def __getattr__(cls, name):
            return Mock()
        
MOCK_MODULES = ['numpy', 'scipy', 'scikit-learn', 'astropy', 'copy', 'time', 'matplotlib']
sys.modules.update((mod_name, Mock()) for mod_name in MOCK_MODULES)
        
def readme():
    with open('README.md') as f:
        return f.read()

setup(name='py2pac',
      version='0.1',
      description='Python package for computing 2-point angular correlation functions',
      url='https://github.com/cwhite1026/Py2PAC',
      author='Catherine White',
      author_email='ccavigl1@jh.edu',
      packages=['py2pac'],
      )
