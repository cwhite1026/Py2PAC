#My attempt at a setup... I have no idea how this is supposed to work.

from setuptools import setup
import os
import sys

#Yoinked from HMF: https://github.com/steven-murray/hmf/blob/master/setup.py
class Mock(object):
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return Mock()

    @classmethod
    def __getattr__(cls, name):
        if name in ('__file__', '__path__'):
            return '/dev/null'
        elif name[0] == name[0].upper():
            mockType = type(name, (), {})
            mockType.__module__ = __name__
            return mockType
        else:
            return Mock()

MOCK_MODULES = ['numpy', 'scipy', 'scikit-learn', 'astropy', 'copy', 'time', 'matplotlib']
for mod_name in MOCK_MODULES:
    sys.modules[mod_name] = Mock()
#End yoinking

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
      install_requires=[
          'numpy',
          'scipy',
          'scikit-learn',
          'astropy',
          ],
      )
