#My attempt at a setup... I have no idea how this is supposed to work.

from setuptools import setup

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
          ],
      )
