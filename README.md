# README #


## Py2PAC: Python 2-Point Angular Correlations  ##

This package was developed to calculate the galaxy angular two-point correlation functions on actual data.  It's under development and currently only contains the calculation of the angular correlation functions within a region of arbitrary outline with one depth.  The correlation functions may be calculated with error estimates via single galaxy bootstrapping, block bootstrapping, or jackknifing.

Documentation: [https://py2pac.readthedocs.io/en/latest/](https://py2pac.readthedocs.io/en/latest/)

### Current Status ###

Everything is ready for use at the most basic level of functionality: calculating correlations with error estimates.  There are some more things that I want to do with the ImageMask class and a lot of fitting and post-processing that I want to put in, but nothing that is necessary for the basic functionality.

### To-do before release ###

- Get the setup straightened out (make sure it knows all the requirements, actually sets up properly, etc)
- Come up with example data sets and add to repository
- Tutorial ipython notebook that walks you from importing py2pac to plotting each kind of CF.  Also make sure it has the different ways to create ImageMasks

### To-do eventually ###

- Set up an ImageMask classmethod to deal with rotated boxes
- Incorporate variable completeness into the ImageMasks
- Fitting things
- After that, it's multi-bin catalogs.  
- Parallel processing for CF calculation?
- Fixing the god-forsaken hand-made WCS instances
