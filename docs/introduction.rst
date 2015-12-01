Introduction
==================================

**IMPORTANT NOTE**: This code does not function in its current state.  When I put things up on github, I decided that the code could use a little cleaning up and it ended up being a major reorganization.  The code is about 90% of the way through that reorganization.  There's a little more cleaning up of documentation/formatting all the Sphinx stuff and I need to debug and accuracy check the AngularCatalog class, but that's it.

Py2PAC: Python 2-Point Angular Correlations
-------------------------------------

This package was developed to calculate the galaxy angular two-point correlation functions on actual data.  It's under development and currently only contains the calculation of the angular correlation functions within a region of arbitrary outline with one depth.  The correlation functions may be calculated with error estimates via single galaxy bootstrapping, block bootstrapping, or jackknifing. 

Current Status
^^^^^^^^^^^

Everything is pretty much ready for use except for the AngularCatalog class, which needs all the changes to the organization propagated to it and then needs to be checked for accuracy.  There are some more things that I want to do with the ImageMask class and a lot of fitting and post-processing that I want to put in, but nothing that is necessary for the basic functionality, so it's just debugging that's left.

To-do before release
^^^^^^^^^^^^^^^^

* Finish changes to AngularCatalog
* Check all routines in AngularCatalog
* Check the jackknife error bar magnitude
* Come up with example data sets and add to repository
* Tutorial ipython notebook

To-do eventually
^^^^^^^^^^^^^^

* Set up an ImageMask classmethod to deal with rotated boxes (currently just a range in RA and Dec)
* Next feature: fitting correlation functions
* After that, multi-catalogs that take a catalog and slice by property into different AngularCatalogs
* Make hand-made WCS instances (in ImageMask.from\_array and ImageMask.from\_ranges) moveable on the sky
* Parallel processing for CF calculation
