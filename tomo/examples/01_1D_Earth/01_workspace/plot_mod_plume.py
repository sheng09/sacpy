#!/usr/bin/env python3

from sacpy.tomo import plot_tomo as PT
from mayavi import mlab
##
fnm = 'plume.grd.txt'
PT.plot_relative_earth_model(fnm, -0.05, 0.0, 0.01, 'dvs')
mlab.view(0.0, 55.0 )
# mlab.show()


