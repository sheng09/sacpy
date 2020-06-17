#!/usr/bin/env python3

from sacpy.tomo import plot_tomo as PT
from mayavi import mlab
##
fnm = 'earth1d.txt'
PT.plot_absolute_earth_model(fnm)
mlab.view(0.0, 55.0 )
mlab.show()


