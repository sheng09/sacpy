#!/usr/bin/env python3
import numpy as np
from mayavi import mlab

n_step = 360
f = mlab.gcf()
scene = f.scene
camera = f.scene.camera
da = 360.0/n_step
for i in range(n_step):
    camera.azimuth(da)
    scene.reset_zoom()
    scene.render()
    mlab.savefig('pics/anim%03d.png' % i, size=(900,1200) )