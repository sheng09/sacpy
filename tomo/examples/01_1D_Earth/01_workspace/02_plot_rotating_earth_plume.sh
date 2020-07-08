#!/usr/bin/env bash

###
#  generate pngs
###
# mayavi2 -x plot_mod_plume.py -x movie.py -o


###
#  convert to mp4
###
rm pics_plume/anim000.png -rf
convert -delay 3 -loop 0 pics_plume/anim*.png plume.mp4
