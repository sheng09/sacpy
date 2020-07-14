#!/usr/bin/env bash

###
#  generate Earth model
###

/home/catfly/Program_Sheng/sacpy/tomo/cxx/fw_generate_mod3d.x --dlon=2.0 --dlat=2.0 \
    --model_o=01_workspace/earth1d.mod.txt --model_o_mod=p \
    --grd_o=01_workspace/earth1d.grd.txt --grd_o_mod=a


/home/catfly/Program_Sheng/sacpy/tomo/cxx/fw_generate_mod3d.x --dlon=2.0 --dlat=2.0 \
    --model_o=01_workspace/plume.mod.txt --model_o_mod=p \
    --grd_o=01_workspace/plume.grd.txt --grd_o_mod=d \
    --plume=30.0/0.0/10.0/660/2850/1.00/0.98 \
    --smooth=3.0
