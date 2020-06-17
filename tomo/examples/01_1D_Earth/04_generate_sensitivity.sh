#!/usr/bin/env bash

../../cxx/fw_traveltime_sensitivity.x --model3d=p/01_workspace/earth1d.mod.txt --raypath=03_workspace/raypath.h5 --verbose --output_raypath_segments=04_workspace/sens.h5
