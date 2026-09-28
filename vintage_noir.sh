#!/bin/bash

in="$1"
base="${in%.*}"        # strip extension
out="${base}_noir_vintage.mp4"

ffmpeg -i $in -vf "
format=gray,
drawgrid=w=0:h=2:color=black@0.12,
noise=alls=12:allf=t
" -c:v libx264 -crf 18 $out

