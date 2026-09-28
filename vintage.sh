#!/bin/bash

in="$1"
base="${in%.*}"        # strip extension
out="${base}_vintage.mp4"

ffmpeg -i "$in" -vf "
format=yuv420p,
geq=lum='lum(X,Y)':cb='(cb(X,Y)+cb(X+1,Y)+cb(X,Y+1)+cb(X+1,Y+1))/4':cr='(cr(X,Y)+cr(X+1,Y)+cr(X,Y+1)+cr(X+1,Y+1))/4',
drawgrid=w=0:h=2:color=black@0.15
" -c:v libx264 -crf 18 "$out"


