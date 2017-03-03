#!/bin/bash
for j in $(ls out/dens/dens-*.txt); do
read  dum nnn lll ddd zzz mode < $j
NAME=${j%.txt}  			# get the part before the colon
NAME=${NAME#*/dens-}		# get the part after the rh- (number)
FILE="ploter4.gnu"
/bin/cat <<EOM > $FILE
set title "(Logdens N=$nnn, mode=$mode) z=$zzz" offset 0,-1
set terminal png medium size 1200,1200
set xrange [0:$lll]
set yrange [0:$lll]
set size 1,1
#set pm3d
unset surface
set palette defined ( 0 0 0 0, 1 1 1 1 )
set view map
set key outside
set output 'out/plots/dens/dens-$NAME.png'
plot '$j' u (\$1*$ddd):(\$2*$ddd):(log10(\$3)) matrix with image notitle
#plot '$j' binary array=${nnn}x$nnn dx=$ddd dy=$ddd format="%float" with image notitle
EOM
gnuplot 'ploter4.gnu'
rm 'ploter4.gnu'
done
ffmpeg -nostats -loglevel 0 -r 5 -i out/plots/dens/dens-%05d.png -c:v libx264 -r 30 -pix_fmt yuv420p out/logdens-movie.mp4
