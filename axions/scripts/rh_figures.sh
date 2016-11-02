#!/bin/bash
for j in $(ls out/rho/rho-*.txt); do
read  dum nnn lll ddd zzz < $j
NAME=${j%.txt}  			# get the part before the colon
NAME=${NAME#*/rho-}		# get the part after the rh- (number)
FILE="ploter2.gnu"
/bin/cat <<EOM > $FILE
set title "(N=$nnn, prec=?) z=$zzz" offset 0,-1
set terminal png medium size 1200,1200
set xrange [0:$lll]
set yrange [0:$lll]
set size 1,1
#set pm3d
set cbrange [0:1.2]
set zrange [0:1.2]
unset surface
set palette defined ( 0 1 1 1, 1 1 0 0, 2 1 0 0, 3 1 0 0, 4 1 1 0, 5 0 0 0, 6 0 1 1)
set view map
set key outside
set output 'out/plots/rho/rho-$NAME.png'
plot '$j' u (\$1*$ddd):(\$2*$ddd):3 matrix with image notitle
#plot '$j' binary array=${nnn}x$nnn dx=$ddd dy=$ddd format="%float" with image notitle
EOM
gnuplot 'ploter2.gnu'
done
ffmpeg -nostats -loglevel 0 -r 5 -i out/plots/rho/rho-%05d.png -c:v libx264 -r 30 -pix_fmt yuv420p out/rh-movie.mp4
