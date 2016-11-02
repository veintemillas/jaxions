#!/bin/bash
for j in $(ls out/at/at-*.txt); do
read  dum nnn lll ddd zzz < $j
#echo $nnn
#echo $lll
#echo $ddd
NAME=${j%.txt}  			# get the part before the colon
NAME=${NAME#*/at-}		# get the part after the at- (number)
FILE="ploter.gnu"
/bin/cat <<EOM > $FILE
set title "(N=$nnn, prec=?) z=$zzz" offset 0,-1
set terminal png medium size 1200,1200
set xrange [0:$lll]
set yrange [0:$lll]
set size 1,1
#set pm3d
set cbrange [-pi:pi]
set zrange [-pi:pi]
unset surface
set palette defined ( 0 1 1 1, 1 1 1 1, 3 1 0 0, 4 0 0 0, 5 0 0 0, 6 0 0 1, 8 1 1 1, 9 1 1 1)
set view map
set key outside
set output 'out/plots/at/at-$NAME.png'
plot '$j' u (\$1*$ddd):(\$2*$ddd):3 matrix with image notitle
#plot '$j' binary array=${nnn}x$nnn dx=$ddd dy=$ddd format="%float" with image notitle
EOM
gnuplot 'ploter.gnu'
done
ffmpeg -nostats -loglevel 0 -r 3 -i out/plots/at/at-%05d.png -c:v libx264 -r 30 -pix_fmt yuv420p out/at-movie.mp4
