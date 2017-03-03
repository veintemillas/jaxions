#!/bin/bash
for j in $(ls out/con/con-*.txt); do
read  dum nnn lll ddd zzz < $j
NAME=${j%.txt}  			# get the part before the colon
NAME=${NAME#*/con-}		# get the part after the rh- (number)
FILE="ploter8.gnu"
/bin/cat <<EOM > $FILE
set title "(N=$nnn, prec=?) z=$zzz" font "Helvetica,20" offset -50,-10 tc lt 1
set object 1 rectangle from screen 0,0 to screen 1,1 fillcolor rgb"black" behind
set palette rgb 33,13,10;
set border lw 1 lc rgb "white"
set tics font "Helvetica,12"
set origin -0.04,-0.05
set size 1.1,1.1
set terminal png medium size 1200,1200
set xrange [0:$lll]
set yrange [0:$lll]
set zrange [0:$lll]
set ticslevel 0
set output 'out/plots/con/con-$NAME.png'
set key off
unset colorbox
splot '$j' u (\$1*$ddd):(\$2*$ddd):(\$3*$ddd):(\$4) with points palette pointsize 0.2 pointtype 7 notitle
EOM
gnuplot 'ploter8.gnu'
#echo $j printed
done

for j in $(ls out/con/con-*.txt); do
  NAME=${j%.txt}  			# get the part before the colon
  NAME=${NAME#*/con-}		# get the part after the con- (number)
break
done

ffmpeg -nostats -loglevel 0 -r 3 -start_number $NAME -i out/plots/con/con-%05d.png -c:v libx264 -r 30 -pix_fmt yuv420p out/con-movie.mp4
