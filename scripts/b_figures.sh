#!/bin/bash
for j in $(ls out/at/at-*.txt); do
read  dum nnn lll ddd zzz mode < $j
#echo $nnn
#echo $lll
#echo $ddd
NAME=${j%.txt}  			# get the part before the colon
NAME=${NAME#*/at-}		# get the part after the at- (number)
FILE="ploter9.gnu"
/bin/cat <<EOM > $FILE
set title "(N=$nnn, mode=$mode) z=$zzz" offset 0,-1
set terminal png medium size 1200,1200
set xrange [0:$lll]
set yrange [0:$lll]
set size 1,1
#set pm3d
unset surface
set palette defined ( 0 0 0 0, 1 1 1 1 )
set view map
set key outside
set output 'out/plots/con/con-$NAME.png'
plot '$j' u (\$1*$ddd):(\$2*$ddd):3 matrix with image notitle
#plot '$j' binary array=${nnn}x$nnn dx=$ddd dy=$ddd format="%float" with image notitle
EOM
gnuplot 'ploter9.gnu'
rm 'ploter9.gnu'
done
