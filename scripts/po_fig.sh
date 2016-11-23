#!/bin/bash
FILE="ploter5.gnu"
/bin/cat <<EOM > $FILE
#set title "(N=$nnn, prec=?) z=$zzz" offset 0,-1
set terminal png medium size 1200,1200
set yrange [0:1.2]
set size 1,1
set output 'out/sample.png'
plot 'out/sample.txt' u 1:2 with lines
EOM
gnuplot 'ploter5.gnu'
