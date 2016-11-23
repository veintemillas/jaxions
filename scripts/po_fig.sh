#!/bin/bash
FILE="ploter5.gnu"
/bin/cat <<EOM > $FILE
#set title "(N=$nnn, prec=?) z=$zzz" offset 0,-1
set terminal png medium size 600,600
set size 1,1
set output 'out/sample.png'
plot 'out/sample.txt' u 1:(\$2*\$2+\$3*\$3) with lines lt -1 lw 1 title 'rho'
EOM
gnuplot 'ploter5.gnu'
