#!/bin/bash
FILE="ploter5.gnu"
/bin/cat <<EOM > $FILE
#set title "(N=$nnn, prec=?) z=$zzz" offset 0,-1
set term postscript eps enhanced color
set size 1,1
set output '|ps2pdf -dEPSCrop - out/sample.pdf'
plot 'out/sample.txt' u 1:(sqrt(\$2*\$2+\$3*\$3)/\$1) with lines lt -1 lw 1 title 'rho'
EOM
gnuplot 'ploter5.gnu'
rm 'ploter5.gnu'
