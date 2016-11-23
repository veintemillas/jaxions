#!/bin/bash
FILE="ploter6.gnu"
/bin/cat <<EOM > $FILE
#set title "(N=$nnn, prec=?) z=$zzz" offset 0,-1
set key textcolor rgb "white"
set terminal png medium size 600,600
set object 1 rectangle from screen 0,0 to screen 1,1 fillcolor rgb"black" behind
set palette rgb 33,13,10;
set border lw 1 lc rgb "white"
set tics font "Helvetica,12"
set size 1,1
set output 'out/energy.png'
plot 'out/energy.txt' u 1:3 with lines lt -1 lw 1 lc rgb "red" title 'Va',\
'out/energy.txt' u 1:5 with lines lt -1 lw 1 lc rgb "orange" title 'Ka',\
'out/energy.txt' u 1:9 with lines lt -1 lw 1 lc rgb "yellow" title 'Gax',\
'out/energy.txt' u 1:10 with lines lt -1 lw 1 lc rgb "yellow" title 'Gay',\
'out/energy.txt' u 1:11 with lines lt -1 lw 1 lc rgb "yellow" title 'Gaz',\
'out/energy.txt' u 1:4 with lines lt -1 lw 1 lc rgb "white" title 'Krho',\
'out/energy.txt' u 1:6 with lines lt -1 lw 1 lc rgb "blue" title 'Grhox',\
'out/energy.txt' u 1:7 with lines lt -1 lw 1 lc rgb "blue" title 'Grhoy',\
'out/energy.txt' u 1:8 with lines lt -1 lw 1 lc rgb "blue" title 'Grhoz',\
'out/energy.txt' u 1:2 with lines lt -1 lw 1 lc rgb "white" title 'Vrho'
EOM
gnuplot 'ploter6.gnu'
rm 'ploter6.gnu'
