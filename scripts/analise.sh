echo $1
#%%%%%%%%%%%%%%%%%%%%%%%%# generates plots and this file
read -r -p 'generate pdfs [y/n]:' CONT
if [[ "$CONT" =~ ^([yY][eE][sS]|[yY])+$ ]]; then
 echo '  your call!'
 ipython summaryplot.py $1
 #ipython autoplot.py
 ipython pspecevol.py every10
 ipython specevol.py every10
 ipython contrastevol.py every10
 ipython thetaBINevol.py every10
else
  echo '  menos curro!'
fi
pdflatex --jobname $1 "\def\simlabel{$1.sh} \input{jaxions.tex}"
rm $1.aux $1.log
#mv $1.pdf ../pdfs/
#rm $1.aux $1.log
