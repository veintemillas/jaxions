echo $1
#%%%%%%%%%%%%%%%%%%%%%%%%# generates plots and this file
cd out/
#ipython autoplot.py
#ipython pspecevol.py all
#ipython specevol.py all
#ipython contrastevol.py all
#ipython thetaBINevol.py all
pdflatex --jobname $1 "\def\simlabel{$1.sh} \input{jaxions.tex}" 
mv $1.pdf ../pdfs/
rm $1.*
cd ..
cp $1.sh scripts/

