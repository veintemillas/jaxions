#!/bin/bash


if [ ! "$SAA" ]
then 
	echo Error: Environmental variable SAA needs to be defined! 
	echo        export SAA=user@machine:/path/to/output
	exit
fi	

CURDIR=$(pwd)
echo current dir is $CURDIR

# prepare labels

if [ "$1" ]
then 
	PASS=1
	MA=$1
else
	PASS=0
fi 

if [ "$PASS" -eq 0 ]
then
NUSH=$(ls *.sh | wc -l)

 if [ "$NUSH" -eq 1 ]
 then 
 	echo anda1
	PASS=1
	MA=$(ls *.sh)
	MA=${MA%.sh}	
 fi 

 if [ "$NUSH" -gt 1 ] 
 then
        echo 'no script selected to use as label!'
        echo "which did you use?"
        ls *.sh
        exit
 fi
 if [ "$NUSH" -eq 0 ]
 then 
	echo 'no .sh script found in current directory!' 
	echo "enter a label name"
	exit
 fi 
fi

echo " " 1 - $MA selected as label! 

echo " " 2 - Locating files:
LOCFIL=$(echo $(ls $MA* 2>/dev/null ) $(ls *.txt 2>/dev/null ) $(ls axion.log.*  | tail -1))
echo "    [./]     "     $LOCFIL
if [ ! -d "./out" ]; then
   echo Error: no out/folder!
   exit
fi
#if [ -d "./$MA.sh" ]; then
	 #cp $MA out/ 
#fi  

cd $CURDIR/out
OUTFIL=$(ls *.txt 2>/dev/null)
OUTMEASFIL=$(ls axion.m.* 2>/dev/null)
echo "    [./out]  "   $OUTFIL $OUTMEASFIL

if [ ! -d $CURDIR/out/m ]; then
 echo Warning: no out/m folder!
 MFOLDER=0
else 
 MFOLDER=1
fi                     

if [ "$MFOLDER" -eq 1 ]
then 
 cd m
 if [  -d "./axion.m.10000" ]; then
	mv axion.m.10000 ../
	echo axion.m.10000 moved
	OUTMEASFIL=$(ls $CURDIR/out/axion.m.* 2>/dev/null)
 fi
 if [  -d "./axion.m.10001" ]; then
	mv axion.m.10001 ../
	echo axion.m.10001 moved
	OUTMEASFIL=$(ls $CURDIR/out/axion.m.* 2>/dev/null)
 fi	
 shopt -s extglob
 MMFIL=$(ls axion.m.+([0-9]))
 #echo $MMFIL
 PUR=($MMFIL)
 CUR=${PUR[0]}
 NUM0=${CUR#*axion.m.}
 #MEASLIST="out/m/"$CUR
 MEASLIST=$CUR
 for j in $MMFIL; do
	 if [[ $j -nt $CUR ]];  
	 then 
		# echo $j "is newer (modification date) than" $CUR "-- PASSED!"
		# size control?
		#echo $(du -k "$j" | cut -f1)
		CUR=$j
		#MEASLIST=${MEASLIST}" out/m/"$CUR
                MEASLIST=${MEASLIST}" "$CUR  
		NUM=${j#*axion.m.}           # get the number			 
	 fi
	 #NUM=${NUM#*/dens-}	# get the part after the rh- (number)
 done	 
 # echo $MEASLIST
fi

echo "    [./out/m]"    $NUM0-$NUM

echo " " 3 - Copying files: $LOCFIL to ./out/ 
cd $CURDIR
#cp $LOCFIL $CURDIR/out/
echo " " 4 - Creating symbolic link: $CURDIR/out/ "->" $CURDIR/out_$MA
#ln -s $CURDIR/out $CURDIR/out_$MA
echo " " 5 - Transfer files to maturino



AUXDIR=$CURDIR/out_$MA
mkdir $AUXDIR
mkdir $AUXDIR/m

#TRAN=""
for j in $LOCFIL; do
	#TRAN=${TRAN}" out_$MA/"$j
	ln -s $CURDIR/$j $AUXDIR/$j
done	
for j in $OUTFIL; do
        #TRAN=${TRAN}" out_$MA/"$j
	ln -s $CURDIR/out/$j $AUXDIR/$j
done
for j in $OUTMEASFIL; do
	#TRAN=${TRAN}" out_$MA/"$j
	ln -s $CURDIR/out/$j $AUXDIR/$j
done
echo $MEASLIST
for j in $MEASLIST; do
	#TRAN=${TRAN}" out_$MA/"$j
	ln -s $CURDIR/out/m/$j $AUXDIR/m/$j
done


scp -r $AUXDIR $SAA
rm -r $AUXDIR










