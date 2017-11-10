echo " " h5cpBNS Copies bins/ nSpectrum/ pSpectrum/ from 1st to 2nd input
echo " " from axion.m."$1" to "$2"
read -r -p ' Are you bloody sure?? [y/n]:' CONT
if [[ "$CONT" =~ ^([yY][eE][sS]|[yY])+$ ]]; then
 echo '   your call!'
else
  exit 1
fi

h5copy -i axion.m."$1" -o axion.m."$2" -s /bins -d /bins
h5copy -i axion.m."$1" -o axion.m."$2" -s /nSpectrum -d /nSpectrum
h5copy -i axion.m."$1" -o axion.m."$2" -s /pSpectrum -d /pSpectrum
echo " done!"

echo " have a look at h5ls axion.m."$2"
h5ls axion.m."$2"
echo " and inside ... "
h5ls axion.m."$2"/nSpectrum
