import numpy as np
import matplotlib.pyplot as plt
#from __future__ import print_function

from sklearn.cluster import DBSCAN
from sklearn import metrics

from random import randint
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import pdist

import time
import sys
import h5py

from math import pow
from math import ceil

from numba import jit

#for plots
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph.opengl as gl
import pyqtgraph as pg
import numpy as np


print('')
print('')

#### PREAMBLE FUNCTIONS

# UNION functio function
# function

def sex (labels,labels2):

   npo = len(labels)

   # 1. create auxilliary label arrays
   # conventions: points already visited are -2 in labels11, labels22
   #              points not visited yet are -2 in newlabels
   newlabels = np.zeros(npo,dtype=int) -2
   labels11 = np.zeros(npo,dtype=int)
   labels22 = np.zeros(npo,dtype=int)

   labels11[:] = labels
   labels22[:] = labels2

   # account for all points which are marked as noise in both datasets
   parla1 = (labels  == -1)
   parla2 = (labels2 == -1)

   newlabels[parla1*parla2]   = -1
   labels11[parla1*parla2] = -2
   labels22[parla1*parla2] = -2

   # 2. create a list with the first index at which a label appears
   m1 = max(labels)
   jta = np.arange(0,npo,dtype=int)

   po1 = []
   for j in set(labels):
       if j > -1:
           parla1 = (labels  == j)
           po1.append(jta[parla1][0])
   po1 = np.array(po1)

   # 3. loop over all clusters found in set1
   dicta = []
   for i in set(labels):
       if i > -1:
           # reference point
           posi = po1[i]
           ll = labels[posi]

           # get the cluster of this point in both labeling systems
           parla1 = (labels  == ll)
           parla2 = (labels2 == labels2[posi])

           # if the point labels two clusters, group them
           if labels2[posi] > -1 :
               mask = np.logical_or(parla1,parla2)
               seta = np.array(list(set(newlabels[mask])))
               seta = seta[seta > -1]

               # actually some of the points found may already be part of a cluster
               # account for all points in that cluster
               # changed from Javiers original code
               # if len(seta) > 1:
               if len(seta) > 0:
                   for lset in seta:
                       minimask = (newlabels == lset)
                       mask = np.logical_or(mask,minimask)
                   ll = min(seta)

               newlabels[mask] = ll
               labels11[mask] = -2
               labels22[mask] = -2

           # if the point belongs to a mc in case 1 but not in case 2:
           # only take the points from case 1 & edit in newlabels
           # the opposite case is accounted for below
           elif labels2[posi] == -1 :
               mask = parla1
               newlabels[mask] = ll
               labels11[mask] = -2

           # store information about which labels have been grouped together
           dicta.append([labels[posi],labels2[posi],ll])


   maxrelab = max(newlabels)

   # 4. account for points which are noise in set1 but belong to a cluster in set2
   rela2 = np.array(list(set(labels22)))

   counter = 0

   for j in rela2:
       if j > -1 :
           # must anypoint of this MC need be asssotiated with a mc already present in newlabels?
           # changed from Javiers initial code
           # asolabel = np.array(list(set(labels2[parla22])))
           # parla22 = (labels22  == j)
           parla22 = (labels2 == j)
           asolabel = np.array(list(set(newlabels[parla22])))
           asolabel = asolabel[asolabel > -1]

           ll2 = j

           # changed from Javiers original code:
           #if asolabel == []
           if not asolabel:
               newlabels[parla22] = maxrelab + counter
               counter +=1
               dicta.append([-1,ll2, maxrelab + counter])

           elif len(asolabel) > 0:
               lala = min(asolabel)

               for l in asolabel:
                   para = (newlabels == l)
                   newlabels[para] = lala
                   dicta.append([-1,j, lala])
               newlabels[parla22] = lala

           labels11[parla22] = -2
           labels22[parla22] = -2

   print('crosscheck: # of points not visited by the function: %d' %(sum(newlabels==-2)))
   print('final number of clusters found: %d' %(len(set(newlabels))))
   return newlabels #, list(dicta)


## RELABEL ACCORDING TO MASS -> masslabels
## RELABEL ACCORDING TO #POINTS -> numblabels

def getMass(it):
    return it[1]

def getNumb(it):
    return it[2]

def relabelMN(newlabels,den):

    minilabels = np.array(list(set(newlabels)))

    poso = []

    #cocha = np.empty_like(minilabels)
    for l in minilabels:
        if l > -1:
            class_member_mask = (newlabels == l)
            poso.append( [l ,np.add.reduce(den[class_member_mask]), np.add.reduce(class_member_mask)] )
    # mass = <avdensity> (delta=sizeL/n)^3
    # cube = (sizeL/n)**3

    oroM = sorted(poso,key=getMass)
    oroN = sorted(poso,key=getNumb)

    numberoflabels = len(oroM)

    # relabel
    masslabels = np.empty_like(newlabels)
    numblabels = np.empty_like(newlabels)

    # noise
    masslabels[:] = newlabels[:]
    numblabels[:] = newlabels[:]

    counter = 0

    for i in range(0,numberoflabels):
        para = (newlabels == oroM[i][0])
        masslabels[para] = numberoflabels-i-1

    for i in range(0,numberoflabels):
        para = (newlabels == oroN[i][0])
        numblabels[para] = numberoflabels-i-1

    print('max mass ',oroM[-1])
    print('max numb ',oroN[-1])

    return masslabels, numblabels

# function to create color list
def createmycolorlist(n):
    p = int(n**(1/3.)+1)
    p2=p*p
    p3=p**3
    list= []
    for i in range(0,n):
        x = i//p2
        y = (i - x*p2)//p
        z = i%p
        list.append([x,y,z,0.5])
    return list

############# CODE ################################################


#parameters for DBSCAN
eps=2 #was 5
min_samples=20
algorithm='kd_tree'
n_jobs=-1

# get from input
#dens_th = 15
#data_file = 'axion.r.00046'

if len(sys.argv) > 1:
    f         = h5py.File('./' + sys.argv[1], "r")

    if 'energy/density' in f:
        print('Reading ',sys.argv[1])
        n     = f.attrs[u'Size']
        n2 = n*n
        n3 = n2*n
        data = f['energy/density'].value.reshape(n3)
    elif 'energy/redensity' in f:
        print('Reading reducedCon from ',sys.argv[1])
        n =256
        n2 = n*n
        n3 = n2*n
        data = f['energy/redensity'].value.reshape(n3)
    else:
        sys.exit()


    sizeL = f.attrs[u'Physical size']




    maxi = max(data)
    print('max contrast = %.2f' %(maxi))

if len(sys.argv) < 3:
    print('insuficient arguments!')
    print('enter a h5 axion.x.XXXXX file name and a density contrast')
    sys.exit()


dens_th   = float(sys.argv[2])
print('proceeding with contrast threshold = ',dens_th)



# thresholded data
mask = ( data > dens_th)


# basicstuff
start = time.time()

bas = np.zeros((n,n,n),dtype=float)
linap = np.arange(0,n,dtype=float).reshape((n,1,1))
x_pos = bas + linap
y_pos = bas + linap.reshape((1,n,1))
z_pos = bas + linap.reshape((1,1,n))
del bas, linap
x_pos = np.reshape(x_pos,(n3,))
y_pos = np.reshape(y_pos,(n3,))
z_pos = np.reshape(z_pos,(n3,))

# position np.array
npo = len(data[mask])
pos = np.empty((npo,3))
pos[:,0] = x_pos[mask]
pos[:,1] = y_pos[mask]
pos[:,2] = z_pos[mask]

# density numpy
den = data[mask]
maxi = max(den)

end = time.time()
print('building took %1.3f s' %(end-start))

# clustering #1
print()
print('clustering ...' , end=' ')
start = time.time()
db = DBSCAN(eps=eps, min_samples=min_samples, algorithm=algorithm, n_jobs=n_jobs).fit(pos)
labels = np.array(db.labels_)
end = time.time()

Nmcs = len(set(labels))-1
print('clustering of the point set [%d points] took %1.3f s' %(npo, end-start))
print('%d miniclusters found' %(Nmcs))
mc_mask = (labels > -1)
print('%d points rejected'%(npo-len(pos[mc_mask])))
print()

# shift coordinates of the grid by n/2 in Z and make mod[256]
print('clustering once more...' , end=' ')
nh = n//2
pos2 = (pos + [nh,nh,nh])%n
start = time.time()
db2 = DBSCAN(eps=eps, min_samples=min_samples, algorithm=algorithm, n_jobs=n_jobs).fit(pos2)
labels2 = np.array(db2.labels_)
end = time.time()
print('clustering n/2-shifted took %1.3f s' %(end-start))

Nmcs2 = len(set(labels2))-1
print('clustering of the point set [%d points] took %1.3f s' %(npo, end-start))
print('%d miniclusters found' %(Nmcs2))
mc_mask = (labels2 > -1)
print('%d points rejected'%(npo-len(pos[mc_mask])))
print()

# joining sets
print('joining sets ... ', end=' ')
newlabels = sex(labels,labels2)
Nmcs = len(set(newlabels[newlabels > -1]))
print('%d MCs found' %(Nmcs))


masslabels, numblabels = relabelMN(newlabels,den)
print('relabeling per mass/number [%d, %d]'%(len(set(masslabels[masslabels > -1])),len(set(numblabels[numblabels > -1]))))

# make a nice 3d plot
print('plotting')



unique_labels = set(masslabels)
#print('unique mass labels ',unique_labels)
primcol=np.array(createmycolorlist(Nmcs))
mc_mask = (masslabels > -1)
cola = primcol[masslabels[ mc_mask ]]
# cola[:,0] *= den[mc_mask]/maxi
# cola[:,1] *= den[mc_mask]/maxi
# cola[:,2] *= den[mc_mask]/maxi
cola[:,3] *= den[mc_mask]/maxi

# make a nice plot of the clusters
# check also isocontours!

poso = (pos[mc_mask]-(n/2))
phase = 0.

step  = 1
tStep = 100
pause = False
timer = pg.QtCore.QTimer()

app = QtGui.QApplication([])
w = gl.GLViewWidget()
w.opts['distance'] = 2*n
w.show()
w.setWindowTitle('pyqtgraph example: GLScatterPlotItem')

xGrid = gl.GLGridItem()
yGrid = gl.GLGridItem()
zGrid = gl.GLGridItem()

w.addItem(xGrid)
w.addItem(yGrid)
w.addItem(zGrid)

xGrid.rotate(90, 0, 1, 0)
yGrid.rotate(90, 1, 0, 0)

nh = n/2
xGrid.translate(-1.0*nh, 0, 0)
yGrid.translate(0, -1.0*nh, 0)
zGrid.translate(0, 0, -1.0*nh)

xGrid.scale(0.1*nh, 0.1*nh, 0.1*nh)
yGrid.scale(0.1*nh, 0.1*nh, 0.1*nh)
zGrid.scale(0.1*nh, 0.1*nh, 0.1*nh)

sp1 = gl.GLScatterPlotItem(pos=poso, size=1.0, color=cola, pxMode=False)
w.addItem(sp1)

def update():
    ## update s
    global phase, sp1, poso
    poso = (pos[mc_mask] + [1.*phase,0.5*phase,0.33*phase])%n -(n/2)
    sp1.setData(pos=poso)
    phase += 1.

timer.timeout.connect(update)
timer.start(50)

# baseKeyPress = w.keyPressEvent


# def	keyPressEvent(event):
# 	key = event.key()
#
# 	baseKeyPress(event)
#
# 	if key == QtCore.Qt.Key_Space :
# 		if pause:
# 			pause = False
# 			timer.start(50)
# 		else:
# 			pause = True
# 			timer.stop()

	# elif key == QtCore.Qt.Key_M:
	# 	self.tStep = tStep + 10
	# 	if self.tStep > 1500:
	# 		self.tStep = 1500
	# 	self.timer.setInterval(self.tStep)
	# elif key == QtCore.Qt.Key_N:
	# 	self.tStep = self.tStep - 10
	# 	if self.tStep < 20:
	# 		self.tStep = 20
	# 	self.timer.setInterval(self.tStep)
	# elif key == QtCore.Qt.Key_R:
	# 	self.step = -self.step

# t = QtCore.QTimer()



## Start Qt event loop unless running in interactive mode.
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
