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

    #aux label matrices
    newlabels = np.zeros(npo,dtype=int) -2
    labels11 = np.zeros(npo,dtype=int)
    labels22 = np.zeros(npo,dtype=int)

    labels11[:] = labels
    labels22[:] = labels2


    # list of 1st points
    m1 = max(labels)
    jta = np.arange(0,npo,dtype=int)

    po1 = []
    for j in set(labels):
        if j > -1:
            parla1 = (labels  == j)
            po1.append(jta[parla1][0])
    po1 = np.array(po1)

    # list of 1st points
    m2 = max(labels2)

#     po2 = []
#     for j in range(0,m2):
#         parla2 = (labels2  == j)
#         po2.append(jta[parla2][0])
#     po2 = np.array(po2)

    # NOISE intersection
    parla1 = (labels  == -1)
    parla2 = (labels2 == -1)

    newlabels[parla1*parla2]   = -1
    labels11[parla1*parla2] = -2
    labels22[parla1*parla2] = -2

    dicta = []
    # clusters UNION
    for i in set(labels):
        if i > -1:
            # reference point
            posi = po1[i]
            ll = labels[posi]

            # all the neightbours of my point in the two labeling systems
            parla1 = (labels  == ll)
            parla2 = (labels2 == labels2[posi])

    #       print(ll,labels[posi],labels2[posi],len(labels[parla1]),len(labels2[parla2]))

            # if the point labels two clusters, group them
            if labels2[posi] > -1 :
                mask = np.logical_or(parla1,parla2)
                # any of the points is already labelled?
                seta = np.array(list(set(newlabels[mask])))
                if max(seta) > -1:
                    ll = max(seta[seta > -1])
                    print('ojo')

                newlabels[mask] = ll
                labels11[mask] = -2
                labels22[mask] = -2

            elif labels2[posi] == -1 :
                mask = parla1
                newlabels[mask] = ll
                labels11[mask] = -2

            dicta.append([labels[posi],labels2[posi]])


    maxrelab = max(newlabels)

    # list of labels of clusters in 2, which are noise in 1
    #
    rela2 = np.array(list(set(labels22)))
    # oldstuff
    # add them to newlabels
    # remove them from labels_aux
#     counter = 0
#     for j in rela2:
#         if j > -1 :
#             parla22 = (labels22  == j)

#             ll2 = j

#             newlabels[parla22] = maxrelab + counter
#             dicta.append([-1,ll2])
#             counter += 1

#             labels11[parla22] = -2
#             labels22[parla22] = -2
    # however
    counter = 0

    print('rela2',rela2)
    for j in rela2:
        if j > -1 :
            parla22 = (labels22  == j)
            # must anypoint of this MC need be asssotiated with a mc already present in newlabels?
            asolabel = np.array(list(set(labels2[parla22])))
            print(j, asolabel)
            asolabel = asolabel[asolabel > -1]

            ll2 = j

            if asolabel == []:
                newlabels[parla22] = maxrelab + counter
                counter +=1
                dicta.append([-1,ll2])
            elif len(asolabel) > 0:
                lala = min(asolabel)

                for l in asolabel:
                    para = (newlabels == l)
                    newlabels[para] = lala
                    dicta.append([lala,j])
                newlabels[parla22] = lala

            labels11[parla22] = -2
            labels22[parla22] = -2


    if len(labels[labels11 != -2])>0:
        print('warning1',labels[labels11 != -2])
    if len(labels[labels11 != -2])>0:
        print('warning2',labels2[labels22 != -2])
        print('continue with l2')

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
    else:
        sys.exit()

    n     = f.attrs[u'Size']
    sizeL = f.attrs[u'Physical size']

    n2 = n*n
    n3 = n2*n

    data = f['energy/density'].value.reshape(n3)

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
