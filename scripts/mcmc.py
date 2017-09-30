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


# GET DATA

@jit
def pumba_pos_data(data_file, th):

    file = h5py.File(data_file, 'r')

    n = file.attrs[u'Size']
    sizeL = file.attrs[u'Physical size']

    n2 = n*n
    n3 = n2*n

    data = file['energy/density'].value

    nth = len(np.extract(data>th,data))

    print ("points per axis: %d" %n)
    print ("Physical size : (%1.2f)*#" %sizeL)

    a_out = np.zeros((nth,3)) #make this an integer array?
    counter = 0

    for index in range(0,n3):
        if data[index] > th:

            i = index//n2
            j = (index - i*n2)//n
            k = (index%n2)%n

            a_out[counter][0] = i
            a_out[counter][1] = j
            a_out[counter][2] = k

            counter +=1
    file.close()

    return data, a_out, n, sizeL, counter


#parameters for DBSCAN
eps=2 #was 5
min_samples=20
algorithm='kd_tree'
n_jobs=-1

#get from input
dens_th = 15
data_file = 'axion.r.00046'

f         = h5py.File('./' + sys.argv[1], "r")
dens_th   = float(sys.argv[2])

n     = f.attrs[u'Size']
sizeL = f.attrs[u'Physical size']

n2 = n*n
n3 = n2*n

an_contrastmap = 'energy/density' in data_file

if an_contrastmap:
    print('Reading ',sys.argv[1], ' with contrast threshold = ',dens_th)
else:
    exit()

data = f['energy/density'].value
nth = len(np.extract( data > dens_th ,data))

print ("points per axis = %d" %n)
print ("Physical size   = %1.2f^3" %sizeL)

pos = np.zeros((nth,3)) #make this an integer array?
counter = 0

start = time.time()

print('computing position array with pumba ... ')

for index in range(0,n3):
    if data[index] > dens_th:

        i = index//n2
        j = (index - i*n2)//n
        k = (index%n2)%n

        pos[counter][0] = i
        pos[counter][1] = j
        pos[counter][2] = k

        counter +=1
end=time.time()

print('took %1.3f s' %(end-start))

print('data=%1.2f,..., pos=(%d,%d,%d)..., n=%d, n3=%d and counti=%d' %(data[0],pos[0,0],pos[0,1],pos[0,2],n,n3,counter))
print('%2.2f per cent of points selected with contrast>%2.2f ' %(100.*counter/n3, dens_th))

print()
print('Clustering of the full dataset ',end='')
start = time.time()
db = DBSCAN(eps=eps, min_samples=min_samples, algorithm=algorithm, n_jobs=n_jobs).fit(pos)
labels = np.array(db.labels_)
end = time.time()
print('took %1.3f s' %(end-start))

unique_labels = set(labels)
colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
delta = sizeL/n

from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph.opengl as gl
import numpy as np

app = QtGui.QApplication([])
w = gl.GLViewWidget()
w.opts['distance'] = 2*n
w.show()
w.setWindowTitle('pyqtgraph example: GLScatterPlotItem')

g = gl.GLGridItem()
w.addItem(g)

for k, col in zip(unique_labels, colors):
    if (k>-1):

        class_member_mask = (labels == k)
        xyz = pos[class_member_mask] #list of coordinates of all points in one cluster
        sp1 = gl.GLScatterPlotItem(pos=xyz, size=1.0, color=col, pxMode=False)
        # sp1.translate(5,5,0)
        w.addItem(sp1)

# sp1 = gl.GLScatterPlotItem(pos=pos, size=1.0, color=(1.,1.,0.2,0.5), pxMode=False)
# #sp1.translate(-n//2,-n//2,-n//2)
# w.addItem(sp1)

## Start Qt event loop unless running in interactive mode.
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
