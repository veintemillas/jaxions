#!/usr/bin/python3

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import math
import re, os
import h5py
import datetime
import glob
from pyaxions import jaxions as pa

def spectime(simutaba,timo,acc,nqcd=7.0):
    nslist = []
    nslist_t = []
    ta = []
    klista = []
    for sim in simutaba:
        if sim.nqcd == nqcd :
            ms2 = sim.msa*sim.N/sim.L
            if (sim.spok) and (sim.safe) :
                ta, klista, nspa = sim.listnspct(timo)
                if np.abs(1-ta/timo) < acc:
                    nslist_t.append(ta)
                    nslist.append(nspa)
    nslist_t = np.array(nslist_t)
    nslist = np.array(nslist)
    err35 = nslist.std(axis=0)/np.sqrt(len(nslist))
    mea35 = nslist.sum(axis=0)/len(nslist)
    if klista is not None:
        return klista, mea35, err35

def autosep(dirlist,timset,nqcd=7.0):
    simutaba=[]
    klista=[]
    mt=[]
    et=[]
    for dir in dirlist:
        a = pa.simu(dir)
        simutaba.append(a)
    for t in timset:
        klista, meanS, error = spectime(simutaba,t,0.02,nqcd)
        mt.append(meanS)
        et.append(error)
        print(t,'done',klista[0],meanS[0],error[0])
    if klista is not None:
        return klista, np.array(mt), np.array(et)

def pspectime(simutaba,timo,acc,nqcd=7.0):
    nslist = []
    nslist_t = []
    ta = []
    klista = []
    for sim in simutaba:
        if sim.nqcd == nqcd :
            ms2 = sim.msa*sim.N/sim.L
            if (sim.spok) and (sim.safe) :
                ta, klista, nspa = sim.listpspct(timo)
                if np.abs(1-ta/timo) < acc:
                    nslist_t.append(ta)
                    nslist.append(nspa)
    nslist_t = np.array(nslist_t)
    nslist = np.array(nslist)
    err35 = nslist.std(axis=0)/np.sqrt(len(nslist))
    mea35 = nslist.sum(axis=0)/len(nslist)
    if klista is not None:
        return klista, mea35, err35

def pautosep(dirlist,timset,nqcd=7.0):
    simutaba=[]
    klista=[]
    mt=[]
    et=[]
    for dir in dirlist:
        a = pa.simu(dir)
        simutaba.append(a)
    for t in timset:
        klista, meanS, error = pspectime(simutaba,t,0.02,nqcd)
        mt.append(meanS)
        et.append(error)
        print(t,'done',klista[0],meanS[0],error[0])
    if klista is not None:
        return klista, np.array(mt), np.array(et)

coli=np.array(((255,20,147),(75,0,130),(100,149,237),(46,139,87),(255, 10, 10),
               (255, 165, 0),(0,191,255),(255,69,0),(127,255,0),(25, 31, 29),
              (42, 212, 237),(227, 212, 237),(201, 187, 237),(113, 155, 46),(25, 31, 29)))
coli=coli/255
