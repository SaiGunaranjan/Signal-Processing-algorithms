# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 21:23:27 2022

@author: saiguna
"""


import numpy as np
from scipy.signal import argrelextrema


def computeTwoLargestLocalMaxima(x):

    Ncases= x.shape[0]

    twoLargestLocalPeaks_matrix = np.zeros((0,2)).astype('int32')

    for i in range(Ncases):
        data=x[i,:]
        inx= argrelextrema(data,np.greater,order=1)[0]
        if (len(inx) >= 2):
            twoLargestLocalPeaks = inx[np.argsort(data[inx])][-2::]
        else:
            twoLargestLocalPeaks = np.array([0,len(data)-1])

        twoLargestLocalPeaks_matrix = np.vstack((twoLargestLocalPeaks_matrix, twoLargestLocalPeaks))

    return twoLargestLocalPeaks_matrix