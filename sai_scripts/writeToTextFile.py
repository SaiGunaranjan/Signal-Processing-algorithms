# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 12:42:03 2018

@author: lenovo
"""

import numpy as np

Th = np.load('thresholdCurve.npy')
file_name = 'thresholdCurve.txt'
file = open(file_name,'w')
for ele in Th:
    file.write(str(ele)+','+'\n')
file.close()




def write2txtfile_complex(file_name, flattened_2d_aray):
    num_rows = np.arange(flattened_2d_aray.shape[0])
    file = open(file_name,'w')
    for row in num_rows:
        for ele in flattened_2d_aray[row]:
            file.write(str(np.real(ele))+','+'\n')
            file.write(str(np.imag(ele))+','+'\n')
    file.close()