# -*- coding: utf-8 -*-
"""
Created on Fri Dec 30 20:33:51 2022

@author: Sai Gunaranjan
"""


import numpy as np
from fixedPointLibrary import convert_float_to_fixedPointInt, dropFractionalBits_fixedPointInt,\
    matrixMultiplicationFixedPoint

# np.random.seed(10)

numRowsA= 3
numColsA = numRowsB = 8
numColsB = 4
A1 = np.random.uniform(-10,10,numRowsA*numColsA).reshape(numRowsA,numColsA)
B1 = np.random.uniform(-12,12,numColsB*numRowsB).reshape(numRowsB,numColsB)

maxVal = np.amax(np.abs(np.hstack((A1.flatten(), B1.flatten()))))

totalBitWidth = 32
numIntBits = np.ceil(np.log2(maxVal)).astype('int64') + 1 # (includes signed bit as well and hence the additonal 1) #4
numFracBits = totalBitWidth - numIntBits #28
numSignBits = 1
A1fixed = convert_float_to_fixedPointInt(A1, numIntBits, numFracBits, numSignBits)
B1fixed = convert_float_to_fixedPointInt(B1, numIntBits, numFracBits, numSignBits)

inputArrFracBits = numFracBits
outputArrFracBits = inputArrFracBits
C1fixed = matrixMultiplicationFixedPoint(A1fixed, B1fixed, inputArrFracBits, outputArrFracBits)
C1fixedConvFloat = C1fixed/(2**outputArrFracBits)

C1float = A1 @ B1


print('Float result\n', C1float)
print('Fixed result\n', C1fixedConvFloat)
print('Total Error:\n',np.sum(C1float-C1fixedConvFloat))



