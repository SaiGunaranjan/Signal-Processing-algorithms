# -*- coding: utf-8 -*-
"""
Created on Fri Dec 30 20:33:51 2022

@author: Sai Gunaranjan
"""


import numpy as np
from fixedPointLibrary import convert_float_to_fixedPointInt, dropFractionalBits_fixedPointInt,\
    matrixMultiplicationFixedPoint, matrixMultiplicationFixedPointComplexInput, \
        convert_Complexfloat_to_fixedPointInt

# np.random.seed(10)
print('\n\n Real Multiplication')

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


print('\n\n Complex Multiplication')

numRowsA= 3
numColsA = numRowsB = 8
numColsB = 4
A1 = np.random.uniform(-10,10,numRowsA*numColsA).reshape(numRowsA,numColsA) + 1j*np.random.uniform(-10,10,numRowsA*numColsA).reshape(numRowsA,numColsA)
B1 = np.random.uniform(-12,12,numColsB*numRowsB).reshape(numRowsB,numColsB) + 1j*np.random.uniform(-12,12,numRowsB*numColsB).reshape(numRowsB,numColsB)

realArray = np.hstack((A1.real.flatten(),B1.real.flatten()))
imagArray = np.hstack((A1.imag.flatten(),B1.imag.flatten()))
maxVal = np.amax(np.abs(np.hstack((realArray, imagArray))))

totalBitWidth = 32
numIntBits = np.ceil(np.log2(maxVal)).astype('int64') + 1 # (includes signed bit as well and hence the additonal 1) #4
numFracBits = totalBitWidth - numIntBits #28
numSignBits = 1

# A1fixedReal = convert_float_to_fixedPointInt(A1.real, numIntBits, numFracBits, numSignBits)
# A1fixedImag = convert_float_to_fixedPointInt(A1.imag, numIntBits, numFracBits, numSignBits)
# B1fixedReal = convert_float_to_fixedPointInt(B1.real, numIntBits, numFracBits, numSignBits)
# B1fixedImag = convert_float_to_fixedPointInt(B1.imag, numIntBits, numFracBits, numSignBits)
# A1fixed = (A1fixedReal + 1j*A1fixedImag).astype('complex64')
# B1fixed = (B1fixedReal + 1j*B1fixedImag).astype('complex64')

A1fixed = convert_Complexfloat_to_fixedPointInt(A1,numIntBits, numFracBits, numSignBits)
B1fixed = convert_Complexfloat_to_fixedPointInt(B1,numIntBits, numFracBits, numSignBits)

inputArrFracBits = numFracBits
outputArrFracBits = inputArrFracBits
C1fixed = matrixMultiplicationFixedPointComplexInput(A1fixed, B1fixed, inputArrFracBits, outputArrFracBits)
C1fixedConvFloat = C1fixed/(2**outputArrFracBits)

C1float = A1 @ B1


print('Float result\n', C1float)
print('Fixed result\n', C1fixedConvFloat)
print('Total Error:\n',np.sum(C1float-C1fixedConvFloat))



