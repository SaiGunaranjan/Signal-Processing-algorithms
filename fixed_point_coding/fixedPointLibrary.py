# -*- coding: utf-8 -*-
"""
Created on Thu Dec 29 21:13:23 2022

@author: Sai Gunaranjan
"""


import numpy as np
import matplotlib.pyplot as plt


def convert_float_to_fixedPointInt(floatArr, numIntBits, numFracBits, numSignBits):

    """
    floatArr: must be a real valued array
    """

    effectiveNumBits = numIntBits + numFracBits - numSignBits # Number of integer bits also includes the signed bit

    posmaxVal = 2**effectiveNumBits - 1

    scaledValue = np.floor(floatArr*(2**numFracBits) + 0.5)

    scaledValue[scaledValue>posmaxVal] = posmaxVal # clip/saturate values above posmaxVal to posmaxVal to avoid overflow

    if numSignBits:
        negativeminVal = -(2**effectiveNumBits) # If signed
        scaledValue[scaledValue < negativeminVal] = negativeminVal # For signed integers, clip/saturate values below negativeminVal to negativeminVal to avoid underflow
    else:
        scaledValue[scaledValue < 0] = 0 # For unsigned integers, clip/saturate values below 0 to 0 to avoid underflow

    return scaledValue.astype('int32') # can also be type cast as float 32. In that case, it will be an integer but with a .0


def dropFractionalBits_fixedPointInt(inputFixedPointArr, inputArrFracBits, outputArrFracBits):

    """
    inputFixedPointArr : should be an array of integers in int32 format
    """

    numFracBitsToBeDropped = inputArrFracBits - outputArrFracBits # Works only when inputArrFracBits >= outputArrFracBits

    # outputFixedPointArr = (inputFixedPointArr + 2**(numFracBitsToBeDropped-1)) >> numFracBitsToBeDropped
    outputFixedPointArr = (inputFixedPointArr + (1<<(numFracBitsToBeDropped-1))) >> numFracBitsToBeDropped # Replaced 2**(n-1) with bit shift operation of shifting binary 1 by n-1 bits

    return outputFixedPointArr


def addFractionalBits_fixedPointInt(inputFixedPointArr, inputArrFracBits, outputArrFracBits):

    """
    inputFixedPointArr : should be an array of integers in int32 format
    """

    numFracBitsToBeAdded = outputArrFracBits - inputArrFracBits # Works only when inputArrFracBits >= outputArrFracBits

    outputFixedPointArr = inputFixedPointArr << numFracBitsToBeAdded

    return outputFixedPointArr


# def convert_Complexfloat_to_fixedPointInt(complexArr, numIntBits, numFracBits, numSignBits):

#     """
#     complexArr: must be a complex valued array
#     """

#     effectiveNumBits = numIntBits + numFracBits - numSignBits # Number of integer bits also includes the signed bit

#     posmaxVal = 2**effectiveNumBits - 1
#     negativeminVal = -(2**effectiveNumBits)

#     realPart = np.real(complexArr)
#     imagPart = np.imag(complexArr)

#     scaledValueRealPart = np.floor(realPart*(2**numFracBits) + 0.5)
#     scaledValueImagPart = np.floor(imagPart*(2**numFracBits) + 0.5)

#     scaledValueRealPart[scaledValueRealPart>posmaxVal] = posmaxVal # clip/saturate values above posmaxVal to posmaxVal to avoid overflow
#     scaledValueImagPart[scaledValueImagPart>posmaxVal] = posmaxVal # clip/saturate values above posmaxVal to posmaxVal to avoid overflow

#     if numSignBits:
#         scaledValueRealPart[scaledValueRealPart < negativeminVal] = negativeminVal # For signed integers, clip/saturate values below negativeminVal to negativeminVal to avoid underflow
#         scaledValueImagPart[scaledValueImagPart < negativeminVal] = negativeminVal # For signed integers, clip/saturate values below negativeminVal to negativeminVal to avoid underflow
#     else:
#         scaledValueRealPart[scaledValueRealPart < 0] = 0 # For unsigned integers, clip/saturate values below 0 to 0 to avoid underflow
#         scaledValueImagPart[scaledValueImagPart < 0] = 0 # For unsigned integers, clip/saturate values below 0 to 0 to avoid underflow

#     scaledcomplexArr = (scaledValueRealPart + 1j*scaledValueImagPart).astype('complex64') # can also be type cast as float 32. In that case, it will be an integer but with a .0

#     return scaledcomplexArr
