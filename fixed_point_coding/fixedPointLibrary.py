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


def matrixMultiplicationFixedPoint_nonOptimal(A, B, inputArrFracBits, outputArrFracBits):

    """
    This function does a fixed point matrix multiplication by dropping bits only at the final stage.
    This is not an optimal method of product and sum because, this would require a much larger bitwidth
    accumulator. For example, if the inputs are 32 bit numbers, then the product will be at max a 64 bit number.
    Now, if we were to sum/accumulate these products, then the accumulator should be of bit width 64
    which takes more area. Rather the more optimal way would be to drop 32 bits post multiplication itself.
    That way, the size of the accumulator would need to be only 32 bits.

    Note: The accuracy with this method might be slighly higher though since we are working with
    higher bitwidth numbers till the final stage.

    Input
     A: Fixed point matrix of dimension m x n and number of fractional bits = inputArrFracBits
     B: Fixed point matrix of dimension n x k and number of fractional bits = inputArrFracBits
     inputArrFracBits: Number of fraction bits allocated for each of the input matrices A, B
     outputArrFracBits: Number of fractional bits that must be allocated to the final result.
    Output
     C: Fixed point matrix of dimension m x k and number of fractional bits = outputArrFracBits


     The fixed point numbers can be of form mQn like 1Q15 or 1Q31 or 1Q63

    """

    numRowsC = A.shape[0]
    numColsC = B.shape[1]
    C = np.zeros((numRowsC, numColsC),dtype = A.dtype)
    numColsA = numRowsB = A.shape[1]
    bitGrowthDueToAddns = np.round(np.log2(numColsA)).astype('int32')
    inputFracBits = inputArrFracBits + inputArrFracBits # Assuming 1Q31 x 1Q31
    outputFracBitsPostMult = outputArrFracBits  # To accomodate the bit growth due to additions.
    bitToBeDrop = inputFracBits - outputFracBitsPostMult

    for i in range(numRowsC):
        for j in range(numColsC):
            sumVal = 0
            for k in range(numColsA):
                prod = ((A[i,k].astype('int64'))*B[k,j]).astype('int64') # 32 x 32. The product should be atleast int64 because 32 x 32 results in 64 bits
                sumVal += prod
                print(sumVal)
            sumVal = dropFractionalBits_fixedPointInt(sumVal, inputFracBits, outputFracBitsPostMult)
            C[i,j] = sumVal
    return C


def matrixMultiplicationFixedPoint(A, B, inputArrFracBits, outputArrFracBits):

    """
    This function does a fixed point matrix multiplication by dropping bits post each multiplication itself
    and thus ensuring that the intermediate result is always limited to the outputArrFracBits.
    This method of fixed point multiplication requires lesser area since the size of the accumulator is
    limited to 32 bit unlike the previous method where the size of the accumulator is 64 bits.

    The accuracy with this method might be slightly(very marginally) inferior to the previous function.

    Input
     A: Fixed point matrix of dimension m x n and number of fractional bits = inputArrFracBits
     B: Fixed point matrix of dimension n x k and number of fractional bits = inputArrFracBits
     inputArrFracBits: Number of fraction bits allocated for each of the input matrices A, B
     outputArrFracBits: Number of fractional bits that must be allocated to the final result.
    Output
     C: Fixed point matrix of dimension m x k and number of fractional bits = outputArrFracBits


     The fixed point numbers can be of form mQn like 1Q15 or 1Q31 or 1Q63

    """

    numRowsC = A.shape[0]
    numColsC = B.shape[1]
    # C = np.zeros((numRowsC, numColsC),dtype = A.dtype)
    """ I have initialized the output matrix as int64 but it will contain int32 elements.
    If I initialize as int32, I was getting over flow errors"""
    C = np.zeros((numRowsC, numColsC),dtype = np.int64) #
    numColsA = numRowsB = A.shape[1]
    bitGrowthDueToAddns = np.ceil(np.log2(numColsA)).astype('int32')
    # bitGrowthDueToAddns = numColsA
    inputFracBits = inputArrFracBits + inputArrFracBits # Assuming 1Q31 x 1Q31
    """ Don't have accomodate for bit growth due to additions when the number of additions are small."""
    outputFracBitsPostMult = outputArrFracBits #- bitGrowthDueToAddns
    bitToBeDrop = inputFracBits - outputFracBitsPostMult

    for i in range(numRowsC):
        for j in range(numColsC):
            sumVal = 0
            for k in range(numColsA):
                """ The below product should be atleast int64 because 32 x 32 results in 64 bits and
                hence to ensure the output of the product is 64 bits, atleast one of the input
                has to be typecast as int64 so that the final output is also int64"""
                prod = (A[i,k].astype('int64')*B[k,j])
                prodBitsDropped = dropFractionalBits_fixedPointInt(prod, inputFracBits, outputFracBitsPostMult)
                sumVal += prodBitsDropped
            C[i,j] = sumVal
    return C



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
