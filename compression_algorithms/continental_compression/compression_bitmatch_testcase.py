# -*- coding: utf-8 -*-
"""
Created on Tue May 14 12:50:22 2024

@author: Sai Gunaranjan
"""

"""
This file contains the test case example from the conti patent. I use the example to check if the paper output and
my implementation output bitmatch exactly.
The link to the patent/paper is available here:
    https://saigunaranjan.atlassian.net/wiki/spaces/RM/pages/53903364/RADAR+data+compression+scheme+-+Continental
"""

import numpy as np
from conti_compression import ContiCompression


signalBitwidth = 32
numMantissaBits = 7
numComplexRxs = 4

rx1Real = np.uint32(int('11111110101110000000010000010111',2))
rx1Imag = np.uint32(int('11111111110100101000101110100101',2))
rx2Real = np.uint32(int('00000000101010000101000100010010',2))
rx2Imag = np.uint32(int('11111011100010101000001000010001',2))
rx3Real = np.uint32(int('00000101010100000111101010101010',2))
rx3Imag = np.uint32(int('11111111010101000000101010010100',2))
rx4Real = np.uint32(int('00000000100001010010100001001010',2))
rx4Imag = np.uint32(int('11111101010010100001010010000001',2))

rxRealGT = np.array([rx1Real,rx2Real,rx3Real,rx4Real]).astype(np.uint32)
rxImagGT = np.array([rx1Imag,rx2Imag,rx3Imag,rx4Imag]).astype(np.uint32)

compressedDataGT = np.uint64(int('0000010011101011111110000010110111000101010111101000001001101010',2))

rx1RealdecompGT =  np.uint32(int('11111110101000000000000000000000',2))
rx1ImagdecompGT =  np.uint32(int('11111111110000000000000000000000',2))
rx2RealdecompGT =  np.uint32(int('00000000101000000000000000000000',2))
rx2ImagdecompGT =  np.uint32(int('11111011100000000000000000000000',2))
rx3RealdecompGT =  np.uint32(int('00000101010000000000000000000000',2))
rx3ImagdecompGT =  np.uint32(int('11111111010000000000000000000000',2))
rx4RealdecompGT =  np.uint32(int('00000000100000000000000000000000',2))
rx4ImagdecompGT =  np.uint32(int('11111101010000000000000000000000',2))

rxRealdecompGT = np.array([rx1RealdecompGT,rx2RealdecompGT,rx3RealdecompGT,rx4RealdecompGT]).astype(np.uint32)
rxImagdecompGT = np.array([rx1ImagdecompGT,rx2ImagdecompGT,rx3ImagdecompGT,rx4ImagdecompGT]).astype(np.uint32)

rxdecompGT = np.zeros((2*numComplexRxs),dtype=np.uint32)
rxdecompGT[0::2] = rxRealdecompGT
rxdecompGT[1::2] = rxImagdecompGT


contComp = ContiCompression(signalBitwidth,numMantissaBits,numComplexRxs)
contComp.compress_rx_samples(rxRealGT,rxImagGT)
compressedData = contComp.compressedRxSamples
# print('GT compressed data = {}, implementation compressed data = {}\n'.format(compressedDataGT,compressedData))
if (compressedDataGT == compressedData):
    print('Compressed data exactly matches!\n')
else:
    print('Compressed data mismatch!!!')

contComp.decompress_rx_samples()
rxRealdecomp = contComp.rxSamplesRealRecon
rxImagdecomp = contComp.rxSamplesImagRecon

rxdecomp = np.zeros((2*numComplexRxs),dtype=np.uint32)
rxdecomp[0::2] = rxRealdecomp
rxdecomp[1::2] = rxImagdecomp

# for ele in range(numComplexRxs):
#     print('decompressed Rx {} Real GT = {}, decompressed Rx {} Real est = {}'.format(ele,rxRealdecompGT[ele],ele,rxRealdecomp[ele]))

# for ele in range(numComplexRxs):
#     print('decompressed Rx {} Imag GT = {}, decompressed Rx {} Imag est = {}'.format(ele,rxImagdecompGT[ele],ele,rxImagdecomp[ele]))

if np.sum(rxdecompGT-rxdecomp) == 0:
    print('Decompressed data exactly matches!')
else:
    print('Some of the decompressed samples do not bitmatch!')



