# -*- coding: utf-8 -*-
"""
Created on Sun May 12 20:00:51 2024

@author: Sai Gunaranjan
"""

"""
Conti compression scheme analysis

I have implemented Conti’s compression scheme for RADAR data. It achieves a 4x compression.
The fundamental principle of the scheme is as follows:
    1. This scheme does a compression across Rx channels.
    2. The input to the compression is 32 bit numbers stored in 2’s complement format (uint32).
    3. Find the channels(real and imaginary) with the largest value by looking at the number of sign extension bits. The number of sign extension bits is referred to as the block shift in the patent.
    4. Find the block shift across all the channels and obtain the minimum of the block shifts across the channels (4 complex channels → 8 real channels). Call this as minBlockShift. A larger block shift implies smaller values and a smaller block shift implies larger values.
    5. Code the minBlockShift number as an 8 bit unsigned number.
    6. Discard minBlockShift number of bits from the MSB and pick the next 7 bits from all the channels. These are called the mantissa bits.
    7. Discard all the other bits post the 7 bits.
    8. Now we have minBlockShift (8 bits), 7 mantissa bits for each of the 8 channels (4 x 2). So, in total, 8 + 7*8 = 64 bits.
    9. So, in essence, a 32 x 8 bit data per range bin and per chirp is compressed to a 64 bit data. This is 4 x compression!
    10. When decompressing, the MSB from the 7 bit mantissa is obtained for each of the 8 channels.
    For each channel, the msb is repeated minBlockShift times. To this, the 7 bit mantissa of each channel
    is appended. The remaining LSB bits post the sign extension + 7 mantissa bits is packed with 0s to make it
    a 32 bit number again(for each of the channels). Thus, we get back the 32 x 8 bit data post-decompression
    but with a precision/quantization loss.

This is the Conti RADAR data compression scheme.
The link to the patent doc is available in my confluence page here:
    https://saigunaranjan.atlassian.net/wiki/spaces/RM/pages/53903364/RADAR+data+compression+scheme+-+Continental



Some important points about the Conti compression scheme:
    1. If each Rx has a different sensitivity, then a scaling is done to the rx channels before subjecting
to blockshift/compression. Else, the fundamental assumption that all Rx channels see a similar target and
hence have similar values breaks down.

    2. This compression scheme is valid for only single angle targets in a given range bin.
    If there are multiple angle taregts in the same range, then it is like a sum of sinusoids and this leads
    to amplitude variations across the Rx channels. This could lead to some channels having
    very low values (content sitting in last few LSBs) and some having high values.
    The resultant block shift will capture the largest channel value and the subsequent 7 mantissa bits
    may not capture any content for the low strength rx channels. This could lead to loss of signal and
    hence loss of taregts. Impact of this compression on multiple targets in the same range bin needs to be
    studied.

    3. For signals/targets which are buried in noise post range FFT, there could be a loss of these taregts.
    Hence the leading 7 mantissa bits might not be able to capture the content of weak targets which are still
    buried in noise post range FFT but which might show up post Doppler and RX FFT.
    These taregts might be lost. This aspect also needs to be shown in the sims.
    This scheme doesnt account for an absolute signal power values. It only looks at the relative signal values
    across Rxs.

    4. In this scheme, stronger targets loose more precision and weaker taregts loose leser precision.
    This aspect also needs to be highlighted in sims.

    5. Since this scheme involves dropping the bits post the mantissa, there is a loss in precision.
    This is a quantization noise. Now will this result in a correlated or uncorrelated noise.
    Also, will this be a phase noise/multiplicative noise. In that case, what is the dBc of this noise?
    The scheme talks about this aspect as well. Need to study/invetigate this as well.

    6. In order to remove correlations/ make the quantization noise uncorrelated, this scheme talks about
    adding a random number from 0 to just less than 1 LSB of the mantissa for each of the Rx channels
    (real and imag). They claim this breaks the correlations across the ramps/rxs. Need to verify this.
    Also, sometimes this addition of a random number can lead to overflow of the mantissa, especially if
    the mantissa is like 0111111. Now when a random number is added to the bits subsequent to the
    7 mantissa bits this can cause a carry and this carry can percolate to the mantissa bits.
    This will result in a change in the mantissa MSB. This will then lead to change in the block shift as well.
    This is handled separately in the compression scheme.

    7. Instead of just dropping the bits post mantissa bits, the scheme also talks about rounding and then
    dropping the bits. How much does this help?

    8. The scheme also talks about variable length mantissa bits for the real and imaginary parts to
    maintain a final output length (post compression) of a power of 2.

"""

import numpy as np

class ContiCompression:

    def __init__(self,signalBitwidth,numMantissaBits,numComplexRxChannels):

        self.signalBitwidth = signalBitwidth # Includes integer(including sign bit) + fractional bits
        self.signalFracBits = self.signalBitwidth - 1
        self.numMantissaBits = numMantissaBits
        self.numComplexRxChannels = numComplexRxChannels
        self.numRealRxChannels = 2*self.numComplexRxChannels
        compressedDataSize = self.numMantissaBits*self.numRealRxChannels
        self.totalBitsPerSamp = int(2**(np.ceil(np.log2(compressedDataSize)))) #64
        self.numBlockShiftBits =  self.totalBitsPerSamp - compressedDataSize  # numBlockShiftBits also called sign extension bits
        self.compressionRatio = int((self.signalBitwidth*self.numRealRxChannels)/self.totalBitsPerSamp)


    def print_compression_ratio(self):

        print('{}x compression achieved!'.format(self.compressionRatio))


    def compute_blockshift(self, rxSampRealImag):

        self.BlockShiftArray = np.zeros(self.numRealRxChannels,dtype=np.uint32)
        for ele2 in range(self.numRealRxChannels):
            if (rxSampRealImag[ele2] != 0):
                msb = rxSampRealImag[ele2] >> (self.signalBitwidth-1) # msb can be 0 or 1. 0 for positive numbers and 1 for negative numbers
                bitFromLeft = msb
                count = 0
                while bitFromLeft == msb:
                    count += 1
                    bitFromLeft = (rxSampRealImag[ele2] & (np.uint32(1) << (self.signalBitwidth-1-count))) >> (self.signalBitwidth-1-count)
                self.BlockShiftArray[ele2] = count - 1
            else:
                self.BlockShiftArray[ele2] = self.signalBitwidth-1



        self.blockShift = np.amin(self.BlockShiftArray) # as type uint64


    def compress_rx_samples(self,rxSamplesReal, rxSamplesImag):

        """
        rxSamplesReal: should be signalBitwidth and datatype uint32 i.e. 2's complement format
        rxSamplesImag: should be signalBitwidth and datatype uint32 i.e. 2's complement format

        """
        rxSampRealImag = np.zeros((self.numRealRxChannels,),dtype=np.uint32) # storing as 2's complement form which is like uint
        rxSampRealImag[0::2] = rxSamplesReal
        rxSampRealImag[1::2] = rxSamplesImag

        self.compute_blockshift(rxSampRealImag)

        self.compressedRxSamples = np.uint64(0) # Initialize as a uint 64 bit number
        self.compressedRxSamples = self.blockShift << (self.totalBitsPerSamp - self.numBlockShiftBits)
        """When block shift is large, we may not be able to get numMantissaBits number of valid mantissa bits,
        in that case append zeros to the end or equivalently, left shift the data by the remaining bits required to get
        numMantissaBits number of bits"""
        for ele1 in range(self.numRealRxChannels):
            precisionBitsDropped = (self.signalBitwidth - (self.blockShift + self.numMantissaBits))
            if precisionBitsDropped >= 0:
                rxSamplesRealImagQuant = rxSampRealImag[ele1] >> precisionBitsDropped
            else:
                numValidMantissaBits = np.uint32(self.signalBitwidth - self.blockShift)
                rxSamplesRealImagQuant = rxSampRealImag[ele1] << (self.numMantissaBits - numValidMantissaBits)
            channelMantissa = rxSamplesRealImagQuant & (2**self.numMantissaBits - 1)#0x7F # 127 (pick the 7 LSBs(mantissa bits) without the sig extension/block shift bits)
            temp = channelMantissa << (self.totalBitsPerSamp - self.numBlockShiftBits - (ele1+1)*self.numMantissaBits)
            self.compressedRxSamples  = self.compressedRxSamples | temp

        return self.compressedRxSamples



    def decompress_rx_samples(self, compressedRxSamples):

        self.blockShiftReconstr = compressedRxSamples >> (self.numMantissaBits * self.numRealRxChannels)
        self.decompressedRxSamples = np.zeros((self.numRealRxChannels),dtype=np.uint32)
        for ele3 in range(self.numRealRxChannels):
            channelMantissaRecon = (compressedRxSamples >> self.numMantissaBits*ele3) & (2**self.numMantissaBits - 1)#0x7F
            msbRecon = channelMantissaRecon >> (self.numMantissaBits-1)
            if msbRecon == 1:
                signExtensionPart = np.uint32((2**self.blockShiftReconstr - 1) << (self.signalBitwidth - self.blockShiftReconstr))
            else:
                signExtensionPart = np.uint32(0)

            self.decompressedRxSamples[self.numRealRxChannels-ele3-1] = signExtensionPart | (channelMantissaRecon << \
                                                           (self.signalBitwidth-self.blockShiftReconstr-self.numMantissaBits))


        self.rxSamplesRealRecon = self.decompressedRxSamples[0::2] # uint32
        self.rxSamplesImagRecon = self.decompressedRxSamples[1::2] # uint32
















