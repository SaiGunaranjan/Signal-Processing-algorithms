# -*- coding: utf-8 -*-
"""
Created on Sat Oct  1 20:24:00 2022

@author: Sai Gunaranjan
"""

"""
In phase Quadrature phase Mismatch (IQMM).
Reference document for the IQMM correction is available at the below link:
    https://www.faculty.ece.vt.edu/swe/argus/iqbal.pdf
"""

import numpy as np
import matplotlib.pyplot as plt

numSamp = 128
numFFTBins = 128
freqBin = 6#np.random.randint(0,numSamp) + np.random.uniform(0,1)

""" Actual I channel and Q channel signals"""
iChannelSignal = np.cos(2*np.pi*freqBin*np.arange(numSamp)/numSamp)
qChannelSignal = np.sin(2*np.pi*freqBin*np.arange(numSamp)/numSamp)

complexSignal = iChannelSignal + 1j*qChannelSignal



amplitudeMismatchFactor = 0.3
phaseMismatchFactorDeg = 10
phaseMismatchFactorRad = (phaseMismatchFactorDeg/180) * np.pi
""" I channel and Q channel signals at the receiver chain due to receiver chain having an IQ imbalance i.e
I channel and Q channel and not exactly orthogonal to each other and have
amplitude and phase mismatches"""
iChannelSignalWithIQMM = amplitudeMismatchFactor*np.cos(2*np.pi*freqBin*np.arange(numSamp)/numSamp)
qChannelSignalWithIQMM = np.sin(2*np.pi*freqBin*np.arange(numSamp)/numSamp + phaseMismatchFactorRad)

complexBaseBandSignalWithIQMM = iChannelSignalWithIQMM + 1j*qChannelSignalWithIQMM
complexBaseBandSignalWithIQMMFFT = np.fft.fft(complexBaseBandSignalWithIQMM)/numSamp
complexBaseBandSignalWithIQMMFFT = np.fft.fftshift(complexBaseBandSignalWithIQMMFFT)

""" Correction mechanism:
    Step 1: Estimate the imbalance factors amplitudeMismatchFactor(alpha), phaseMismatchFactorRad(phi)
    Step 2: Apply the correction to the IQ imbalanced signals using the parameters estimated from step 1
    to estimate the true I, Q channels
"""

""" Estimation of amplitude Mismatch Factor (alpha)"""
""" alpha = np.sqrt(2*<IChanIQMM,IChanIQMM>) """
""" phi = sin-1(2*<IChanIQMM, QChanIQMM>/alpha)"""

innerProductII = np.inner(iChannelSignalWithIQMM,iChannelSignalWithIQMM)/numSamp
amplitudeMismatchFactorEst = np.sqrt(2*innerProductII)

innerProductIQ = np.inner(iChannelSignalWithIQMM, qChannelSignalWithIQMM)/numSamp
phaseMismatchFactorRadEst = np.arcsin(2*innerProductIQ/amplitudeMismatchFactorEst)
phaseMismatchFactorDegEst = (phaseMismatchFactorRadEst/np.pi) * 180

print('Actual Amplitude Mismatch factor (alpha) = {0:.2f}'.format(amplitudeMismatchFactor))
print('Estimated Amplitude Mismatch factor = {0:.2f}'.format(amplitudeMismatchFactorEst))

print('Actual Phase Mismatch factor phi (deg) = {0:.2f}'.format(phaseMismatchFactorDeg))
print('Estimated Phase Mismatch factor phi (deg) = {0:.2f}'.format(phaseMismatchFactorDegEst))

""" IQMM correction matrix
iqMMCorrMatrix = np.array([[1/alpha, 0],
                           [-np.tan(phi)/alpha, 1/np.cos(phi)]
                           ])
"""

iChannelSignalPostIQMMCorrection = (1/amplitudeMismatchFactorEst)*iChannelSignalWithIQMM
qChannelSignalPostIQMMCorrection = ((-np.tan(phaseMismatchFactorRadEst)/amplitudeMismatchFactorEst) * iChannelSignalWithIQMM) + \
    (1/np.cos(phaseMismatchFactorRadEst)) * qChannelSignalWithIQMM

complexBaseBandSignalPostIQMMCorrection = iChannelSignalPostIQMMCorrection + 1j*qChannelSignalPostIQMMCorrection
complexBaseBandSignalPostIQMMCorrectionFFT = np.fft.fft(complexBaseBandSignalPostIQMMCorrection)/numSamp
complexBaseBandSignalPostIQMMCorrectionFFT = np.fft.fftshift(complexBaseBandSignalPostIQMMCorrectionFFT)


fftAxis = np.arange(-numFFTBins//2, numFFTBins//2)

plt.figure(1,figsize=(20,10),dpi=200)
plt.subplot(2,2,1)
plt.plot(iChannelSignalWithIQMM,label='I channel with IQMM')
plt.plot(qChannelSignalWithIQMM,label='Q channel with IQMM')
plt.grid(True)
plt.legend()

plt.subplot(2,2,2)
plt.plot(fftAxis, np.abs(complexBaseBandSignalWithIQMMFFT)**2,label='IQMM signal spectrum')
plt.grid(True)
plt.legend()

plt.subplot(2,2,3)
plt.plot(iChannelSignalPostIQMMCorrection,label='I channel post IQMM correction')
plt.plot(qChannelSignalPostIQMMCorrection,label='Q channel post IQMM correction')
plt.grid(True)
plt.legend()

plt.subplot(2,2,4)
plt.plot(fftAxis, np.abs(complexBaseBandSignalPostIQMMCorrectionFFT)**2, label='Signal spectrum post IQMM correction')
plt.grid(True)
plt.legend()




