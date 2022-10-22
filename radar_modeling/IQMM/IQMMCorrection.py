# -*- coding: utf-8 -*-
"""
Created on Sat Oct  1 20:24:00 2022

@author: Sai Gunaranjan
"""

"""
In phase Quadrature phase Mismatch (IQMM).
Reference document for the IQMM correction is available at the below link:
    https://www.faculty.ece.vt.edu/swe/argus/iqbal.pdf

Model IQMM mismatch correction

I have modelled a baseband sinusoidal signal with a IQ
Mismatch i.e the Inphase component of the complex baseband signal and
the quadrature phase component of the complex baseband signal have
different magnitudes and a phase delta which is not equal to 90 degrees.
The inphase component is cos(wt) and the quadrature
phase component is sin(wt). Together, they form a complex baseband
signal cos(wt) + j sin(wt). They have the same magnitude and a phase
delta of 90 degrees. This is the ideal baseband signal. However, in
practical scenarios, complex baseband receivers have an IQ imbalance. The
received high frequency RF signal is demodulated using two different
chains, one with the In phase component of the Local Oscillator (LO
which is also used to generate the RF signal at the Tx) and also
demodulated with the Quadrature phase component of the LO. Now, if there
is no IQ imbalance in the RX chain, then we will see a perfect
analytic(frequency component lies only on one side of spectrum)
sinusoidal signal whose spectrum is perfectly one sided. However, if
there is an IQ imbalance, we will not get a perfect analytic signal and
there will be two components in the signal spectrum. One at the positive
frequency and the other at the negative frequency. One component will be
strong and the other component will be slightly below the true
component. Ideally, the other component should have been perfectly 0 but
due to the IQ imbalance, the image component shoots up. This is a common
problem in complex baseband receivers. There is a way to eliminate this
IQ imbalance and hence estimate the true complex baseband signal using
some correction factors. In this commit, I have implemented a method to
estimate the IQMM parameters and then correct the signal to retrieve
back the true signal. The link to the document whoich shows this method
is given above. Results show that by estimating
the magnitude and phase parameters and using them to correct the signal,
I was able to retrieve back the true signal. For now, in this commit, I have
not added any noise. In the subsequenct commits, I will make the script more generalized.
"""

import numpy as np
import matplotlib.pyplot as plt
plt.close('all')

numSamp = 128
numFFTBins = 128
freqBin = 6 + np.random.uniform(0,1)#np.random.randint(0,numSamp) + np.random.uniform(0,1)

""" Actual I channel and Q channel signals"""
iChannelSignal = np.cos(2*np.pi*freqBin*np.arange(numSamp)/numSamp)
qChannelSignal = np.sin(2*np.pi*freqBin*np.arange(numSamp)/numSamp)

complexSignal = iChannelSignal + 1j*qChannelSignal



amplitudeMismatchFactor = np.random.uniform(0.03,0.5) #0.3
phaseMismatchFactorDeg = np.random.uniform(0,80) #10
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
phaseMismatchFactorDegEst = np.mod(phaseMismatchFactorDegEst,360)

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
eps = 1e-10

plt.figure(1,figsize=(20,10),dpi=200)
plt.suptitle('IQMM correction')
plt.subplot(2,2,1)
plt.title('Time domain')
plt.plot(iChannelSignalWithIQMM,label='I channel with IQMM')
plt.plot(qChannelSignalWithIQMM,label='Q channel with IQMM')
plt.grid(True)
plt.legend()

plt.subplot(2,2,2)
plt.title('Frequency domain')
plt.plot(fftAxis, 10*np.log10(np.abs(complexBaseBandSignalWithIQMMFFT)**2 + eps),label='IQMM signal spectrum')
plt.grid(True)
plt.legend()

plt.subplot(2,2,3)
plt.plot(iChannelSignalPostIQMMCorrection,label='I channel post IQMM correction')
plt.plot(qChannelSignalPostIQMMCorrection,label='Q channel post IQMM correction')
plt.xlabel('signal index')
plt.grid(True)
plt.legend()

plt.subplot(2,2,4)
plt.plot(fftAxis, 10*np.log10(np.abs(complexBaseBandSignalPostIQMMCorrectionFFT)**2 + eps), label='Signal spectrum post IQMM correction')
plt.xlabel('Frequency bin')
plt.grid(True)
plt.legend()




