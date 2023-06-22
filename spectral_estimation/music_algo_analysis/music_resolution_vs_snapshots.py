# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 13:08:31 2023

@author: Sai Gunaranjan
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('..//')
from scipy.signal import argrelextrema
from spectral_estimation_lib import music_snapshots

np.random.seed(5)

plt.close('all')
num_samples = 6#8#32
c = 3e8
fc = 79e9
lamda = c/fc
mimoSpacing = lamda/2
fsSpatial = lamda/mimoSpacing
nativeAngResDeg = np.arcsin(fsSpatial/num_samples)*180/np.pi
print('Native Angular Resolution for {0} element ULA = {1:.2f} deg'.format(num_samples,nativeAngResDeg))


spectrumGridOSRFact = 43#To obatin about 256 sample MUSIC
digital_freq_grid = np.arange(-np.pi,np.pi,2*np.pi/(spectrumGridOSRFact*num_samples))
numPointMUSIC = len(digital_freq_grid)
angleGrid = np.arcsin(((digital_freq_grid/(2*np.pi))*fsSpatial))*180/np.pi
AngbinRes = np.arcsin(fsSpatial/numPointMUSIC)*180/np.pi

""" Noise parameters"""
noiseFloordB = -100
noise_power_db = noiseFloordB + 10*np.log10(num_samples)
noise_variance = 10**(noise_power_db/10)
noise_sigma = np.sqrt(noise_variance)

num_sources = 2
resol_fact = np.arange(0.1,1.1,0.1)
numResol = len(resol_fact)
digFreqRes = resol_fact*((2*np.pi)/num_samples)
angResDeg = np.arcsin((digFreqRes/(2*np.pi))*fsSpatial)*180/np.pi

snapshotsArray = np.arange(1,16,1)
lenSnapShotArray = len(snapshotsArray)

snrArray = np.arange(10,60,10) # np.array([40])
numSNR = len(snrArray)
snrdelta = 3 # This indicates by how much dB is the second target below the 1st target

numMonteCarlo = 200

estAngSepDegArray = np.zeros((numSNR,numResol,lenSnapShotArray,numMonteCarlo),dtype=np.float32)
for ele3 in range(numSNR):
    snr = snrArray[ele3]
    object_snr = np.array([snr,snr-snrdelta])
    weights = np.sqrt(10**((noiseFloordB + object_snr)/10))

    for ele1 in range(numResol):
        source_freq = np.array([0, 0 + digFreqRes[ele1]])
        source_angle_deg = np.arcsin((source_freq/(2*np.pi))*fsSpatial)*180/np.pi
        signalPhase = np.exp(1j*np.outer(np.arange(num_samples),source_freq))

        for ele2 in range(lenSnapShotArray):
            numSnapshots = snapshotsArray[ele2]
            signal_phases = np.exp(1j*np.random.uniform(low=-np.pi, high = np.pi, size = numSnapshots*num_sources))
            signal_phases = signal_phases.reshape(num_sources,numSnapshots)
            complex_signal_amplitudes = weights[:,None]*signal_phases
            source_signals = signalPhase @ complex_signal_amplitudes


            for ele4 in range(numMonteCarlo):
                wgn_noise = (noise_sigma/np.sqrt(2))*np.random.randn(num_samples * numSnapshots) + 1j*(noise_sigma/np.sqrt(2))*np.random.randn(num_samples * numSnapshots)
                received_signal = source_signals + wgn_noise.reshape(-1,numSnapshots)
                pseudo_spectrum = music_snapshots(received_signal, num_sources, num_samples, digital_freq_grid)
                pseudo_spectrum = pseudo_spectrum/np.amax(pseudo_spectrum)
                pseudo_spectrumdB = 10*np.log10(pseudo_spectrum)

                """  Estimated Azimuth resolution computation"""
                localMaxInd = argrelextrema(pseudo_spectrumdB,np.greater,axis=0,order=1)[0]
                try:
                    peakInd = np.argsort(pseudo_spectrumdB[localMaxInd])[-2::]
                    localMaxPeaks = localMaxInd[peakInd]
                    estAngleSepDeg = (np.arcsin(np.abs(np.diff(localMaxPeaks)) * fsSpatial/numPointMUSIC)*180/np.pi)[0]
                    if (np.isnan(estAngleSepDeg)):
                        estAngleSepDeg = 250
                except IndexError:
                    estAngleSepDeg = 250

                estAngSepDegArray[ele3,ele1,ele2,ele4] = estAngleSepDeg # snr, resol, snapshots, montecarlo

    print('\n\n Completed {}/{} SNRs'.format(ele3+1,numSNR))


estAngSepDegMeanArray = np.mean(estAngSepDegArray,axis=3)
errorAngResDeg = np.abs(angResDeg[None,:,None] - estAngSepDegMeanArray)
deltaMarginDeg = 2*AngbinRes

minSnapshotsArray = np.zeros((numSNR,numResol),dtype=np.int32)
for ele5 in range(numSNR):
    for ele6 in range(numResol):
        try:
            snapShotInd = np.array([np.where(errorAngResDeg[ele5,ele6,:] < deltaMarginDeg)]).min()
            numMinSnapshots = snapshotsArray[snapShotInd]
        except ValueError:
            numMinSnapshots = 200
        minSnapshotsArray[ele5,ele6] = numMinSnapshots


angResDegQunatized = np.round(angResDeg*10)/10
snrString = ['SNR = {} dB'.format(ele) for ele in snrArray]
numPointMUSICEff = 2**(np.floor(np.log2(numPointMUSIC)).astype('int32'))


plt.figure(1,figsize=(20,10),dpi=200)
plt.title('Snapshots vs Angular resolution with {} ULA and {} point MUSIC'.format(num_samples,numPointMUSICEff))
plt.plot(angResDeg,minSnapshotsArray.T,'-o')
plt.xlabel('Programmed Angular resolution (deg)')
plt.ylabel('Minimum snapshots required')
plt.yticks(np.arange(-2,15,2))
plt.xticks(angResDegQunatized)
plt.ylim([-1,15])
plt.grid(True)
plt.legend(snrString)








