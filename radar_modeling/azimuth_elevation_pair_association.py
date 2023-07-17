# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 18:26:28 2023

@author: Sai Gunaranjan
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('..//')
from spectral_estimation.spectral_estimation_lib import music_snapshots
from scipy.signal import argrelextrema
from scipy.linalg import toeplitz

np.random.seed(5)
plt.close('all')


def missingNumber(arr1, arr2):

    """
    missing number in shuffled array
    Returns the missing number
    Size of arr2[] is n - 1 """

    n = arr1.shape[0]
    # missing number 'mnum'
    mnum = 0

    # 1st array is of size 'n'
    for i in range(n):
        mnum = mnum ^ arr1[i]

    # 2nd array is of size 'n - 1'
    for i in range(n - 1):
        mnum = mnum ^ arr2[i]

    # Required missing number
    return mnum


def mimoPhasorSynth(platform, lamda, objectAzAngle_rad, objectElAngle_rad):

    path = 'DopplerDivisionMultiplexing\\antenna_cordinates\\' + platform + '\\antenna\\' #For association debug

    if (platform == 'L_shaped_array'):

        numTx = 6
        numRx = 6
        txSeq = np.array([0,1,2,3,4,5])
        ulaInd = np.array([0,1,2,3,4,5]) # 6 element Az ULA


    physicalTxCordinates = np.loadtxt(path+'LTXMA.txt')
    physicalRxCordinates = np.loadtxt(path+'LRXMA.txt')

    SeqBasedTxCordinates = physicalTxCordinates[txSeq,:]
    SeqBasedRxCordinates = np.copy(physicalRxCordinates)

    virtualArrayCordinates = SeqBasedTxCordinates[:,:,None] + SeqBasedRxCordinates.T[None,:,:] # [numTx, 3, numRx]
    virtualArrayCordinates = np.transpose(virtualArrayCordinates, (1,0,2)).reshape(3,numTx*numRx) # [3, numTx x numRx]

    azComp = np.sin(objectAzAngle_rad)
    elComp = np.sin(objectElAngle_rad)
    radialComp = np.cos(objectAzAngle_rad)

    # azComp = np.sin(objectAzAngle_rad)*np.cos(objectElAngle_rad)
    # elComp = np.sin(objectAzAngle_rad)*np.sin(objectElAngle_rad)
    # radialComp = np.cos(objectAzAngle_rad)

    objUnitVector = np.vstack((azComp,elComp,radialComp)).T # [numObj, 3]

    beta = 2*np.pi/lamda
    mimoPhasor = np.exp(1j * beta * (objUnitVector @ virtualArrayCordinates)) # [numObj, numTx x numRx]

    mimoPhasor_txrx = mimoPhasor.reshape(-1, numTx, numRx) # [numObj, numTx, numRx]

    return mimoPhasor, mimoPhasor_txrx, ulaInd




def music_on_autocorrMat(auto_corr_matrix, num_sources, num_samples, digital_freq_grid):

    signal_length = auto_corr_matrix.shape[0]

    # auto_corr_matrix = (auto_corr_matrix + np.fliplr(np.flipud(np.conj(auto_corr_matrix))))*0.5

    auto_corr_matrix = auto_corr_matrix/signal_length # Divide the auto-correlation matrix by the signal length
    u, s, vh = np.linalg.svd(auto_corr_matrix) # Perform SVD of the Auto-correlation matrix
    noise_subspace = u[:,num_sources::] # The first # number of sources eigen vectors belong to the signal subspace and the remaining eigen vectors of U belong to the noise subspace which is orthogonal to the signal subspace. Hence pick these eigen vectors

    """ The below step is to reduce the compute by taking only 1 noise subspace eigen vector and
    performing FFT on it instead of on all the noise subspace eigen vectors. Theoretically this is correct
    but with real data, using all the noise subspace eigen vectors and taking FFT and then
    magnitude square and sum across all the FFTed noise subspace eigen vectors helps
    give a smoother pseudo spectrum. If we use only 1/2 noise subspace eigen vectors, The true peaks are
    undisturbed but some smaller flase peaks start to show up. Using all the noise subspace eigen vectors helps smoothen
    and eliminate the false peaks.
    Thus I have taken only 2 of the noise subpace eigen vectors instead of all. Enable below line
    if you want to use only 1/2 noise subspace eigen vectors to reduce compute.
    """

    """ GhA can be computed in 2 ways:
        1. As a correlation with a vandermonde matrix
        2. Oversampled FFT of each of the noise subspace vectors
        Both 1 and 2 are essentially one and the same. But 1 is compute heavy in terms of MACS while 2 is more FFT friendly

    """
    GhA = np.fft.fftshift(np.fft.fft(noise_subspace.T.conj(),n=len(digital_freq_grid),axis=1),axes=(1,)) # Method 2
    AhG = GhA.conj() # A*G
    AhGGhA = np.sum(AhG*GhA,axis=0) # A*GG*A
    pseudo_spectrum = 1/np.abs(AhGGhA) # Pseudo spectrum
    return pseudo_spectrum

platform = 'L_shaped_array'
lightSpeed = 3e8
centerFreq = 76.5e9
lamda = lightSpeed/centerFreq
numTx = 6
numRx = 6
rxSpacing = lamda/2
fsRx = lamda/rxSpacing
txSpacing = lamda #lamda/2
fsTx = lamda/txSpacing
num_sources = 3

rxAngRes = np.arcsin(fsRx/numRx)*180/np.pi
txAngRes = np.arcsin(fsTx/numTx)*180/np.pi

rxMaxAng = np.arcsin(fsRx/2)*180/np.pi
txMaxAng = np.arcsin(fsTx/2)*180/np.pi

print('Rx Ang Res = {0:.2f} deg, Tx Ang Res = {1:.2f} deg'.format(rxAngRes,txAngRes))
print('Rx Ang Max = {0:.2f} deg, Tx Ang Max = {1:.2f} deg'.format(rxMaxAng,txMaxAng))

spectrumGridOSRFact = 43#To obtain about 256 sample MUSIC
digital_freq_grid = np.arange(-np.pi,np.pi,2*np.pi/(spectrumGridOSRFact*numTx))
numPointMUSIC = len(digital_freq_grid)

angleGridRx = np.arcsin(((digital_freq_grid/(2*np.pi))*fsRx))*180/np.pi
AngbinResRx = np.arcsin(fsRx/numPointMUSIC)*180/np.pi
angleGridTx = np.arcsin(((digital_freq_grid/(2*np.pi))*fsTx))*180/np.pi
AngbinResTx = np.arcsin(fsTx/numPointMUSIC)*180/np.pi

# offsetaz = 10
# objectAzAngle_deg = np.array([-np.round(rxMaxAng) + offsetaz,-np.round(rxMaxAng) + np.round(rxAngRes) + offsetaz ,\
#                               -np.round(rxMaxAng) + 2*np.round(rxAngRes) + offsetaz])
# offsetel = 5
# objectElAngle_deg = np.array([-np.round(txMaxAng) + offsetel,-np.round(txMaxAng) + np.round(txAngRes) + offsetel ,\
#                               -np.round(txMaxAng) + 2*np.round(txAngRes) + offsetel])

objectAzAngle_deg = np.array([-5,5,10])
objectElAngle_deg = np.array([-10,-5, 2])

objectAzAngle_rad = (objectAzAngle_deg/360) * (2*np.pi)
objectElAngle_rad = (objectElAngle_deg/360) * (2*np.pi)
actualAzElAnglePairs = np.hstack((objectAzAngle_deg[:,None],objectElAngle_deg[:,None]))

_, mimoPhasor_txrx, _ = mimoPhasorSynth(platform, lamda, objectAzAngle_rad, objectElAngle_rad)

ver_ula_signal = np.conj(mimoPhasor_txrx[:,:,0].T)
hor_ula_signal = np.conj(mimoPhasor_txrx[:,0,:].T)

numSnapshots = 100
snr = 50
snrdelta = 0#3 # This indicates by how much dB is the second target below the 1st target and so on
object_snr = np.array([snr,snr-snrdelta,snr-2*snrdelta])
""" Noise parameters"""
noiseFloordB = -100
noise_power_db = noiseFloordB + 10*np.log10(numTx)
noise_variance = 10**(noise_power_db/10)
noise_sigma = np.sqrt(noise_variance)
weights = np.sqrt(10**((noiseFloordB + object_snr)/10))

signal_phases = np.exp(1j*np.random.uniform(low=-np.pi, high = np.pi, size = numSnapshots*num_sources))
signal_phases = signal_phases.reshape(num_sources,numSnapshots)
complex_signal_amplitudes = weights[:,None]*signal_phases
ver_signal = ver_ula_signal @ complex_signal_amplitudes
hor_signal = hor_ula_signal @ complex_signal_amplitudes

wgn_noise_tx = (noise_sigma/np.sqrt(2))*np.random.randn(numTx * numSnapshots) + 1j*(noise_sigma/np.sqrt(2))*np.random.randn(numTx * numSnapshots)
ver_signal = ver_signal + wgn_noise_tx.reshape(-1,numSnapshots)

wgn_noise_rx = (noise_sigma/np.sqrt(2))*np.random.randn(numRx * numSnapshots) + 1j*(noise_sigma/np.sqrt(2))*np.random.randn(numRx * numSnapshots)
hor_signal = hor_signal + wgn_noise_rx.reshape(-1,numSnapshots)

pseudo_spectrum_rx = music_snapshots(hor_signal, num_sources, numRx, digital_freq_grid)
pseudo_spectrum_rx = pseudo_spectrum_rx/np.amax(pseudo_spectrum_rx)
pseudo_spectrum_rxdB = 10*np.log10(pseudo_spectrum_rx)

pseudo_spectrum_tx = music_snapshots(ver_signal, num_sources, numTx, digital_freq_grid)
pseudo_spectrum_tx = pseudo_spectrum_tx/np.amax(pseudo_spectrum_tx)
pseudo_spectrum_txdB = 10*np.log10(pseudo_spectrum_tx)

localMaxInd = argrelextrema(pseudo_spectrum_rxdB,np.greater,axis=0,order=1)[0]
peakInd = np.argsort(pseudo_spectrum_rxdB[localMaxInd])[-num_sources::]
localMaxPeaks = localMaxInd[peakInd]
azDigFreq = digital_freq_grid[localMaxPeaks]
estAzAngles = angleGridRx[localMaxPeaks]

localMaxInd = argrelextrema(pseudo_spectrum_txdB,np.greater,axis=0,order=1)[0]
peakInd = np.argsort(pseudo_spectrum_txdB[localMaxInd])[-num_sources::]
localMaxPeaks = localMaxInd[peakInd]
elDigFreq = digital_freq_grid[localMaxPeaks]
estElAngles = angleGridTx[localMaxPeaks]

azelMat = azDigFreq[:,None] - elDigFreq[None,:]

azelCrossCorrMat = ver_signal @ np.conj(hor_signal).T #hor_signal @ np.conj(ver_signal).T

Rxz = azelCrossCorrMat.diagonal()

Rxztoeplitz = toeplitz(Rxz)
pseudo_spectrum_rxtx = music_on_autocorrMat(Rxztoeplitz, num_sources, numTx, digital_freq_grid)
pseudo_spectrum_rxtxdB = 10*np.log10(pseudo_spectrum_rxtx)

localMaxInd = argrelextrema(pseudo_spectrum_rxtxdB,np.greater,axis=0,order=1)[0]
peakInd = np.argsort(pseudo_spectrum_rxtxdB[localMaxInd])[-num_sources+1::] # Rely on the strongest guys
localMaxPeaks = localMaxInd[peakInd]
crossDigFreq = digital_freq_grid[localMaxPeaks]

f_ijk = np.abs(azelMat[:,:,None] - crossDigFreq[None,None,:])
azelIndPairs = np.zeros((num_sources,2),dtype=np.int32)
estAzElAnglePairs = np.zeros((num_sources,2),dtype=np.float32)
for ele in range(f_ijk.shape[2]):
    linIndex = np.argmin(f_ijk[:,:,ele])
    indexPairs = np.unravel_index(linIndex,azelMat.shape)
    azelIndPairs[ele,:] = np.array(indexPairs)
    estAzElAnglePairs[ele,0] = estAzAngles[indexPairs[0]]
    estAzElAnglePairs[ele,1] = estElAngles[indexPairs[1]]

missAzInd = missingNumber(np.arange(num_sources),azelIndPairs[0:num_sources-1,0])
missElInd = missingNumber(np.arange(num_sources),azelIndPairs[0:num_sources-1,1])
azelIndPairs[-1,:] = np.array([missAzInd,missElInd])
estAzElAnglePairs[-1,0] = estAzAngles[missAzInd]
estAzElAnglePairs[-1,1] = estElAngles[missElInd]

actualAzElAnglePairs = actualAzElAnglePairs[np.argsort(actualAzElAnglePairs[:,0]),:]
estAzElAnglePairs = estAzElAnglePairs[np.argsort(estAzElAnglePairs[:,0]),:]
print('\n True Az/El angle pairs = \n', actualAzElAnglePairs)
print('\n Estimated Az/El angle pairs = \n', estAzElAnglePairs)

plt.figure(1,figsize=(20,10),dpi=200)
plt.subplot(1,2,1)
plt.title('Num Rx samples = ' + str(numRx))
plt.plot(angleGridRx, pseudo_spectrum_rxdB, label='MUSIC snapshots = {}'.format(numSnapshots))
plt.vlines(objectAzAngle_deg,-60,5, alpha=1,color='black',ls='dashed',label = 'Ground truth')
plt.xlabel('Angle(deg)')
plt.legend()
plt.grid(True)

plt.subplot(1,2,2)
plt.title('Num Tx samples = ' + str(numTx))
plt.plot(angleGridTx, pseudo_spectrum_txdB, label='MUSIC snapshots = {}'.format(numSnapshots))
plt.vlines(objectElAngle_deg,-60,5, alpha=1,color='black',ls='dashed',label = 'Ground truth')
plt.xlabel('Angle(deg)')
plt.legend()
plt.grid(True)

plt.figure(2,figsize=(20,10),dpi=200)
plt.title('MUSIC spectrum of cross corr mat')
plt.plot(angleGridRx, pseudo_spectrum_rxtxdB)
plt.xlabel('Angle(deg)')
plt.grid(True)
