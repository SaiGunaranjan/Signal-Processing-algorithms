# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 16:04:02 2023

@author: Sai Gunaranjan
"""


"""
Array 1 (L shaped array): Tx separated by lamda vertically and Rxs separated by lamda/2 horizontally

      Tx1

      Tx2

      Tx3

      Tx4

      Tx5

      Tx6


      Rx1 Rx2 Rx3 Rx4 Rx5 Rx6



Array 2: Tx2, Tx3, Tx4 separated by 3lamda/2 horizontally.
Tx1 and Tx2 separated by 2lamda in vertical.
Rx2, Rx3, Rx4 separated by lamda/2.
Rx1 and Rx2 separated by lamda.



      Tx2   Tx3   Tx4



      Tx1

      Rx2 Rx3 Rx4

      Rx1

So the virtual array for array 2, in essence, the azimuth ULA has (3Tx x 3Rx)9 elements separated by lamda/2 and
elevation ULA has 4 elements separated by lamda. So the virtual array for array2 looks as below:

    * * * * * * * * *

    *

    *

    *


"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema


np.random.seed(5)
plt.close('all')

def music_snapshots(received_signal, num_sources, num_samples, digital_freq_grid):

    signal_length = received_signal.shape[0]
    # numSnapshots = received_signal.shape[1]

    received_signal = np.flipud(received_signal) # Need to remove this compute and push the sign change to the peak picking
    auto_corr_matrix = received_signal @ np.conj(received_signal.T) # E[YY*] is accomplished as summation (yi * yih)

    """ The below step is done to improve noise spatial smoothing which further improves the resolvability.
    The proof for this is available in a technical report by MIT Lincoln laboratory by Evans, Johnson, Sun.
    The report was published in 1982. The proof is available in page 2-30. The link to the pdf is available in the below link:
        https://archive.ll.mit.edu/mission/aviation/publications/publication-files/technical_reports/Evans_1982_TR-582_WW-18359.pdf
    """
    auto_corr_matrix = (auto_corr_matrix + np.fliplr(np.flipud(np.conj(auto_corr_matrix))))*0.5

    auto_corr_matrix = auto_corr_matrix/signal_length # Divide the auto-correlation matrix by the signal length
    u, s, vh = np.linalg.svd(auto_corr_matrix) # Perform SVD of the Auto-correlation matrix
    noise_subspace = u[:,num_sources::] # The first # number of sources eigen vectors belong to the signal subspace and the remaining eigen vectors of U belong to the noise subspace which is orthogonal to the signal subspace. Hence pick these eigen vectors
    # vandermonde_matrix = np.exp(-1j*np.outer(np.arange(num_samples),digital_freq_grid)) # [num_samples,num_freq] # construct the vandermond matrix for several uniformly spaced frequencies

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
    # GhA = np.matmul(noise_subspace.T.conj(),vandermonde_matrix) #G*A essentially projects the vandermond matrix (which spans the signal subspace) on the noise subspace
    GhA = np.fft.fftshift(np.fft.fft(noise_subspace.T.conj(),n=len(digital_freq_grid),axis=1),axes=(1,)) # Method 2
    AhG = GhA.conj() # A*G
    AhGGhA = np.sum(AhG*GhA,axis=0) # A*GG*A
    pseudo_spectrum = 1/np.abs(AhGGhA) # Pseudo spectrum
    return pseudo_spectrum

def mimoPhasorSynth(lamda, objectAzAngle_rad, objectElAngle_rad):

    """ L_shaped_array """
    # numTx and numRx available as global variables

    txSeq = np.arange(numTx)
    ulaInd = np.arange(numRx)

    tx_Yoffset = 0#10e-3
    tx_ycordinates = tx_Yoffset + txSpacing*np.arange(numTx)

    rx_Xoffset = 0#20e-3
    rx_xcordinates = rx_Xoffset + rxSpacing*np.arange(numRx)

    physicalTxCordinates = np.zeros((numTx,3),dtype=np.float32) # [x,y,z]
    physicalTxCordinates[:,1] = np.flipud(tx_ycordinates)

    physicalRxCordinates = np.zeros((numRx,3),dtype=np.float32)
    physicalRxCordinates[:,0] = np.flipud(rx_xcordinates)


    SeqBasedTxCordinates = physicalTxCordinates[txSeq,:]
    SeqBasedRxCordinates = np.copy(physicalRxCordinates)

    virtualArrayCordinates = SeqBasedTxCordinates[:,:,None] + SeqBasedRxCordinates.T[None,:,:] # [numTx, 3, numRx]
    virtualArrayCordinates = np.transpose(virtualArrayCordinates, (1,0,2)).reshape(3,numTx*numRx) # [3, numTx x numRx]

    # for debug
    # plt.scatter(physicalRxCordinates[:,0],physicalRxCordinates[:,1])
    # plt.scatter(physicalTxCordinates[:,0],physicalTxCordinates[:,1])
    # plt.gca().set_aspect('equal')


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
num_sources = 2
numPointAngle = 256

rxAngRes = np.arcsin(fsRx/numRx)*180/np.pi
txAngRes = np.arcsin(fsTx/numTx)*180/np.pi

rxMaxAng = np.arcsin(fsRx/2)*180/np.pi
txMaxAng = np.arcsin(fsTx/2)*180/np.pi

print('Rx Ang Res = {0:.2f} deg, Tx Ang Res = {1:.2f} deg'.format(rxAngRes,txAngRes))
print('Rx Ang Max = {0:.2f} deg, Tx Ang Max = {1:.2f} deg'.format(rxMaxAng,txMaxAng))

digital_freq_grid = np.arange(-np.pi,np.pi,2*np.pi/(numPointAngle))
numPointMUSIC = len(digital_freq_grid)

angleGridRx = np.arcsin(((digital_freq_grid/(2*np.pi))*fsRx))*180/np.pi
AngbinResRx = np.arcsin(fsRx/numPointMUSIC)*180/np.pi
angleGridTx = np.arcsin(((digital_freq_grid/(2*np.pi))*fsTx))*180/np.pi
AngbinResTx = np.arcsin(fsTx/numPointMUSIC)*180/np.pi

# numSnapshots = numTx
numMonteCarlo = 200
num_sources = 2
resol_fact = np.arange(0.1,1.1,0.1)
numResol = len(resol_fact)
snrArray = np.array([40])#np.arange(10,60,10) # np.array([40])
numSNR = len(snrArray)
snrdelta = 0#3 # This indicates by how much dB is the second target below the 1st target
""" Noise parameters"""
noiseFloordB = -100
noise_power_db = noiseFloordB + 10*np.log10(numTx) # change this from numTx
noise_variance = 10**(noise_power_db/10)
noise_sigma = np.sqrt(noise_variance)

digFreqResAz = resol_fact*((2*np.pi)/numRx)
azAngResDeg = np.arcsin((digFreqResAz/(2*np.pi))*fsRx)*180/np.pi

digFreqResEl = resol_fact*((2*np.pi)/numTx)
elAngResDeg = np.arcsin((digFreqResEl/(2*np.pi))*fsTx)*180/np.pi


estAzAngleSepDegArr = np.zeros((numSNR,numResol,numMonteCarlo))
estElAngleSepDegArr = np.zeros((numSNR,numResol,numMonteCarlo))
for ele_snr in range(numSNR):
    snr = snrArray[ele_snr]
    object_snr = np.array([snr,snr-snrdelta])
    weights = np.sqrt(10**((noiseFloordB + object_snr)/10))
    signal_phases = np.exp(1j*np.random.uniform(low=-np.pi, high = np.pi, size = num_sources))
    complex_signal_amplitudes = weights * signal_phases

    for ele_res in range(numResol):
        objectAzAngle_deg = np.array([0,0+azAngResDeg[ele_res]])
        objectElAngle_deg = np.array([0,0+elAngResDeg[ele_res]])
        objectAzAngle_rad = (objectAzAngle_deg/360) * (2*np.pi)
        objectElAngle_rad = (objectElAngle_deg/360) * (2*np.pi)

        _, mimoPhasor_txrx, _ = mimoPhasorSynth(lamda, objectAzAngle_rad, objectElAngle_rad) # numObj, numTx, numRx
        mimoPhasor = np.conj(mimoPhasor_txrx) # Remove this conj by removing flipud in music snapshots function
        angleSignal = np.sum(mimoPhasor * complex_signal_amplitudes[:,None,None],axis=0)

        for ele_mc in range(numMonteCarlo):
            wgn_noise = (noise_sigma/np.sqrt(2))*np.random.randn(numTx * numRx) + 1j*(noise_sigma/np.sqrt(2))*np.random.randn(numTx * numRx)
            angleSignalwithNoise = angleSignal + wgn_noise.reshape(numTx,numRx)

            pseudo_spectrum_rx = music_snapshots(angleSignalwithNoise.T, num_sources, numRx, digital_freq_grid)
            pseudo_spectrum_rx = pseudo_spectrum_rx/np.amax(pseudo_spectrum_rx)
            pseudo_spectrum_rxdB = 10*np.log10(pseudo_spectrum_rx)

            pseudo_spectrum_tx = music_snapshots(angleSignalwithNoise, num_sources, numTx, digital_freq_grid)
            pseudo_spectrum_tx = pseudo_spectrum_tx/np.amax(pseudo_spectrum_tx)
            pseudo_spectrum_txdB = 10*np.log10(pseudo_spectrum_tx)

            """  Estimated Azimuth resolution computation"""
            localMaxInd = argrelextrema(pseudo_spectrum_rxdB,np.greater,axis=0,order=1)[0]
            try:
                peakInd = np.argsort(pseudo_spectrum_rxdB[localMaxInd])[-num_sources::]
                localMaxPeaks = localMaxInd[peakInd]
                estAzAngleSepDeg = np.abs(np.diff(angleGridRx[localMaxPeaks]))
                if (np.isnan(estAzAngleSepDeg)):
                    estAzAngleSepDeg = 250
            except IndexError:
                estAzAngleSepDeg = 250

            estAzAngleSepDegArr[ele_snr,ele_res,ele_mc] = estAzAngleSepDeg

            """  Estimated elevation resolution computation"""
            localMaxInd = argrelextrema(pseudo_spectrum_txdB,np.greater,axis=0,order=1)[0]
            try:
                peakInd = np.argsort(pseudo_spectrum_txdB[localMaxInd])[-num_sources::]
                localMaxPeaks = localMaxInd[peakInd]
                estElAngleSepDeg = np.abs(np.diff(angleGridTx[localMaxPeaks]))
                if (np.isnan(estElAngleSepDeg)):
                    estElAngleSepDeg = 250
            except IndexError:
                estElAngleSepDeg = 250

            estElAngleSepDegArr[ele_snr,ele_res,ele_mc] = estElAngleSepDeg


percentestAzAngSepArrMusic = np.percentile(estAzAngleSepDegArr,90,axis=2)
# percentestAzAngSepArrCapon = np.percentile(estAzAngleSepDegArrCapon,90,axis=2)

percentestElAngSepArrMusic = np.percentile(estElAngleSepDegArr,90,axis=2)
# percentestElAngSepArrCapon = np.percentile(estElAngleSepDegArrCapon,90,axis=2)


plt.figure(1,figsize=(20,10),dpi=200)
# plt.suptitle('Target SNR = ' + str(binSNRdBArray[0]) + ' dB')
plt.subplot(1,2,1)
plt.title('Azimuth 90 percentile separation')
plt.plot(azAngResDeg,percentestAzAngSepArrMusic.T, '-o', label='MUSIC', alpha=0.7)
# plt.plot(azAngResDeg,percent90estAngSepArrCapon.T, '-s', label='Capon', alpha=0.6)
plt.plot(azAngResDeg, azAngResDeg, color='k', label='Expectation')
plt.xlabel('GT angular separation (deg)')
plt.ylabel('estimated angular separation (deg)')
plt.ylim([0,np.ceil(azAngResDeg[-1])])
plt.grid(True)
plt.legend()


plt.subplot(1,2,2)
plt.title('Elevation 90 percentile separation')
plt.plot(elAngResDeg,percentestElAngSepArrMusic.T, '-o', label='MUSIC', alpha=0.7)
# plt.plot(azAngResDeg,percent90estAngSepArrCapon.T, '-s', label='Capon', alpha=0.6)
plt.plot(elAngResDeg, elAngResDeg, color='k', label='Expectation')
plt.xlabel('GT angular separation (deg)')
plt.ylabel('estimated angular separation (deg)')
# plt.axis([angSepDeg[0], angSepDeg[-1], angSepDeg[0], angSepDeg[-1]])
plt.ylim([0,np.ceil(elAngResDeg[-1])])
plt.grid(True)
plt.legend()




# plt.figure(1,figsize=(20,10),dpi=200)
# plt.subplot(1,2,1)
# plt.title('Num Rx samples = ' + str(numRx))
# plt.plot(angleGridRx, pseudo_spectrum_rxdB)
# plt.vlines(objectAzAngle_deg,-60,5, alpha=1,color='black',ls='dashed',label = 'Ground truth')
# plt.xlabel('Angle(deg)')
# plt.legend()
# plt.grid(True)

# plt.subplot(1,2,2)
# plt.title('Num Tx samples = ' + str(numTx))
# plt.plot(angleGridTx, pseudo_spectrum_txdB)
# plt.vlines(objectElAngle_deg,-60,5, alpha=1,color='black',ls='dashed',label = 'Ground truth')
# plt.xlabel('Angle(deg)')
# plt.legend()
# plt.grid(True)
