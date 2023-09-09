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
import warnings
warnings.filterwarnings("ignore")


def capon_method_marple_dev(datain,Nsen,Nfft):
    """
    capon_method_marple_dev

    capon_method_marple_dev returns the power spectrum uing Capon Marple algorithm (batch calling)

    Parameters:
    datain (2D array, complex64): Input time samples. [cases, time samples]
    Nsen (integer): No of time samples = datain.shape[1]
    Nfft (integer): No of frequency samples in pseudo - spectrum, typically 1024, 2048
    Ncas (integer): No of cases needing to be submitted to Capon = datain.shape[0]

    Returns:
    pssps_dBm (2D array, float32) : Power spectrum [cases, Nfft]

    """

    Ncas= datain.shape[0]
    pssp= np.zeros((Ncas,Nfft),dtype=np.float32)
    order = (Nsen*3)//4
    for i in range(Ncas): # this for loop needs to be handled in GPU
        psdx = stsminvar_marple(datain[i,:],order,1,Nfft)
        pssp[i,:]= psdx
    pssp_dBm= -10*np.log10(pssp)+10
    pssps_dBm= np.fft.fftshift(pssp_dBm,axes=1)

    return pssps_dBm


def stsminvar_marple(X,order,sampling,NFFT):
    """
    stsminvar_marple

    stsminvar_marple returns the power spectrum uing Capon Marple algorithm (basic engine)

    Parameters:
    X (array, complex64): Input time samples.
    order (integer): AR model order. This should be set as not exceeding 3/4 of the Nsen (no of samples)
    sampling (integer): sampling rate is set as 1
    NFFT (integer): No of frequency samples in pseudo - spectrum, typically 1024, 2048

    Returns:
    PSD (array, float32) : Power spectrum

    References:

    Super-Fast Algorithm for Minimum Variance (Capon) Spectral Estimation, S. Lawrence Marple, Majid Adeli et al.
    Asilomar 2010

    """

    gamma,err,A= myhayes_burg(X, order - 1)

    Am= np.fft.fft(A,NFFT)
    Bm= np.fft.fft(A*np.arange(order),NFFT)

    Amc= np.conj(Am)
    Bmc= np.conj(Bm)

    den= order*Am*Amc - Am*Bmc - Bm*Amc

    PSD= np.real(den)  # this is energy

    return PSD


def myhayes_burg(x,p):


    """
    myhayes_burg

    myhayes_burg returns the AR coefficients using Burg Algorithm (basic engine)

    Parameters:
    x (array, complex64): Input time samples.
    p (integer): AR model order. This should be set as not exceeding 3/4 of the Nsen (no of samples)

    Returns:
    gamma (array, float32) : Reflection Coefficients which is the parameter of the Lattice filter representation
    err (array, float32) : Error in the model for all model orers till p
    ap (array, complex64): AR model for p. That is np.dot( [1 ap1 ap2 ... app],[xn xn-1 xn-2 ... xn-p]). ap= [1 ap1 ap2 ... app]

    Only ap is used by the calling function.

    References:

    Hayes: page-319. The LD to compute AR coefficients from gamma is adopted from Orifanidis
    Optimum Signal Processing (http://www.ece.rutgers.edu/~orfanidi/osp2e). Page 195-196

    """

    N= x.size

# Initilizations

    ep= x[1:]
    em= x[0:-1]
    N +=-1

    gamma= np.zeros(p+1,dtype='complex64')
    err= np.zeros(p+1,dtype='complex64')
    A= np.eye(p+1,dtype='complex64') # A[-1,::-1] is the AR coeffecients

    for j in range(1,p+1,1):
        gamma[j]= -2*np.dot(em.conj(),ep) / (np.dot(ep.conj(),ep) + np.dot(em.conj(),em))

        temp1= ep + gamma[j]*em
        temp2= em + np.conj(gamma[j])*ep
        err[j]= np.dot(temp1.conj(),temp1) + np.dot(temp2.conj(),temp2)

        ep= temp1[1:]
        em= temp2[0:-1]

        A[j,0]= gamma[j]

        if (j>1):
            A[j,1:j]= A[j-1,0:j-1] +  gamma[j]* np.flipud(A[j-1,0:j-1].conj())

        N +=-1

    ap= A[-1,::-1]

    return gamma,err,ap


def music_snapshots(received_signal, num_sources, num_samples, digital_freq_grid):

    signal_length = received_signal.shape[0]
    # numSnapshots = received_signal.shape[1]

    # received_signal = np.flipud(received_signal) # Need to remove this compute and push the sign change to the peak picking
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

def mimoPhasorSynth(platform, lamda, objectAzAngle_rad, objectElAngle_rad):

    # numTx and numRx available as global variables
    txSeq = np.arange(numTx)

    if (platform == 'array_1'):
        # platform description for array_1 in docstring
        tx_Yoffset = 0#10e-3
        tx_ycordinates = tx_Yoffset + txSpacing*np.arange(numTx)

        rx_Xoffset = 0#20e-3
        rx_xcordinates = rx_Xoffset + rxSpacing*np.arange(numRx)

        physicalTxCordinates = np.zeros((numTx,3),dtype=np.float32) # [x,y,z]
        physicalTxCordinates[:,1] = np.flipud(tx_ycordinates)

        physicalRxCordinates = np.zeros((numRx,3),dtype=np.float32)
        physicalRxCordinates[:,0] = np.flipud(rx_xcordinates)

    elif (platform == 'array_2'):
        # platform description for array_2 in docstring
        """ Array 2"""
        physicalTxCordinates = np.array([[0,0.01,0],
                                         [0,0.017843,0],
                                         [0.005882,0.017843,0],
                                         [0.011764,0.017843,0]]) # [x,y,z]

        physicalRxCordinates = np.array([[0,0,0],
                                         [0,0.003921,0],
                                         [0.001960,0.003921,0],
                                         [0.003921, 0.003921, 0]])



    SeqBasedTxCordinates = physicalTxCordinates[txSeq,:]
    SeqBasedRxCordinates = np.copy(physicalRxCordinates)

    virtualArrayCordinates = SeqBasedTxCordinates[:,:,None] + SeqBasedRxCordinates.T[None,:,:] # [numTx, 3, numRx]
    virtualArrayCordinates = np.transpose(virtualArrayCordinates, (1,0,2)).reshape(3,numTx*numRx) # [3, numTx x numRx]

    # for debug
    # plt.scatter(physicalRxCordinates[:,0],physicalRxCordinates[:,1])
    # plt.scatter(physicalTxCordinates[:,0],physicalTxCordinates[:,1])
    # plt.scatter(virtualArrayCordinates[0,:],virtualArrayCordinates[1,:])
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

    return mimoPhasor, mimoPhasor_txrx


# np.random.seed(5)
plt.close('all')

platform = 'array_1'
if (platform == 'array_1'):
    numTx = 6
    numRx = 6
    numAzUla = 6
    numElUla = 6
elif (platform == 'array_2'):
    numTx = 4
    numRx = 4
    azUlaInd = np.array([5,6,7,9,10,11,13,14,15])
    elUlaInd = np.array([0,1,4,5])
    numAzUla = 9
    numElUla = 4


lightSpeed = 3e8
centerFreq = 76.5e9
lamda = lightSpeed/centerFreq
rxSpacing = lamda/2
fsRx = lamda/rxSpacing
txSpacing = lamda #lamda/2
fsTx = lamda/txSpacing
num_sources = 2
numPointAngle = 256

rxAngRes = np.arcsin(fsRx/numAzUla)*180/np.pi
txAngRes = np.arcsin(fsTx/numElUla)*180/np.pi

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
resol_fact = np.arange(0.1,2.1,0.1)#np.arange(0.1,2.9,0.1)#np.arange(0.1,1.1,0.1)
numResol = len(resol_fact)
snrArray = np.array([40])
numSNR = len(snrArray)
snrdelta = 0 #3 # This indicates by how much dB is the second target below the 1st target
""" Noise parameters"""
noiseFloordB = -100
noise_power_db = noiseFloordB + 10*np.log10(numTx) # change this from numTx
noise_variance = 10**(noise_power_db/10)
noise_sigma = np.sqrt(noise_variance)

digFreqResAz = resol_fact*((2*np.pi)/numAzUla)
azAngResDeg = np.arcsin((digFreqResAz/(2*np.pi))*fsRx)*180/np.pi

digFreqResEl = resol_fact*((2*np.pi)/numElUla)
elAngResDeg = np.arcsin((digFreqResEl/(2*np.pi))*fsTx)*180/np.pi


estAzAngleSepDegArr = np.zeros((numSNR,numResol,numMonteCarlo))
estElAngleSepDegArr = np.zeros((numSNR,numResol,numMonteCarlo))

estAzAngleSepDegArrCapon = np.zeros((numSNR,numResol,numMonteCarlo))
estElAngleSepDegArrCapon = np.zeros((numSNR,numResol,numMonteCarlo))
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

        _, mimoPhasor_txrx = mimoPhasorSynth(platform, lamda, objectAzAngle_rad, objectElAngle_rad) # numObj, numTx, numRx
        mimoPhasor = mimoPhasor_txrx
        angleSignal = np.sum(mimoPhasor * complex_signal_amplitudes[:,None,None],axis=0)

        for ele_mc in range(numMonteCarlo):
            wgn_noise = (noise_sigma/np.sqrt(2))*np.random.randn(numTx * numRx) + 1j*(noise_sigma/np.sqrt(2))*np.random.randn(numTx * numRx)
            angleSignalwithNoise = angleSignal + wgn_noise.reshape(numTx,numRx)

            if (platform == 'array_1'):
                """ MUSIC analysis"""
                pseudo_spectrum_rx = music_snapshots(angleSignalwithNoise.T, num_sources, numRx, digital_freq_grid)
                pseudo_spectrum_rx = pseudo_spectrum_rx/np.amax(pseudo_spectrum_rx)
                pseudo_spectrum_rxdB = 10*np.log10(pseudo_spectrum_rx)

                pseudo_spectrum_tx = music_snapshots(angleSignalwithNoise, num_sources, numTx, digital_freq_grid)
                pseudo_spectrum_tx = pseudo_spectrum_tx/np.amax(pseudo_spectrum_tx)
                pseudo_spectrum_txdB = 10*np.log10(pseudo_spectrum_tx)

                """ CAPON Analysis"""
                azSignal = np.conj(angleSignalwithNoise)[0,:][None,:]
                spectrum_capon_rx = capon_method_marple_dev(azSignal,numRx,numPointAngle)
                spectrum_capon_rx = spectrum_capon_rx[0,:]
                spectrum_capon_rx -= np.amax(spectrum_capon_rx)

                elSignal = np.conj(angleSignalwithNoise)[:,0][None,:]
                spectrum_capon_tx = capon_method_marple_dev(elSignal,numTx,numPointAngle)
                spectrum_capon_tx = spectrum_capon_tx[0,:]
                spectrum_capon_tx -= np.amax(spectrum_capon_tx)

            elif (platform == 'array_2'):

                mimoSignalFlatten = angleSignalwithNoise.flatten()

                azSignal = mimoSignalFlatten[azUlaInd]
                elSignal = mimoSignalFlatten[elUlaInd]
                """ MUSIC analysis"""
                pseudo_spectrum_rx = music_snapshots(np.conj(azSignal)[:,None], num_sources, numAzUla, digital_freq_grid)
                pseudo_spectrum_rx = pseudo_spectrum_rx/np.amax(pseudo_spectrum_rx)
                pseudo_spectrum_rxdB = 10*np.log10(pseudo_spectrum_rx)

                pseudo_spectrum_tx = music_snapshots(np.conj(elSignal)[:,None], num_sources, numElUla, digital_freq_grid)
                pseudo_spectrum_tx = pseudo_spectrum_tx/np.amax(pseudo_spectrum_tx)
                pseudo_spectrum_txdB = 10*np.log10(pseudo_spectrum_tx)

                """ CAPON Analysis"""
                azSignal = azSignal[None,:]
                spectrum_capon_rx = capon_method_marple_dev(azSignal,numAzUla,numPointAngle)
                spectrum_capon_rx = spectrum_capon_rx[0,:]
                spectrum_capon_rx -= np.amax(spectrum_capon_rx)

                elSignal = elSignal[None,:]
                spectrum_capon_tx = capon_method_marple_dev(elSignal,numElUla,numPointAngle)
                spectrum_capon_tx = spectrum_capon_tx[0,:]
                spectrum_capon_tx -= np.amax(spectrum_capon_tx)


            """ Local Maxima for MUSIC pseudo spectrum"""
            """  Estimated Azimuth resolution computation"""
            localMaxInd = argrelextrema(pseudo_spectrum_rxdB,np.greater,axis=0,order=1)[0]
            try:
                peakInd = np.argsort(pseudo_spectrum_rxdB[localMaxInd])[-num_sources::]
                localMaxPeaks = localMaxInd[peakInd]
                estAzAngleSepDeg = np.abs(np.diff(angleGridRx[localMaxPeaks]))
                if (np.isnan(estAzAngleSepDeg) or len(estAzAngleSepDeg)==0):
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
                if (np.isnan(estElAngleSepDeg) or len(estElAngleSepDeg)==0):
                    estElAngleSepDeg = 250
            except IndexError:
                estElAngleSepDeg = 250

            estElAngleSepDegArr[ele_snr,ele_res,ele_mc] = estElAngleSepDeg

            """ Local Maxima for Capon spectrum"""
            """  Estimated Azimuth resolution computation"""
            localMaxInd = argrelextrema(spectrum_capon_rx,np.greater,axis=0,order=1)[0]
            try:
                peakInd = np.argsort(spectrum_capon_rx[localMaxInd])[-num_sources::]
                localMaxPeaks = localMaxInd[peakInd]
                estAzAngleSepDeg = np.abs(np.diff(angleGridRx[localMaxPeaks]))
                if (np.isnan(estAzAngleSepDeg) or len(estAzAngleSepDeg)==0):
                    estAzAngleSepDeg = 250
            except IndexError:
                estAzAngleSepDeg = 250

            estAzAngleSepDegArrCapon[ele_snr,ele_res,ele_mc] = estAzAngleSepDeg

            """  Estimated elevation resolution computation"""
            localMaxInd = argrelextrema(spectrum_capon_tx,np.greater,axis=0,order=1)[0]
            try:
                peakInd = np.argsort(spectrum_capon_tx[localMaxInd])[-num_sources::]
                localMaxPeaks = localMaxInd[peakInd]
                estElAngleSepDeg = np.abs(np.diff(angleGridTx[localMaxPeaks]))
                if (np.isnan(estElAngleSepDeg) or len(estElAngleSepDeg)==0):
                    estElAngleSepDeg = 250
            except IndexError:
                estElAngleSepDeg = 250

            estElAngleSepDegArrCapon[ele_snr,ele_res,ele_mc] = estElAngleSepDeg


percentestAzAngSepArrMusic = np.percentile(estAzAngleSepDegArr,90,axis=2)
percentestAzAngSepArrCapon = np.percentile(estAzAngleSepDegArrCapon,90,axis=2)

percentestElAngSepArrMusic = np.percentile(estElAngleSepDegArr,90,axis=2)
percentestElAngSepArrCapon = np.percentile(estElAngleSepDegArrCapon,90,axis=2)


plt.figure(1,figsize=(20,10),dpi=200)
plt.suptitle('Target SNR = ' + str(snrArray[-1]) + ' dB')
plt.subplot(1,2,1)
plt.title('Azimuth')
plt.plot(azAngResDeg,percentestAzAngSepArrMusic.T, '-o', label='MUSIC', alpha=0.7)
plt.plot(azAngResDeg,percentestAzAngSepArrCapon.T, '-s', label='Capon', alpha=0.6)
plt.plot(azAngResDeg, azAngResDeg, color='k', label='Expectation')
plt.axvline(rxAngRes, alpha=1,color='black',ls='dashed',label = 'Native resolution')
plt.xlabel('GT angular separation (deg)')
plt.ylabel('estimated angular separation (deg)')
plt.ylim([0,np.ceil(azAngResDeg[-1])])
plt.grid(True)
plt.legend()


plt.subplot(1,2,2)
plt.title('Elevation')
plt.plot(elAngResDeg,percentestElAngSepArrMusic.T, '-o', label='MUSIC', alpha=0.7)
plt.plot(elAngResDeg,percentestElAngSepArrCapon.T, '-s', label='Capon', alpha=0.6)
plt.plot(elAngResDeg, elAngResDeg, color='k', label='Expectation')
plt.axvline(txAngRes, alpha=1,color='black',ls='dashed',label = 'Native resolution')
plt.xlabel('GT angular separation (deg)')
plt.ylabel('estimated angular separation (deg)')
# plt.axis([angSepDeg[0], angSepDeg[-1], angSepDeg[0], angSepDeg[-1]])
plt.ylim([0,np.ceil(elAngResDeg[-1])])
plt.grid(True)
plt.legend()


if 0:
    np.save('azAngResDeg_' + str(numAzUla) + '.npy', azAngResDeg)
    np.save('percentestAzAngSepArrMusic_' + str(numAzUla) + '.npy', percentestAzAngSepArrMusic)
    np.save('percentestAzAngSepArrCapon_' + str(numAzUla) + '.npy', percentestAzAngSepArrCapon)
    np.save('rxAngRes_' + str(numAzUla) + '.npy', rxAngRes)

    np.save('elAngResDeg_' + str(numElUla) + '.npy', elAngResDeg)
    np.save('percentestElAngSepArrMusic_' + str(numElUla) + '.npy', percentestElAngSepArrMusic)
    np.save('percentestElAngSepArrCapon_' + str(numElUla) + '.npy', percentestElAngSepArrCapon)
    np.save('txAngRes_' + str(numElUla) + '.npy', txAngRes)


plt.figure(2,figsize=(20,10),dpi=200)
plt.subplot(1,2,1)
plt.title('Azimuth Spectrum. Num Rx samples = ' + str(numRx))
plt.plot(angleGridRx, pseudo_spectrum_rxdB)
plt.plot(angleGridRx, spectrum_capon_rx)
plt.vlines(objectAzAngle_deg,-60,5, alpha=1,color='black',ls='dashed',label = 'Ground truth')
plt.xlabel('Angle(deg)')
plt.legend()
plt.grid(True)

plt.subplot(1,2,2)
plt.title('Elevation spectrum. Num Tx samples = ' + str(numTx))
plt.plot(angleGridTx, pseudo_spectrum_txdB)
plt.plot(angleGridTx, spectrum_capon_tx)
plt.vlines(objectElAngle_deg,-60,5, alpha=1,color='black',ls='dashed',label = 'Ground truth')
plt.xlabel('Angle(deg)')
plt.legend()
plt.grid(True)
