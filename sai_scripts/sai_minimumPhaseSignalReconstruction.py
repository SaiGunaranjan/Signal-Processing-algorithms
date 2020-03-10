# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 18:08:12 2020

@author: Sai Gunaranjan Pelluri
"""

""" Script to reconstruct a minimum phase signal from only is magnitude spectrum
Reference : https://en.wikipedia.org/wiki/Minimum_phase"""


import numpy as np;
import matplotlib.pyplot as plt

plt.close('all')

num_samp = 512; # 512
mean = 30; # 30
sigma = 2.5; # 2.5
oversampFact = 32; # 32
np.random.seed(11);
realValuedSignal = sigma*np.random.randn(num_samp) + mean; # Create a random signal
#realValuedSignal = 6*np.cos(2*np.pi*20*np.arange(num_samp)/num_samp)
scalingFact = 100;
Delta = scalingFact*np.sum(np.abs(realValuedSignal));
DeltaSignal = Delta*np.hstack((1,np.zeros(num_samp-1)));
minPhaserealValuedSignal = DeltaSignal + realValuedSignal; # Convert the signal to a minimum phase signal by adding a large value at the zeroth sample

# Compute the psd of the minimum phase signal
if 1: # FFT based method of computing psd
    minPhaserealValuedSignal_spectrum = np.fft.fft(minPhaserealValuedSignal,n=num_samp*oversampFact);
    minPhaserealValuedSignal_magSpectrum = np.abs(minPhaserealValuedSignal_spectrum)**2;
if 0: # Autocorrelation method of computing psd
    Rxx = np.correlate(minPhaserealValuedSignal,minPhaserealValuedSignal,mode='full');
    minPhaserealValuedSignal_magSpectrum = np.abs(np.fft.fft(Rxx,n=num_samp*oversampFact));
    
minPhaserealValuedSignal_logmagSpectrum = np.log(minPhaserealValuedSignal_magSpectrum)/2;    
"""Compute Hilbert transform of minPhaserealValuedSignal_logmagSpectrum """
# This is done in the frequency domain #
minPhaserealValuedSignal_logmagSpectrum_cepstrum = np.fft.fft(minPhaserealValuedSignal_logmagSpectrum,n=num_samp*oversampFact);

### Analytic signal based signal reconstruction ### 
"""Construct the complex cepstrum as an analytic signal with the phase of the min phase signal spectrum as a negative hilbert of the log magnitude spectrum """
if 0:
    ### If signal is causal is one domain, it is analytic in the other domain. For proof refer page 30 of Time Frequency analysis book by Cohen
    #It can be shown that for a minimum phase causal signal, the complex cepstrum is causal i.e. F{log(Xmin(omega))} is causal. Implies  log(Xmin(omega)) is analytic
    ## log(Xmin(omega)) = log(|Xmin(omega)}) + j*angle(Xmin(omega)) . This is an analytic signal. implies  angle(Xmin(omega)) = -Hilbert{log(|Xmin(omega)})}}
    """Construct the discrete Hilbert kernel """
    minusj = -1j*np.ones((oversampFact*num_samp)//2-2);
    plusj = 1j*np.ones((oversampFact*num_samp)//2-2);
    Hilbertkernel = np.hstack((0,minusj,1,0,plusj,1));
    """ phaseTheta is computed as a negative of the Hilbert transform of logpsd i.e minPhaserealValuedSignal_logmagSpectrum"""
    phaseTheta = -1*np.fft.ifft(Hilbertkernel*minPhaserealValuedSignal_logmagSpectrum_cepstrum,n=num_samp*oversampFact);
    minPhaseSignal_logSpectrum = minPhaserealValuedSignal_logmagSpectrum + 1j*phaseTheta; # construct the logspectrum of the reconstructed minphase signal
### Complex cepstrum based signal reconstruction ###
""" Construct the complex cepstrum of the min phase signal from the cepstrum of the min phase signal """   
if 1:
    # Let ifft{log(|Xmin(omega)})} = c[n] call it as the cepstrum
    # Complex cepstrum of log(Xmin(omega)) i.e. ifft{log(Xmin(omega))} (say x_hat[n])is causal and is obtained as 
    # x_hat[n] = 0 for n<0 (because it is causal);
    #          = c[n] for n=0;
    #          = 2*c[n] for n>0
    causal_zeros = np.zeros((minPhaserealValuedSignal_logmagSpectrum_cepstrum.shape[0]//2),dtype=np.complex64)
#    minPhaseSignal_logSpectrum_complexCepstrum = np.hstack((minPhaserealValuedSignal_logmagSpectrum_cepstrum[1], \
#                                                            2*minPhaserealValuedSignal_logmagSpectrum_cepstrum[2:minPhaserealValuedSignal_logmagSpectrum_cepstrum.shape[0]//2 +1], \
#                                                            causal_zeros));
    minPhaseSignal_logSpectrum_complexCepstrum = np.hstack((minPhaserealValuedSignal_logmagSpectrum_cepstrum[0], \
                                                            2*minPhaserealValuedSignal_logmagSpectrum_cepstrum[1::]));    
    minPhaseSignal_logSpectrum = np.fft.ifft(minPhaseSignal_logSpectrum_complexCepstrum, n=num_samp*oversampFact);
    
minPhaseSignal_recon = np.real(np.fft.ifft(np.e**(minPhaseSignal_logSpectrum),n=num_samp*oversampFact)[0:num_samp]); #reconstruct the min phase signal by taking ifft of the reconstructed min phase spectrum 
realValuedSignal_recon = minPhaseSignal_recon - DeltaSignal; # remove back the previously added delta signal to get back the original signal


plt.figure(1);
plt.subplot(2,2,1)
plt.plot(minPhaserealValuedSignal,'-o',label='True minPhase signal');
plt.plot(minPhaseSignal_recon,'-o',label='Reconst. minPhase signal', alpha=0.4);
plt.legend()
plt.grid(True);
plt.subplot(2,2,2);
plt.plot(realValuedSignal,'-o',label='True real signal');
plt.plot(realValuedSignal_recon,'-o',label='Reconst. realsignal',alpha= 0.4);
plt.grid(True);
plt.legend();
plt.subplot(2,2,3);
plt.title('Error in actual minimum phase signal and reconstructed minimum phase signal')
plt.plot(minPhaserealValuedSignal-minPhaseSignal_recon,'-o');
plt.ylim(-1/10,1/10);
plt.grid(True);
plt.subplot(2,2,4)
plt.title('Error in actual signal and reconstructed signal')
plt.plot(realValuedSignal-realValuedSignal_recon,'-o');
plt.ylim(-1/10,1/10);
plt.grid(True);

####################################################


