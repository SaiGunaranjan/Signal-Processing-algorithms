# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 21:06:12 2019

@author: Sai Gunaranjan Pelluri
"""

import sys
sys.path.append("..")
import numpy as np
import spectral_estimation_lib as spec_est
import matplotlib.pyplot as plt
from time import time






plt.close('all')
num_samples = 1024
num_sources = 3
object_snr = np.array([20,20,20])
noise_power_db = -40 # Noise Power in dB
noise_variance = 10**(noise_power_db/10)
noise_sigma = np.sqrt(noise_variance)
weights = noise_sigma*10**(object_snr/10)
source_freq = np.random.uniform(low=-np.pi, high=np.pi, size = num_sources)
source_signals = np.matmul(np.exp(-1j*np.outer(np.arange(num_samples),source_freq)),weights[:,None])
wgn_noise = np.random.normal(0,noise_sigma/np.sqrt(2),source_signals.shape) + 1j*np.random.normal(0,noise_sigma/np.sqrt(2),source_signals.shape)
received_signal = source_signals + wgn_noise
corr_mat_model_order = 100 # must be strictly less than num_samples/2
digital_freq_grid = np.arange(-np.pi,np.pi,2*np.pi/num_samples)

if 0:
    #pseudo_spectrum = spec_est.music_toeplitz(received_signal, num_sources, digital_freq_grid)
#    pseudo_spectrum = spec_est.music_forward(received_signal, num_sources, corr_mat_model_order, digital_freq_grid)
    pseudo_spectrum = spec_est.music_backward(received_signal, num_sources, corr_mat_model_order, digital_freq_grid)
    print('\n\n')
    print('True Frequencies:', source_freq, 'Estimated Frequncies:', digital_freq_grid[np.argsort(pseudo_spectrum)[-num_sources::]])
    plt.figure(1)
    plt.title('Pseudo Spectrum from MUSIC')
    plt.plot(digital_freq_grid,10*np.log10(pseudo_spectrum),'o-',alpha=0.7)
    plt.vlines(-source_freq,-30,20)
    plt.xlabel('Digital Frequencies')
    plt.legend(['Estimated Pseudo Spectrum', 'Ground Truth'])
    plt.grid(True)

if 0: 
    #est_freq = spec_est.esprit_toeplitz(received_signal, num_sources)
#    est_freq = spec_est.esprit_forward(received_signal, num_sources, corr_mat_model_order)
    est_freq = spec_est.esprit_backward(received_signal, num_sources, corr_mat_model_order)
    print('True Frequencies:', source_freq, 'Estimated Frequncies:', -est_freq)
if 0:
#    psd_f = spec_est.capon_forward(received_signal, corr_mat_model_order, digital_freq_grid)
    psd_b = spec_est.capon_backward(received_signal, corr_mat_model_order, digital_freq_grid)
    #psd, digital_freq_grid = capon_toeplitz(received_signal)
    plt.figure(2)
    plt.title('Power spectral density from Capon')
#    plt.plot(digital_freq_grid,10*np.log10(psd_f),'o-',alpha=0.7, label = 'CAPON Forward')
    plt.plot(digital_freq_grid,10*np.log10(psd_b),'.-',alpha=0.7, label = 'CAPON Backward')
    plt.vlines(-source_freq,-80,10)
    plt.plot(digital_freq_grid, 20*np.log10(np.abs(np.fft.fftshift(np.fft.fft(received_signal,axis=0,n=digital_freq_grid.shape[0])/len(received_signal),axes=0))),'.-', label = 'FFT')
    plt.legend()
    plt.xlabel('Digital Frequencies')
    
    plt.grid(True)
    
    
#### Resolution analisis of Apes vs FFT vs approximate non-recursive IAA vs recursive IAA   
if 1:
    plt.close('all')
    num_samples = 32
    num_sources = 2
    object_snr = np.array([40,40])
    noise_power_db = -40 # Noise Power in dB
    noise_variance = 10**(noise_power_db/10)
    noise_sigma = np.sqrt(noise_variance)
    weights = noise_variance*(10**(object_snr/10))
    signal_phases = np.exp(1j*np.random.uniform(low=-np.pi, high = np.pi, size = num_sources))
    complex_signal_amplitudes = weights*signal_phases
    random_freq = np.random.uniform(low=-np.pi, high=np.pi, size = 1)
    fft_resol_fact = 2
    resol_fact = 0.53#0.65
    source_freq = np.array([random_freq, random_freq + resol_fact*np.pi/num_samples])
    digital_freq_grid = np.arange(-np.pi,np.pi,2*np.pi/(100*num_samples))
    source_signals = np.matmul(np.exp(-1j*np.outer(np.arange(num_samples),source_freq)),complex_signal_amplitudes[:,None])
    wgn_noise = np.random.normal(0,noise_sigma/np.sqrt(2),source_signals.shape) + 1j*np.random.normal(0,noise_sigma/np.sqrt(2),source_signals.shape)
    received_signal = source_signals + wgn_noise
    corr_mat_model_order = num_samples//2-2 # must be strictly less than num_samples/2
    
    # FFT
    magnitude_spectrum_fft = 20*np.log10(np.abs(np.fft.fftshift(np.fft.fft(received_signal,axis=0, n=len(digital_freq_grid))/received_signal.shape[0],axes=0)))
    phase_spectrum_fft = np.unwrap(np.angle(np.fft.fftshift(np.fft.fft(received_signal,axis=0, n=len(digital_freq_grid))/received_signal.shape[0],axes=0)))
    
    # CAPON
    psd_capon = spec_est.capon_backward(received_signal, corr_mat_model_order, digital_freq_grid)
    
    # APES
    spectrum = spec_est.apes(received_signal, corr_mat_model_order, digital_freq_grid)
    magnitude_spectrum = np.abs(spectrum)
    phase_spectrum = np.unwrap(np.angle(spectrum))
    
    # IAA Non-Recursive
    spectrum_iaa_nr = spec_est.iaa_approx_nonrecursive(received_signal, digital_freq_grid) # non-recursive IAA
#    spectrum_iaa_nr = np.flipud(spectrum_iaa_nr)
    magnitude_spectrum_iaa_nr = np.abs(spectrum_iaa_nr)
    
    # IAA Recursive
    iterations = 10
    spectrum_iaa = spec_est.iaa_recursive(received_signal, digital_freq_grid, iterations) # recursive IAA
#    spectrum_iaa = spec_est.iaa_recursive_levinson_temp(received_signal, digital_freq_grid, iterations) # under debug
    magnitude_spectrum_iaa = np.abs(spectrum_iaa)    
    
    
    print('\nResolution improvement with respect to FFT = {}'.format(np.round(10*(fft_resol_fact/resol_fact))/10))
    
    apes_est_freq_ind = np.argsort(magnitude_spectrum,axis=0)[-num_sources::]
    estimated_complex_signal_amplitudes = spectrum[apes_est_freq_ind]
    
    iaa_nr_est_freq_ind = np.argsort(magnitude_spectrum_iaa_nr,axis=0)[-num_sources::]
    estimated_complex_signal_amplitudes_iaa_nr = spectrum_iaa_nr[iaa_nr_est_freq_ind]
    
    iaa_est_freq_ind = np.argsort(magnitude_spectrum_iaa,axis=0)[-num_sources::]
    estimated_complex_signal_amplitudes_iaa = spectrum_iaa[iaa_est_freq_ind]
    
    
    print('\nTrue Signal Amplitudes:{}, APES based Signal Amplitudes: {}'.format(np.abs(complex_signal_amplitudes),np.abs(estimated_complex_signal_amplitudes)))
    print('\nTrue Signal Phases(deg):{}, APES based Signal Phases(deg): {}'.format(np.angle(signal_phases)*180/np.pi,np.angle(estimated_complex_signal_amplitudes)*180/np.pi))
    
    print('\nTrue Signal Amplitudes:{}, Approx IAA non-recursive based Signal Amplitudes: {}'.format(np.abs(complex_signal_amplitudes),np.abs(estimated_complex_signal_amplitudes_iaa_nr)))
    print('\nTrue Signal Phases(deg):{}, Approx IAA non-recursive based Signal Phases(deg): {}'.format(np.angle(signal_phases)*180/np.pi,np.angle(estimated_complex_signal_amplitudes_iaa_nr)*180/np.pi))
    
    print('\nTrue Signal Amplitudes:{}, IAA recursive based Signal Amplitudes: {}'.format(np.abs(complex_signal_amplitudes),np.abs(estimated_complex_signal_amplitudes_iaa)))
    print('\nTrue Signal Phases(deg):{}, IAA recursive based Signal Phases(deg): {}'.format(np.angle(signal_phases)*180/np.pi,np.angle(estimated_complex_signal_amplitudes_iaa)*180/np.pi))
    
    
    plt.figure(3)
    plt.title('Magnitude Spectrum')
    plt.plot(digital_freq_grid, 20*np.log10(magnitude_spectrum),alpha=0.7, label = 'APES')
    plt.plot(digital_freq_grid, 20*np.log10(magnitude_spectrum_iaa_nr),alpha=0.7, label = 'approx non-recursive IAA')
    plt.plot(digital_freq_grid, 20*np.log10(magnitude_spectrum_iaa),alpha=0.7, label = 'Recursive IAA')
    plt.plot(digital_freq_grid, 20*np.log10(psd_capon), label = 'CAPON')
    plt.plot(digital_freq_grid, magnitude_spectrum_fft, label = 'FFT')
    plt.vlines(-source_freq,-150,25, label = 'Ground truth')
    plt.xlabel('Digital Frequencies')
    plt.legend()    
    plt.grid(True)
    
#    plt.figure(4)
#    plt.title('Phase Spectrum from APES')
#    plt.plot(digital_freq_grid, phase_spectrum,'o-',alpha=0.7, label = 'Est PSD from APES')
#    plt.plot(digital_freq_grid, phase_spectrum_fft,'o-', label = 'FFT Power Specrum')
#    plt.vlines(-source_freq,-150,25, label = 'Ground truth')
#    plt.xlabel('Digital Frequencies')
#    plt.legend()


## IAA approx_non-recursive
if 0:
    plt.close('all')
    num_samples = 1024
    num_sources = 3
    object_snr = np.array([20,20,20])
    noise_power_db = -40 # Noise Power in dB
    noise_variance = 10**(noise_power_db/10)
    noise_sigma = np.sqrt(noise_variance)
    weights = noise_variance*10**(object_snr/10)
    source_freq = np.random.uniform(low=-np.pi, high=np.pi, size = num_sources)
    source_signals = np.matmul(np.exp(-1j*np.outer(np.arange(num_samples),source_freq)),weights[:,None])
    wgn_noise = np.random.normal(0,noise_sigma/np.sqrt(2),source_signals.shape) + 1j*np.random.normal(0,noise_sigma/np.sqrt(2),source_signals.shape)
    received_signal = source_signals + wgn_noise
    corr_mat_model_order = 100 # must be strictly less than num_samples/2
    
    
    digital_freq_grid = np.arange(-np.pi,np.pi,2*np.pi/(100*num_samples))    
    spectrum = spec_est.iaa_approx_nonrecursive(received_signal, digital_freq_grid) # non-recursive IAA
    magnitude_spectrum = np.abs(spectrum)

    plt.figure(3)
    plt.title('Magnitude Spectrum')
    plt.plot(digital_freq_grid, 20*np.log10(magnitude_spectrum),alpha=1, label = 'Approx recursive IAA (compute heavy)')
    plt.vlines(-source_freq,-150,25, label = 'Ground truth')
    plt.xlabel('Digital Frequencies')
    plt.legend()    
    plt.grid(True)


### Approximate recursive IAA compute heavy
if 0:
    plt.close('all')
    num_samples = 512#1024
    num_sources = 3
    object_snr = np.array([20,20,20])
    noise_power_db = -40 # Noise Power in dB
    noise_variance = 10**(noise_power_db/10)
    noise_sigma = np.sqrt(noise_variance)
    weights = noise_variance*10**(object_snr/10)
    source_freq = np.random.uniform(low=-np.pi, high=np.pi, size = num_sources)
    source_signals = np.matmul(np.exp(-1j*np.outer(np.arange(num_samples),source_freq)),weights[:,None])
    wgn_noise = np.random.normal(0,noise_sigma/np.sqrt(2),source_signals.shape) + 1j*np.random.normal(0,noise_sigma/np.sqrt(2),source_signals.shape)
    received_signal = source_signals + wgn_noise
    corr_mat_model_order = 100 # must be strictly less than num_samples/2
    
    
#    digital_freq_grid = np.arange(-np.pi,np.pi,2*np.pi/(100*num_samples))    
#    spectrum = spec_est.iaa_approx_nonrecursive(received_signal, digital_freq_grid) # non-recursive IAA
#    magnitude_spectrum = np.abs(spectrum)
    iterations = 10
    digital_freq_grid = np.arange(-np.pi,np.pi,2*np.pi/(10*num_samples))    
    spectrum = spec_est.iaa_approx_recursive_computeheavy(received_signal, digital_freq_grid,iterations) # non-recursive IAA
    magnitude_spectrum = np.abs(spectrum)
    plt.figure(3)
    plt.title('Magnitude Spectrum')
    plt.plot(digital_freq_grid, 20*np.log10(magnitude_spectrum),alpha=1, label = 'Approx non-recursive IAA')
    plt.vlines(-source_freq,-150,25, label = 'Ground truth')
    plt.xlabel('Digital Frequencies')
    plt.legend()    
    plt.grid(True)
    
    

### recursive IAA 
if 0:
    plt.close('all')
    num_samples = 1024
    num_sources = 3
    object_snr = np.array([20,20,20])
    noise_power_db = -40 # Noise Power in dB
    noise_variance = 10**(noise_power_db/10)
    noise_sigma = np.sqrt(noise_variance)
    weights = noise_variance*10**(object_snr/10)
    source_freq = np.random.uniform(low=-np.pi, high=np.pi, size = num_sources)
    source_signals = np.matmul(np.exp(-1j*np.outer(np.arange(num_samples),source_freq)),weights[:,None])
    wgn_noise = np.random.normal(0,noise_sigma/np.sqrt(2),source_signals.shape) + 1j*np.random.normal(0,noise_sigma/np.sqrt(2),source_signals.shape)
    received_signal = source_signals + wgn_noise
    corr_mat_model_order = 100 # must be strictly less than num_samples/2
    
    
#    digital_freq_grid = np.arange(-np.pi,np.pi,2*np.pi/(100*num_samples))    
#    spectrum = spec_est.iaa_approx_nonrecursive(received_signal, digital_freq_grid) # non-recursive IAA
#    magnitude_spectrum = np.abs(spectrum)
    iterations = 10
    digital_freq_grid = np.arange(-np.pi,np.pi,2*np.pi/(100*num_samples))    
    spectrum = spec_est.iaa_recursive(received_signal, digital_freq_grid,iterations) # non-recursive IAA
    magnitude_spectrum = np.abs(spectrum)
    plt.figure(3)
    plt.title('Magnitude Spectrum')
    plt.plot(digital_freq_grid, 20*np.log10(magnitude_spectrum),alpha=1, label = 'recursive IAA')
    plt.vlines(-source_freq,-150,25, label = 'Ground truth')
    plt.xlabel('Digital Frequencies')
    plt.legend()    
    plt.grid(True)    




