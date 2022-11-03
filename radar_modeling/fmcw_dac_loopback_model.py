# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 17:44:19 2022

@author: Sai Gunaranjan
"""


import numpy as np
import matplotlib.pyplot as plt

plt.close('all')
lightSpeed = 3e8 #mps
initial_phase_deg = 30
initial_phase_rad = initial_phase_deg*np.pi/180
numChirpSamples = 2048
adcSamplingRate = 56.25e6 #Hz
chirp_time = numChirpSamples/adcSamplingRate# 36.4e-6 # us
chirpBW = 2e9#250e6#2e9
slope = chirpBW/chirp_time #25e6/1e-6 # MHz/us
Fstart_hz = 10e9 # GHz

overSampFact = 1500

Fs = overSampFact*adcSamplingRate #50e9 # GHz
Ts = 1/Fs

DACFreqBin = 512 # ADC sampling rate scale
dacDC = 10

num_samples = np.int32(chirp_time//Ts) #2048
time_s = np.arange(num_samples)*Ts
freq_grid = np.arange(-num_samples//2, num_samples//2,1)*Fs/num_samples
fre_vs_time = slope*time_s + Fstart_hz
#chirp_phase = 2*np.pi*(0.5*slope*time_s**2 + Fstart_hz*time_s) + initial_phase_rad;
chirp_phase = 2*np.pi*np.cumsum(fre_vs_time)*Ts + initial_phase_rad
localOscillator = np.exp(1j*chirp_phase)
localOscillator_InstFreq = (np.diff(np.unwrap(np.angle(localOscillator)))/(2*np.pi))/Ts;
localOscillator_fft = np.fft.fft(localOscillator)/num_samples

DACSignalFreq = DACFreqBin*(adcSamplingRate/numChirpSamples) #Hz
DACSignal = 1*(np.exp(1j*2*np.pi*DACSignalFreq*time_s) + np.exp(-1j*2*np.pi*DACSignalFreq*time_s))/2 + dacDC

transmittedSignal = localOscillator*DACSignal

numTargets = 3
rangeRes = lightSpeed/(2*chirpBW) #m
targetDistances = np.array([0,2,10]) # in m
# targetDistances = np.array([0,0.01,26]) # in m
rangeBins = (targetDistances//rangeRes)
targetDelays = (2*targetDistances)/lightSpeed
delaySamples = np.round(targetDelays/Ts).astype(np.int32)

receivedSignalVec = np.zeros((numTargets,num_samples),dtype=np.complex128)
for sourceNum in np.arange(numTargets):
    receivedSignalVec[sourceNum,delaySamples[sourceNum]::] = transmittedSignal[0:num_samples-delaySamples[sourceNum]]

receivedSignal = np.sum(receivedSignalVec,axis=0)

# basebandSignal = receivedSignal * np.conj(localOscillator)
basebandSignal = localOscillator * np.conj(receivedSignal)
basebandSignalFFT = np.fft.fft(basebandSignal*np.hanning(num_samples))

downSampledSignal = basebandSignal[0::overSampFact]
adcSignal = downSampledSignal[0:numChirpSamples]
windowFunction = np.hanning(numChirpSamples)
windowedADCSignal = adcSignal*windowFunction

rangeFFTSignal = np.fft.fft(windowedADCSignal)/numChirpSamples
rangeSpectrum = 20*np.log10(np.abs(rangeFFTSignal))

plt.figure(1,figsize=(20,9))
plt.subplot(1,2,1)
plt.title('Local oscillator signal: Freq vs Time')
plt.plot(time_s/(1e-6),fre_vs_time/1e9)
plt.xlabel('Time (us)')
plt.ylabel('Freq (GHz)')
plt.grid(True)
plt.subplot(1,2,2)
plt.title('Local oscillator signal: Magnitude spectrum (dB)')
plt.plot(freq_grid/1e9, 20*np.log10(np.fft.fftshift(np.abs(localOscillator_fft))))
plt.xlabel('Freq(GHz)')
plt.grid(True)


plt.figure(2,figsize=(20,10))
plt.subplot(1,2,1)
plt.title('Range Spectrum before ADC sampling')
plt.plot(20*np.log10(np.abs(basebandSignalFFT[0:1000])))
plt.vlines(rangeBins,ymin=20,ymax=160)
plt.vlines(rangeBins+DACFreqBin,ymin=20,ymax=160)
plt.grid(True)
plt.subplot(1,2,2)
plt.title('Range Spectrum post ADC sampling')
plt.plot(rangeSpectrum[0:1000])
plt.vlines(rangeBins,ymin=-160,ymax=20)
plt.vlines(rangeBins+DACFreqBin,ymin=-160,ymax=20)
plt.grid(True)
