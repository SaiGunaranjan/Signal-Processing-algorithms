# -*- coding: utf-8 -*-
"""
Created on Wed Dec 15 16:43:01 2021

@author: saiguna
"""


""" For a complex signal of the form V*exp(1j*wt), Amplitude of the signal after taking FFT and normalizing is V.
 So, the power is V**2. But if the signal is real of the form V*cos(wt), we have a tone at +/-w
 with strength = V/2. If we assume only the positive side is relevent and throw away the other half,
 we essentially have a tone of strength V/2 at w. So the power is (V/2)**2 = V**2 / 4.
 So the power delta in dB scale is 6 dB. Hence, going from complex to a real only ADC,
 the signal power is lower by 6dB. What about the noise? If we assume a noise of power sigma**2,
 it gets distributed as sigma**2/2 for the real part and sigma**2/2 for the imaginary part.
 So if the ADC is realy only ADC, we only have the sigma**2/2 which is actually half the noise power.
 So, on the dB scale, the noise power for the real only chanel is 3 dB lower
 as compared to the complex ADC channel.
 In summary, a complex ADC has the signal boosed by 6 dB but noise floor also increases by 3 dB
 (since we have double the noise power in the complex channel).
 So we essentially get a 3 dB SNR improvement by going from a real ADC to a complex ADC"""


import numpy as np
import matplotlib.pyplot as plt


plt.close('all')

signalVoltage = 1
numSamples = 1024
bin_num = np.random.randint(0,numSamples) + np.random.uniform(-0.5, 0.5)
signalComplex = signalVoltage*np.exp(1j*2*np.pi*bin_num/numSamples * np.arange(numSamples))
thermalNoise = -174 # dBm/Hz
rxGain = 44# dB
adcSamplingRate = 112.5e6 # MHz
noisePower_dBm = thermalNoise + rxGain + 10*np.log10(adcSamplingRate)
noisePower = (10**(noisePower_dBm/10)) * 1e-3
noiseSigma = np.sqrt(noisePower)
noiseComplex = noiseSigma/np.sqrt(2) * np.random.randn(numSamples) + 1j*(noiseSigma/np.sqrt(2) * np.random.randn(numSamples))
# noisePowerPerBin_dBm = noisePower_dBm - 10*np.log10(numSamples)
noisySignalComplex = signalComplex + noiseComplex
noisySignalReal = np.real(noisySignalComplex)

noisySignalComplex_fft = np.fft.fft(noisySignalComplex*np.hanning(numSamples))
noisySignalReal_fft = np.fft.fft(noisySignalReal*np.hanning(numSamples))

estSignalPowerComplexSignal = 20*np.log10(np.amax(np.abs(noisySignalComplex_fft)))
estSignalPowerRealSignal = 20*np.log10(np.amax(np.abs(noisySignalReal_fft)))

estNoisePowerComplexSignal = 10*np.log10(np.mean((np.sort(np.abs(noisySignalComplex_fft))[512])**2))
estNoisePowerRealSignal = 10*np.log10(np.mean((np.sort(np.abs(noisySignalReal_fft))[512])**2))

print('Signal Floor for Complex Signal = {} dBFs'.format(np.round(estSignalPowerComplexSignal)))
print('Signal Floor for Real Signal = {} dBFs'.format(np.round(estSignalPowerRealSignal)))

print('Noise Floor for Complex Signal = {} dBFs'.format(np.round(estNoisePowerComplexSignal)))
print('Noise Floor for Real Signal = {} dBFs'.format(np.round(estNoisePowerRealSignal)))

plt.figure(1, figsize=(20,10))
plt.plot(20*np.log10(np.abs(noisySignalComplex_fft)), label='Complex Signal')
plt.plot(20*np.log10(np.abs(noisySignalReal_fft)), label='Real Signal')
plt.xlabel('bin number')
plt.ylabel('Power (dBFs)')
plt.legend()
plt.grid(True)

