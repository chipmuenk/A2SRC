#!/usr/bin/env python
# -*- coding: utf-8 -*-
#===========================================================================
# pyaudio_numpy_example.py
#
# Einfaches Code-Beispiel zum Einlesen / Schreiben von WAV-Dateien mit 
# schneller numpy-Array Arithmetik
#
# Eine Audio-Datei wird frameweise eingelesen, in numpy-Arrays umgewandelt 
# dann werden die Daten prozessiert und blockweise auf ein Audio-Device 
# über pyAudio ausgegeben.
# 
#===========================================================================
#from __future__ import division, print_function, unicode_literals # v3line15

import numpy as np
import numpy.random as rnd
from numpy import (pi, log10, exp, sqrt, sin, cos, tan, angle, arange,
                    linspace, array, zeros, ones)
from numpy.fft import fft, ifft, fftshift, ifftshift, fftfreq
import scipy.signal as sig
import scipy.interpolate as intp

import matplotlib.pyplot as plt
from matplotlib.pyplot import (figure, plot, stem, grid, xlabel, ylabel,
    subplot, title, clf, xlim, ylim)

import my_dsp_lib as dsp
#------------------------------------------------------------------ v3line30
# Ende der gemeinsamen Import-Anweisungen
import pyaudio
import wave

np_type = np.int16 # numpy 16 bit integer for 16 bit WAV-Format
q_obj = (14, 0, 'round', 'wrap') # Define format for Fixpoint object 
# (14b integer, no fractional bits, rounding, wrap around )
# also try 'sat' instead of 'wrap' or varying the number of bits

# Framesize for reading, processing and writing. Larger values reduce
# read / write / array initializing overhead but increase memory consumption,
# latency and granularity:
CHUNK = 1024

wf_in = wave.open(r'C:\Windows\Media\chord.wav', 'r') # open WAV-File for reading
#wf_in = wave.open(r'D:\Musik\wav\Jazz\07 - Duet.wav')
#wf_in = wave.open(r'D:\Daten\share\Musi\wav\Feist - My Moon My Man.wav')

# Read properties of WAV-File
sampwidth = wf_in.getsampwidth() # number of bytes per sample
nchannels = wf_in.getnchannels() # number of channels
framerate = wf_in.getframerate() # framerate in Hz
nframes   = wf_in.getnframes()   # total number of samples
comptype  = wf_in.getcomptype()  # compression type: always 'NONE'
compname  = wf_in.getcompname()  # compression name ('not compressed')


wf_out = wave.open(r'D:\Daten\test.wav','w') # open WAV-File for writing
wf_out.setparams((nchannels, sampwidth, framerate, CHUNK, 
                  comptype, compname))

## Instantiate PyAudio + setup PortAudio system, print all available devices
p = pyaudio.PyAudio()
for i in range(p.get_device_count()):
    print (p.get_device_info_by_index(i))
    
# open a stream for reading or writing on the default audio device with 
# the audio parameters copied from the wave file wf_in:
stream = p.open(format=p.get_format_from_width(sampwidth),
                channels=nchannels,
                rate=framerate,
                output=True)

# allocate variables in memory for subsequent speed-up:    
samples_new = zeros(CHUNK*2, dtype=np_type)  #16 bit numpy array 
samples_np = zeros(CHUNK*2, dtype=np.float32) # float np array
samples_in = zeros(CHUNK*2, dtype=np_type) # float np array
samples_l = zeros(CHUNK, dtype=np_type) # 16 bit L-channel np arr.
samples_r = zeros(CHUNK, dtype=np_type) # 16 bit R-channel np arr.

i = 0
data_out = 'dummy'
while data_out: # read while there are samples

# read CHUNK frames to string and convert to numpy array:
    samples_in = np.fromstring(wf_in.readframes(CHUNK), dtype=np_type)
    i += CHUNK
    print(i, len(samples_in))
#    print(samples[0:10])

# R / L samples of WAV are interleaved, each sample is 16 bit = 2 Bytes
# Split array into an R and an L array to allow for easier processing:

## Sample = array element word length = 16 bit -> optimum!
## dtype = np.int16 (16 bits): 1 ndarray element = 1 sample :
    samples_l = samples_in[0::2] # take every 2nd sample, starting from 0
    samples_r = samples_in[1::2] # take every 2nd sample, starting from 1

## Array element = 8 bit word length, store LSB and MSB separately -> not good
## dtype = np.int8 (8 bits) = 1 ndarray element
##    two consecutive bytes / ndarray elements = 1 sample    
#    samples_l[0::2] = samples[0::4] # take every 4th sample, starting with 0
#    samples_l[1::2] = samples[1::4] # ...
#    samples_r[0::2] = samples[2::4]
#    samples_r[1::2] = samples[3::4]
## Do some numpy magic here
#             <...>
# Combine R and L channel into one array again
#    samples_new[0::4] = samples_l[0::2]
#    samples_new[1::4] = samples_l[1::2]
#    samples_new[2::4] = samples_r[0::2]
#    samples_new[3::4] = samples_r[1::2]

#---------------------------------------------------------------------------
# Now do some signal processing:
#---------------------------------------------------------------------------

## Examples for processing L and R channel separately:
# Swap L and R channel:
    if len(samples_r) < CHUNK: # check whether frame has full length, otherwise
    # the indexing operation will fail (len(samples_new)/2 > len(samples_r))
        samples_new = zeros(len(samples_in), dtype=np_type) 
    samples_new[0::2] = samples_r
    samples_new[1::2] = samples_l
    
## Examples for processing the stereo stream: 
#  This only works for sample-by-sample operations,
#  not e.g. for filtering where consecutive samples are combined
    
#    samples_new = samples # pass-through
#    samples_new, N_ov = dsp.fixed(q_obj, samples) # Fix-point conversion
#    samples_new = abs(samples) # I've got a fuzzbox and I'm gonna use it ...
#   Convert to float, square and calculate the root 
#    samples_new = sqrt(samples.astype(np.float32)**2 )

## Convert data back to string - attention: tostring() constructs a string
#  from the raw bytes - this is very fast, but input data needs to be in
#  16 bit format, so ensure correct format using .astype(np_type)!
    data_out = np.chararray.tostring(samples_new.astype(np_type))
    
#    data_out = wf.readframes(CHUNK) # direct streaming without numpy
    
## Play audio by writing audio data to the stream (blocking)
    stream.write(data_out)  # write to streaming device
    
## and / or write Data to WAV-File
    wf_out.writeframes(data_out) # write to WAV-File


stream.stop_stream() # pause audio stream
stream.close() # close audio stream
wf_in.close()  # close WAV-File
wf_out.close() # close WAV-File

p.terminate() # close PyAudio & terminate PortAudio system