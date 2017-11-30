#!/usr/bin/env python
# -*- coding: utf-8 -*-
#===========================================================================
# interpolate_compare.py
#
# Compare different types of interpolation
# 
#
#
# 
#
#
# (c) 2014-Feb-04 Christian Münker - Files zur Vorlesung "DSV auf FPGAs"
#===========================================================================
from __future__ import division, print_function, unicode_literals # v3line15

import numpy as np
import numpy.random as rnd
from numpy import (pi, log10, exp, sqrt, sin, cos, tan, angle, arange,
                    linspace, array, zeros, ones)
#from numpy.fft import fft, ifft, fftshift, ifftshift, fftfreq
import scipy.signal as sig
import scipy.interpolate as intp

try:
    import vispy.mpl_plot as plt
    VISPY = True
except ImportError:  
    import matplotlib.pyplot as plt
    VISPY = False
    
print(VISPY)
print(plt)
    
#from matplotlib.pyplot import (figure, plot, stem, grid, xlabel, ylabel,
#    subplot, title, clf, xlim, ylim)

import my_dsp_lib_v7 as dsp
#------------------------------------------------------------------------
# Ende der gemeinsamen Import-Anweisunge
from scipy.io import wavfile
import time
import sys
import os, psutil

def memory_usage():
    # return the memory usage of the python process in MB
    process = psutil.Process(os.getpid())
    mem = process.get_memory_info()[0] / float(2 ** 20)
    return mem

mem = [memory_usage()] # initialize list 
#==============================================================================
# Input / Output Files
#==============================================================================

mydir = "D:/Daten/share/Musi/wav/" # base directory for input / output files
file1 = "Feist - My Moon My Man_sel.wav"
file2 = "Feist - My Moon My Man.wav"
file3 = "Wipers-Nothing Left To Lose.wav"
file4 = "D:/Daten/design/python/git/A2SRC/A2SRC/M1F1-int16-AFsp.wav"
file5 = "chris jones.wav"

file_i = mydir+file2 # input file name
file_o = mydir+"1kHz_1UI_"+file2 # output file name

n_smp_max = 2e7 # specify max. number of samples to be read and interpolated
                # or to be generated and interpolated

#==============================================================================
# Synthetic Data
#==============================================================================
SYNTH_DATA = False # use synth. data or WAV-File
#
f_sig = 10.e3  # test tone frequency / max. frequency for chirp
A_sig = 1.    # test tone amplitude
n_smp_i = 2e6 # number of samples to be generated and interpolated
n_chan_i = 2  # number of channels
rate_i = 44100 # input sample rate in Hz
dtype_i = 'float32' # data type for generated data

A_mod = 1e-3
f_mod = 1e3
#==============================================================================
# Interpolation parameters
#==============================================================================
r = 1  # interpolation rate = ratio of output rate / input rate
rs = 1 # ratio of output samples / input samples
ip = 3 # Spline order for continuous interpolation
dtype_int = 'float64' # data type for internal representation
dtype_o = 'int16' # data type for output data (*.wav -> int16)

#==============================================================================
# Plotting Parameters
#==============================================================================
PLT_ENB = True  # enable plotting
PLT_ERR = True  # plot amplitude and time error 
#              (only useful for rate_i = rate_o and n_smp_i = n_smp_o)
PLT_JITTER = True # Plot resampled data against original time vector time_i.
                # This is useful to display the time displacement due to 
                # jitter. For different input / output sample rates, set
                # PLT_JITTER = FALSE
PLT_BEG = 1000000 # first (input) plot sample
PLT_END = 2000000 # last (input) plot sample
PLT_SPECGRM = True

NFFT = 8096     # FFT length for spectrogram and FFT
DB_MIN = -140   # lower display limit (dB) for spectrogram and FFT
DB_MAX = 0      # upper display limit (dB) for spectrogram and FFT

t_label = r'$t$ in s $\rightarrow$'  # time label
f_label = r'$f$ in Hz $\rightarrow$' # frequency label
H_label = r'$|H(e^{j \Omega})|$ in dB $\rightarrow$' # y-Axis label

t0 = time.time() # initialize time measurement

#==============================================================================
# Create / read data
#==============================================================================
if SYNTH_DATA:
    data_i = np.empty((n_smp_i, n_chan_i), dtype = dtype_i)
    time_i = linspace(0, n_smp_i / rate_i, n_smp_i)
    
    # various waveforms (both channels identical, select as you like)
# random input data with uniform distribution in the range [-1,1]
#    data_i[:,0] = data_i[:,1] = 2 * (rnd.random(n_smp_i) - 0.5)
#   data_i[:,0] = data_i[:,1] = sig.sawtooth(2 * pi * f_sig * time_i)#, width=0.5)
#    data_i[:,0] = data_i[:,1] = np.sin(2 * pi * f_sig * time_i)
    data_i[:,0] = data_i[:,1] = sig.chirp(time_i, 0, time_i[-1], f_sig)
#    data_i = np.sign(data_i) # convert the above to rect
    data_i *= 32767. # full scale for int16 
else:
    rate_i, data_i = wavfile.read(file_i) # returns np array with int16 
    n_smp_i = min(len(data_i), n_smp_max) # number of samples per channel
    n_chan_i = np.shape(data_i)[1] # number of channels 
    dtype_i = data_i.dtype # Data type (usually int16)

    data_i = data_i.astype('float') # convert to float for sufficient dynamic range
    time_i = linspace(0, n_smp_i / rate_i, n_smp_i)

Amax = np.amax(np.fabs(data_i))
print("INPUT:", n_smp_i, "x", n_chan_i,"Samples @", rate_i, "Hz")
print("Max. value: %d, data type: %s" %(Amax, dtype_i))

#==============================================================================
# Parameters for output samples
#==============================================================================
n_smp_o = round(rs * n_smp_i) # number of output samples
rate_o  = r * rate_i # output rate
data_o  = zeros((n_smp_o, n_chan_i), dtype = dtype_int) # initialize output data

mem.append(memory_usage()) # 1
print(mem, 'MB')

#==============================================================================
# Create and modulate time array 
#==============================================================================
time_new = linspace(time_i[0], time_i[-1], n_smp_o)

# 100 Hz sinusoidal PM Jitter with 5 UI amplitude:
ModSig1 = A_mod/rate_o*sin(2*pi * time_new * f_mod)
# gaussian distributed noise with a variance of 1
ModSig2 = rnd.randn(n_smp_o)

# Modulate sample times by adding modulation signal to time vector
time_new += ModSig1 # modulate new time vector, keep old one
#time_i += ModSig1 # modulate INPUT (original) time vector

t1 = time.time()
mem.append(memory_usage()) # 2
print("... initialized in %0.2f s using %0.2f MB" %((t1 - t0), mem[-1]))

###############################################################################
# Interpolate the input data 
###############################################################################
# TODO: currently only works for two channels
fu_l = intp.InterpolatedUnivariateSpline(time_i, data_i[:n_smp_i,0], k = ip)
fu_r = intp.InterpolatedUnivariateSpline(time_i, data_i[:n_smp_i,1], k = ip)

data_o[:,0] = fu_l(time_new) # Interpolate left and
data_o[:,1] = fu_r(time_new) # right channel

mem.append(memory_usage()) # 2
t2 = time.time()
print("... interpolated in %0.2f s using %0.2f MB" %(t2 - t1, mem[-1]))

#==============================================================================
# Write resampled data to WAV-File
#==============================================================================
wavfile.write(file_o, rate_o, data_o.astype('int32'))

###############################################################################
# Plot the data
###############################################################################
#
# Time Domain: Input / output samples and amplitude / time difference
#==============================================================================
#plt.close('all') # large plots = lots of memory ...
if PLT_ENB:
    fig1 = plt.figure(1)
    if PLT_ERR:
        ax1 = fig1.add_subplot(211)
    else:
        ax1 = fig1.add_subplot(111)
        ax1.set_xlabel(t_label)
    ax1.set_ylabel(r'Sample Amplitude $\rightarrow$')
    ax1.plot(time_i[PLT_BEG:PLT_END], data_i[:,0][PLT_BEG:PLT_END], 
     'ro', linestyle = ':', label = 'Original')     
    if PLT_JITTER:
    # Plot resampled data against ORIGINAL time vector time_i to show 
    # the time displacement (jitter)         
        ax1.step(time_i[r*PLT_BEG:r*PLT_END], 
             data_o[:,0][r*PLT_BEG:r*PLT_END], 
             'o', where='post', linestyle = '--', label = 'w/ Jitter',
             color = (0.,0.,1,0.5), markerfacecolor=(0.,0.,1,0.5))
    else:
    # Plot resampled data against NEW time vector time_new to show 
    # quality of resampling    
        ax1.step(time_new[r*PLT_BEG:r*PLT_END], 
            data_o[:,0][r*PLT_BEG:r*PLT_END], 
            'o', where='post', linestyle = '--', label = 'Resamp. Data',
            color = (0.,0.,1,0.5), markerfacecolor=(0.,0.,1,0.5))
    ax1.legend()
    
    if PLT_ERR: # assume r == 1
        ax21 = fig1.add_subplot(212, sharex=ax1) # lock x-Axes of both plots
        ax21.plot(time_new[PLT_BEG:PLT_END], 
             (data_o[:,0] - data_i[:,0])[PLT_BEG:PLT_END], 
              color = 'r', label = 'Amp. Error')
        ax22 = ax21.twinx() # second y-axis with separate scaling
        ax22.plot(time_new[PLT_BEG:PLT_END],
             (time_i -time_new)[PLT_BEG:PLT_END] * rate_o,
              color =(0.,0.,1,0.5), label = 'Time Error')
        ax21.set_xlabel(t_label)
        ax21.set_ylabel(r'Amplitude Error  $\rightarrow$')
        ax22.set_ylabel(r'Time Error (UI) $\rightarrow$')
    # legend cannot collect labels from different axes    
    # -> ask matplotlib for plotted objects and their labels
    # and display them in one legend box
        lines, labels = ax21.get_legend_handles_labels()
        lines2, labels2 = ax22.get_legend_handles_labels()
        ax22.legend(lines + lines2, labels + labels2)
#    fig1.tight_layout()
    
#==============================================================================
# Spectrogram
#==============================================================================
    if PLT_SPECGRM:
        plt.figure(2)
        #-----  Define windowing function for Spectrogram / FFT -------------------
        win = sig.windows.kaiser(NFFT,20) # kaiser window needs shape parameter
    #    win = sig.windows.boxcar(NFFT) # rectangular window
        
        # ----- Calculate Equivalent Noise Bandwidth + Coherent Gain --------------
        ENBW = len(win)*np.sum(win**2)/ np.sum(abs(win))**2
        CGain = np.sum(win)/len(win)
    
        # ----- Calculate and plot magnitude spectrogram ------------------------------------
    # TODO: Spectrogram is always scaled for a two-sided spectrum 
    # -> too low by a factor of two (- 3dB) for one-sided spectrum (except @ DC ...)
        Pxx, freqs, bins, im =\
                        plt.specgram(data_o[:,0][r*PLT_BEG:r*PLT_END]/(NFFT*CGain), 
                        NFFT=NFFT, Fs=rate_o, noverlap=None, mode = 'magnitude', 
                        window = win, scale = 'dB', vmin = DB_MIN, vmax = DB_MAX)
        # freqs: DFT frequencies, bins: time steps
        #                           optional: cmap=cm.gist_heat                           
        n_bins_FFT = len(bins)
        plt.xlabel(t_label)
        plt.ylabel(f_label)
        plt.xlim([0, r*(PLT_END - PLT_BEG)/rate_o])
        # TODO: x-Achsenbeschriftung muss um PLT_BEG / rate_o verschoben werden
        plt.ylim([0,rate_o/2])
        plt.colorbar(label = H_label)
#        plt.tight_layout()
#==============================================================================
# Single FFT window
#==============================================================================
        k_bin = n_bins_FFT/2 # select middle FFT bin to display
        fig3 = plt.figure(3)
        ax3 = fig3.add_subplot(111)
        ax3.plot(freqs, 20*log10(Pxx[:,k_bin]))
        
        ax3.set_xlabel(f_label)
        ax3.set_ylabel(H_label)
        ax3.set_title(r'$|H(e^{j 2 \pi f / f_S},\, t)|$ bei $t=%0.1f$ s' %(bins[k_bin]))
        ax3.set_ylim([DB_MIN, DB_MAX])
        ax3.set_xlim([0,rate_o/2])
        ax3.grid(True)
#        fig3.tight_layout()
#-------------------------------------------
    mem.append(memory_usage()) # 3
    print("... plotted in %0.2f s using %0.2f MB" %((time.time()-t2), mem[-1]))
    print("... total time: %0.2f s" %(time.time()-t0))
    if VISPY: plt.draw()
    else: plt.show()
    
# NOTE: show() has currently been overwritten to convert to vispy format, so:
# 1. It must be called to show the results, and
# 2. Any plotting commands executed after this will not take effect.
# We are working to remove this limitation.

# Missing:  plt.legend()
#           plt.tight_layout()
#           plt.specgram()

if __name__ == '__main__':
    fig1.show(True)