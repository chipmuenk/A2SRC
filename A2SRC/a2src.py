#!/usr/bin/env python
# -*- coding: utf-8 -*-
#===========================================================================
# interpolate_compare.py
#
# Interactive Jitter modulation of audio files and generated waveforms
# 
# See http://stackoverflow.com/questions/6783194/background-thread-with-qthread-in-pyqt
#
# Generate UI-File using 
# pyuic4 -o ui_a2src.py ui_a2src.ui
#
# (c) 2014-Feb-04 Christian Münker - Files zur Vorlesung "DSV auf FPGAs"
#===========================================================================
from __future__ import division, print_function, unicode_literals # v3line15

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


#import my_dsp_lib_v7 as dsp
#------------------------------------------------------------------------
# Ende der gemeinsamen Import-Anweisungen
from PyQt4 import QtGui, QtCore
from scipy.io import wavfile
import time
import sys
import os, psutil
import pyaudio
import wave

#from guiA2SRC import Ui_MainWindow as GUI
from ui_a2src import Ui_MainWindow as GUI
#from ui_main_window import Ui_MainWindow as GUI

#from Verstaerker_GUI import Ui_MainWindow as GUI

#==============================================================================
# DEFINES
#==============================================================================

      
def memory_usage():
    # return the memory usage of the python process in MB
    process = psutil.Process(os.getpid())
    mem = process.get_memory_info()[0] / float(2 ** 20)
    return mem

mem = [memory_usage()] # initialize list 

###############################################################################
#
# Global Constants

np_type = np.int16 # WAV = 16 bit signed per sample, 1 frame = R + L sample
# Synthetic Data
SYNTH_RATE_I = 44100 # sample rate in Hz for synthetic signals
dtype_i = 'float32' # data type for generated data
N_SYNTH_I = 2e6

#------------------------------------------------------------------------------
# Plotting Parameters

PLT_IP = True  # enable plotting of interpolated data
PLT_BEG = 50000 # first plot sample
PLT_END = 100000 # last plot sample

t_label = r'$t$ in s $\rightarrow$'  # time label
f_label = r'$f$ in Hz $\rightarrow$' # frequency label
H_label = r'$|H(e^{j \Omega})|$ in dB $\rightarrow$' # y-Axis label

#------------------------------------------------------------------------------
# Interpolation parameters

R = 1  # interpolation rate = ratio of output rate / input rate
rs = 1 # ratio of output samples / input samples
ip = 3 # Spline order for continuous interpolation
ip_fram = 3 #‘nearest’,‘zero’,‘linear’,‘slinear’,‘quadratic,‘cubic’, 1 ...
dtype_int = 'float64' # data type for internal representation

N_SMP_MAX = 2e7 # specify max. number of samples to be read and interpolated
    # or to be generated and interpolated


FR_PAD = 10 # beginning / end padding of input frame
CHUNK = 4000 # number of samples in one frame
FR_LEN_I = CHUNK - 2 * FR_PAD
#FR_LEN_I = 20 # length of one input data frame
# Parameters for output samples
fr_len_o    = R * FR_LEN_I # length of one output data frame
#rate_o  = R * SYNTH_RATE_I # output rate

#------------------------------------------------------------------------------
# Primary thread: Create the GUI and manage Qt signal & slots  
# Variables used by the second thread are stored as instance variables in
# self.sim
    
class JitterGUI(QtGui.QMainWindow, GUI):
    def __init__(self, parent=None):        
        super(JitterGUI, self).__init__(parent)
        self.setupUi(self)
        self.stop = False
        self.lock =QtCore.QReadWriteLock()
        self.path = QtCore.QDir.homePath()
        self.setWindowTitle("Jitter Simulator")

        self.sim = JitterSim(self.lock, self)
        
        self.setSynthParams()
        self.setModulationParams()
        self.setupAudio()
        self.sim.wavFileWrite = False
        
       
        """
        LAYOUT      
        """
        self.comboBoxInputType.clear()
        self.comboBoxInputType.addItem('WAV-File', 'file')
        self.comboBoxInputType.addItem('Record', 'record')
        #self.comboBoxInputType.addItem('Synthetic', 'synth')
        
        self.comboBoxSignalType.clear()
        self.comboBoxSignalType.addItem('Sinusoidal', 'sine')
        self.comboBoxSignalType.addItem('Chirp', 'chirp')
        self.comboBoxSignalType.addItem('Rectangular', 'rect')
        self.comboBoxSignalType.addItem('Random', 'rnd')
        
        self.comboBoxModulationType.clear()
        self.comboBoxModulationType.addItem('Passthrough', 'pass')
        self.comboBoxModulationType.addItem('None', 'none')
        self.comboBoxModulationType.addItem('DC', 'dc')
        self.comboBoxModulationType.addItem('Sinusoidal', 'sine')
        self.comboBoxModulationType.addItem('Rectangular', 'rect')
        self.comboBoxModulationType.addItem('Random', 'rnd')
        
        # ============== Signals & Slots ================================
        # GUI:
        self.pushButtonOpenReadFile.clicked.connect(self.openFile)
#        self.pushButtonPlot.clicked.connect(self.plotError)
    

        # This calls QThread.start() which in turn calls the sim.run() method
        # run() must not be called directly for threading
        self.pushButtonStart.clicked.connect(lambda: self.sim.start())
#        self.pushButtonStart.clicked.connect(lambda: self.sim.run())
        
        self.pushButtonStop.clicked.connect(self.sim.stop)

        self.checkBoxEnableModulation.clicked.connect(self.setModulationParams)
        self.comboBoxModulationType.activated.connect(self.setModulationParams)
        self.checkBoxModulationPhase.clicked.connect(self.setModulationParams)
        self.lineEditModulationFreq.editingFinished.connect(self.setModulationParams)
        self.lineEditModulationAmp.editingFinished.connect(self.setModulationParams)        

        self.comboBoxSignalType.activated.connect(self.setSynthParams)
        self.lineEditSynthFreq.editingFinished.connect(self.setSynthParams)
        self.lineEditSynthAmp.editingFinished.connect(self.setSynthParams)

        self.checkBoxEnableStreamOut.clicked.connect(self.setupAudio)
        self.checkBoxEnableFileWrite.clicked.connect(self.setupAudio)
        # Threads        

    def setupAudio(self):
        """
        Create and manage selection box for audio interfaces
        """
        deviceList = []
        self.comboBoxAudioOut.clear()
        self.comboBoxAudioIn.clear()
        self.p = pyaudio.PyAudio() # instantiate PyAudio, start PortAudio system + list devices
        defaultInIdx = self.p.get_default_input_device_info()['index']
        defaultOutIdx = self.p.get_default_output_device_info()['index']

        print("Defaultin", defaultInIdx)
        for i in range(self.p.get_device_count()):
             deviceList.append(self.p.get_device_info_by_index(i))
    
             print (deviceList[i])
             if deviceList[i]['maxInputChannels'] > 0:
                 if i == defaultInIdx:
                     self.comboBoxAudioIn.addItem('* '+deviceList[i]['name'], str(i))
                     defaultInBoxIdx = self.comboBoxAudioIn.currentIndex()
                 else:
                     self.comboBoxAudioIn.addItem(deviceList[i]['name'], str(i))
                     
#                 self.comboBoxAudioIn.setItemData(str(i))
             else:
                 if i == defaultOutIdx:
                     self.comboBoxAudioOut.addItem('* '+deviceList[i]['name'], str(i))
                     defaultOutBoxIdx = self.comboBoxAudioOut.currentIndex()
                 else:
                     self.comboBoxAudioOut.addItem(deviceList[i]['name'], str(i))   
        self.comboBoxAudioIn.setCurrentIndex(defaultInBoxIdx)
        self.comboBoxAudioOut.setCurrentIndex(defaultOutBoxIdx)
#        print("Default Output Device : %s" % self.p.get_default_output_device_info()['name'])
#        self.comboBoxAudioOut.addItems(deviceList)        

    def openFile(self):
        """
        Set input and output files - see also Summerfield p. 192 ff
        """

        dlg=QtGui.QFileDialog( self )

        self.sim.my_WAV_in_file = dlg.getOpenFileName(filter="WAV-Files (*.wav)\nAll files(*.*)", directory="D:/Daten/share/musi/wav", 
                caption = "Open WAV File")
        self.sim.my_WAV_out_file = os.path.splitext(self.sim.my_WAV_in_file)[0] \
                + "_new" + os.path.splitext(self.sim.my_WAV_in_file)[1]

        self.sim.wavFileWrite = False
            
                                                   
    def setSynthParams(self):
        self.sim.synthFreq = float(self.lineEditSynthFreq.text())
        self.sim.synthAmp  = float(self.lineEditSynthAmp.text()) / 100
        self.sim.SignalTypeTxt = self.comboBoxSignalType.currentText()
        self.stIdx =self.comboBoxSignalType.currentIndex()       
        self.sim.SignalType = str(self.comboBoxSignalType.itemData(self.stIdx))
        
        print(self.sim.synthFreq, self.sim.synthAmp, self.sim.SignalType, 
              self.sim.SignalTypeTxt)

    def setModulationParams(self):
        self.sim.modEn = self.checkBoxEnableModulation.isChecked
        self.sim.modFreq = float(self.lineEditModulationFreq.text())
        self.sim.modAmp  = float(self.lineEditModulationAmp.text())
        self.sim.modTypeTxt = self.comboBoxModulationType.currentText()
        self.mtIdx =self.comboBoxModulationType.currentIndex()       
        self.sim.modType = str(self.comboBoxModulationType.itemData(self.mtIdx))
        self.sim.modInPhase = self.checkBoxModulationPhase.isChecked()

        print(self.sim.modFreq, self.sim.modAmp, self.sim.modType, 
              self.sim.modInPhase)
              
    def plotinit(self):
        if self.checkBoxPlotEnable.isChecked():
            fig = figure(1)
            if self.checkBoxPlotError.isChecked():
                self.ax1 = fig.add_subplot(211)
            else:
                self.ax1 = fig.add_subplot(111)
                self.ax1.set_xlabel(t_label)
            self.ax1.set_ylabel(r'Sample Amplitude $\rightarrow$')
            
            self.plot_i, = self.ax1.plot(self.sim.time_i[PLT_BEG:PLT_END], 
                          self.sim.data_i[:,0][PLT_BEG:PLT_END], 
             'ro', linestyle = ':', label = 'Original')     
            if False: #PLT_JITTER: (currently deactivated)
            # Plot resampled data against ORIGINAL time vector time_i to show 
            # the time displacement (jitter)         
                self.ax1.step(self.sim.time_i[R*PLT_BEG:R*PLT_END], 
                     self.sim.data_o[:,0][R*PLT_BEG:R*PLT_END], 
                     'o', where='post', linestyle = '--', label = 'w/ Jitter',
                     color = (0.,0.,1,0.5), markerfacecolor=(0.,0.,1,0.5))
            else:
            # Plot resampled data against NEW time vector time_new to show 
            # quality of resampling    
                self.plot_o, = self.ax1.step(self.sim.time_new[R*PLT_BEG:R*PLT_END], 
                    self.sim.data_o[:,0][R*PLT_BEG:R*PLT_END], 
                    'o', where='post', linestyle = '--', label = 'Resamp. Data',
                    color = (0.,0.,1,0.5), markerfacecolor=(0.,0.,1,0.5))
            plt.legend()
            
            if self.checkBoxPlotError.isChecked: # assume r == 1
                self.ax21 = fig.add_subplot(212, sharex=self.ax1) # lock x-Axes of both plots
                self.ax21.plot(self.sim.time_new[PLT_BEG:PLT_END], 
                     (self.sim.data_o[:,0] - self.sim.data_i[:,0])[PLT_BEG:PLT_END], 
                      color = 'r', label = 'Amp. Error')
                self.ax22 = self.ax21.twinx() # second y-axis with separate scaling
                self.ax22.plot(self.sim.time_new[PLT_BEG:PLT_END],
                     (self.sim.time_i -self.sim.time_new)[PLT_BEG:PLT_END] * self.rate_o,
                      color =(0.,0.,1,0.5), label = 'Time Error')
                self.ax21.set_xlabel(t_label)
                self.ax21.set_ylabel(r'Amplitude Error  $\rightarrow$')
                self.ax22.set_ylabel(r'Time Error (UI) $\rightarrow$')
            # legend cannot collect labels from different axes    
            # -> ask matplotlib for plotted objects and their labels
            # and display them in one legend box
            lines, labels = self.ax21.get_legend_handles_labels()
            lines2, labels2 = self.ax22.get_legend_handles_labels()
            self.ax22.legend(lines + lines2, labels + labels2)
            plt.tight_layout()
        
              

#------------------------------------------------------------------------------      

class JitterSim(QtCore.QThread):
    def __init__(self, lock, parent=None):        
        super(JitterSim, self).__init__(parent)
        self.lock = lock
        self.stopped = False
        self.mutex = QtCore.QMutex()
        self.completed = False
        self.DEBUG = True
        
    def initialize(self):
        if self.isStopped:
            self.stopped = False        
            self.completed = False        

        
    def run(self):
        """
        run is an overloaded method from QThread. Errors occuring in this scope
        fail silently
        """
#        print("RUN!")
        self.initialize()
        self.initAudio()
    #    self.testPrint(5)
        self.play()
        self.haltAudio()
#        self.finished.emit(self.completed, self.index)
        print("FINISHED!")
        

    def stop(self):
        """
        Use QMutexLocker as a context manager: Lock self.mutex (= block it
        until it can obtain the lock) and unlock it when the control flow 
        leaves the 'with' scope
        """
        with QtCore.QMutexLocker(self.mutex):
            self.stopped = True
            
    def isStopped(self):
        try:
            self.mutex.lock()
            return self.stopped
        finally:
            self.mutex.unlock()
            
    def generateSynthSignal(self):
        self.data_i = np.empty((N_SYNTH_I, 2), dtype = dtype_i)
        self.time_i = linspace(0, N_SYNTH_I / SYNTH_RATE_I, N_SYNTH_I)
        if self.SignalType == 'sine':
            data = self.synthAmp*np.sin(2 * pi * self.synthFreq * self.time_i)
        elif self.SignalType == 'rnd':
            data = self.synthAmp * (rnd.random(N_SYNTH_I) - 0.5)
            # various waveforms (both channels identical, select as you like)
        # random input data with uniform distribution in the range [-1,1]
        #    data_i[:,0] = data_i[:,1] = 2 * (rnd.random(n_smp_i) - 0.5)
        #   data_i[:,0] = data_i[:,1] = sig.sawtooth(2 * pi * f_sig * time_i)#, width=0.5)

        self.data_i[:,0] = self.data_i[:,1] = data * 32767
        
    def initAudio(self):
        # initialize audio devices
        self.wf_in = wave.open(self.my_WAV_in_file, 'r') # open WAV-File for reading
        if self.wavFileWrite:
            self.wf_out = wave.open(self.my_WAV_out_file, 'w') # open WAV-File for writing
            self.wf_out.setparams(self.wf_in.getparams())         # with same parameters as input file

        rate_i = self.wf_in.getframerate()
        self.rate_o = rate_i * R
        self.n_smp_i = self.wf_in.getnframes()
        self.n_chan_i = self.wf_in.getnchannels()
        self.dtype_i = self.wf_in.getsampwidth()
        self.n_frames = int(np.ceil(self.n_smp_i / FR_LEN_I))
        self.n_smp_o = round(rs * self.n_smp_i) # number of output samples

        self.time_i   = arange(self.n_smp_i) / rate_i
        self.time_o = linspace(0, self.n_smp_o/self.rate_o, self.n_smp_o)
#        print("time_new", self.time_o[0:10])

#        self.time_o = linspace(self.time_i[0], self.time_i[-1], self.n_smp_o)

        print("Input File:  %s\nOutput File: %s\nSamples: %d\n\
        Rate_i: %.1f Channels: %d WL %d Bytes\n Rate_o: %.1f " \
        %(self.my_WAV_in_file, self.my_WAV_out_file, self.n_smp_i, rate_i, 
          self.n_chan_i, self.dtype_i, self.rate_o))
        
        self.n_smp_i = max(N_SMP_MAX, self.n_smp_i)
        
        self.p = pyaudio.PyAudio() # instantiate PyAudio, start PortAudio system + list devices
        for i in range(self.p.get_device_count()):
            print (self.p.get_device_info_by_index(i))
        print("Default Output Device : %s" % self.p.get_default_output_device_info()['name'])
        # open a stream on the default audio device with parameters of the WAV-File:
        self.stream = self.p.open(format=self.p.get_format_from_width(self.wf_in.getsampwidth()),
                        channels=self.wf_in.getnchannels(),
                        rate=self.wf_in.getframerate(),
                        output=True)         


    def play(self):
        #------- initialize numpy arrays for speed-up -------------------------
        # Read one input frame of interleaved stereo samples from file:
        samples_in = zeros(CHUNK*2, dtype=np_type)
        # Write one output frame of interleaved stereo samples to file and
        # audio stream; frames are shorter now because of padding:
        samples_out = zeros((CHUNK - 2 * FR_PAD)*2, dtype=np_type)
        print(np.shape(samples_in))
        # Input / Output samples, split into two channels (incl. padding):
        self.data_i = self.data_o = zeros((2, CHUNK), dtype = np_type)
        # Time vector for one frame 
        self.time_f_o = zeros((2, CHUNK))
        #----------------------------------------------------------------------
        
        # read WAV-file chunk by chunk        
        data = 'dummy'
        self.fr = 0 # frame zero
        while (data and not self.isStopped()): 
            # read frames into string and convert to numpy array & split channel
            #  until end of file or "STOP" is clicked,

            samples_in = np.fromstring(self.wf_in.readframes(CHUNK), dtype=np_type)
#            print(np.shape(samples_in))
            wf_in_pos = self.wf_in.tell() # read file index position

            if len(samples_in) < CHUNK * 2: # has current frame full length?
                # no, is it longer than the padding? 
                if len(samples_in) <= 2 * FR_PAD: break # no, exit the loop
                else: # frame has reduced length -> pre-allocate arrays again
                    samples_out = zeros(len(samples_in) - 4 * FR_PAD, dtype=np_type) 
                    self.data_i = self.data_o = \
                            zeros((2, len(samples_in)/2), dtype = np_type)
                    self.time_f_o = zeros((2, len(samples_in)/2))
#            else:
                # yes, frame has full length: set file index position back
            self.wf_in.setpos(wf_in_pos - 2* FR_PAD)
#            print(np.shape(samples_in), np.shape(self.data_i))
            # Deinterleave data by COPYing every 2nd value ...
            self.data_i[0] = samples_in[0::2] # ... starting with 0 (L)
            self.data_i[1] = samples_in[1::2] # ... starting with 1 (R)
#            print("IDs:", id(self.data_i), id(samples_in))
                
            if self.modType == "pass": # pass-through without interpolation
                samples_out[1::2] = self.data_i[0,FR_PAD:len(samples_out)/2+FR_PAD] # L Ch.
                samples_out[0::2] = self.data_i[1,FR_PAD:len(samples_out)/2+FR_PAD] # R Ch. 
            else:            
                self.interpolate_univar(DEBUG = False)
                # copy resampled data except first and last FR_PAD samples and 
                # interleave L + R          
                samples_out[1::2] = self.data_o[0, FR_PAD:
                                        len(samples_out)/2+FR_PAD] # L Ch.
                samples_out[0::2] = self.data_o[1, FR_PAD:
                                        len(samples_out)/2+FR_PAD] # R Ch.

            data = np.chararray.tostring(samples_out) # convert back to string
            self.stream.write(data) # play audio by writing audio data to the stream (blocking)
            if self.wavFileWrite:
                self.wf_out.writeframes(data) # and write to WAV-File
            self.fr += 1 # increase frame counter

    def testPrint(self, count):
        for x in range(count):
            print(x)


    def haltAudio(self):
        self.stop = True
        self.stream.stop_stream() # pause audio stream
        self.stream.close() # close audio stream
        self.wf_in.close()  # close input WAV-File
        if self.wavFileWrite:
            self.wf_out.close() # close output WAV-File

        self.p.terminate() # close PyAudio & terminate PortAudio system


    def interpolate_univar(self, DEBUG):
        
        #=====================================================================
        # Interpolate the input data frame by frame
        #=====================================================================
        fr_len = len(self.data_o[0])-2*FR_PAD # = FR_LEN_I except for last frame
        if self.fr == 0:
            smp_range_i = range(0,fr_len + 2 * FR_PAD) # TODO: 2 * FR_PAD is wrong
        else:
            smp_range_i = range(self.fr*fr_len-FR_PAD, (self.fr+1)*fr_len + FR_PAD)

        if DEBUG:
            print("Frame # %d w/len %d" %(self.fr, fr_len))
            print("Time Index:", min(smp_range_i), max(smp_range_i), len(smp_range_i))
            print("Time_i\n=================\nMin./Max :", 
                  min(self.time_i), max(self.time_i))
            print("Time_o\n=================\nMin./Max :", 
                  min(self.time_o), max(self.time_o))
            print("Time     :", self.time_o[0:10])


        time_f_i = self.time_i[smp_range_i] # time within frame (original)
        self.time_f_o[0] = self.time_f_o[1] = self.time_o[smp_range_i]

        if DEBUG:
            print("Time_f_o\n================\n", self.time_f_o[0][0:10])
            print("Min./Max.:", min(self.time_f_o[0]), 
                  max(self.time_f_o[0]),len(self.time_f_o[0]))
            print("DataType:", np.dtype(self.time_f_o[0,0]))


        ModSig = 0 # modType in {'none', 'pass'}  # zeros(len(smp_range_i))
        # sinusoidal PM Jitter
        if self.modType in {'sine', 'rect'}:
            ModSig = self.modAmp/self.rate_o * sin(2*pi * self.time_f_o[0] * self.modFreq)
            if self.modType == 'rect':
                ModSig = self.modAmp/self.rate_o * np.sign(ModSig)
        elif self.modType == 'dc':
            ModSig = self.modAmp/self.rate_o
        elif self.modType == 'rnd':
        # gaussian distributed noise with a variance of modAmp
            ModSig = self.modAmp/self.rate_o * rnd.randn(CHUNK)
        
        # Modulate sample times by adding modulation signal to time vector
        # keeping the original time vector
        if not self.modInPhase:
            self.time_f_o[0] += ModSig # modulation of left 
            self.time_f_o[1] += ModSig # and right channel in phase            
        #self.time_i += ModSig # modulate INPUT (original) time vector
        else:
            self.time_f_o[0] += ModSig # modulate new time vector, keep old one     
            self.time_f_o[1] -= ModSig # modulate new time vector, keep old one
        
        smp_range_i = range(len(self.data_i[0]))
#        bbox = [time_f_i[0], time_f_i[-1]]
#        bbox = [0,-1]
        if DEBUG:
            print("smp_range_i:", smp_range_i)
            print("time_f_i / data_i:", len(time_f_i[smp_range_i]), len(self.data_i[0][smp_range_i]))
            print(time_f_i[smp_range_i], self.data_i[0][smp_range_i])
            print(time_f_i, self.data_i[0])
        fu_l = intp.UnivariateSpline(
                time_f_i[smp_range_i], self.data_i[0][smp_range_i], 
                k = ip, s = 0) # bbox=bbox,
        fu_r = intp.UnivariateSpline(
                time_f_i[smp_range_i], self.data_i[1][smp_range_i], 
                k = ip, s = 0) # bbox=bbox,

        self.data_o[0] = fu_l(self.time_f_o[0]) # interpolate data at new 
        self.data_o[1] = fu_r(self.time_f_o[1]) # time points

    def interpolate_frame(self):
        
        #=====================================================================
        # Interpolate the input data frame by frame
        #=====================================================================

        smp_range_i = range(self.fr*FR_LEN_I-FR_PAD, (self.fr+1)*FR_LEN_I + FR_PAD)
        print(len(self.time_i[smp_range_i]), len(self.data_i[smp_range_i]))
        f=intp.interp1d(self.time_i[smp_range_i], self.data_i[smp_range_i], 
                        axis = 0, kind = ip_fram, assume_sorted = True)
        print(min(self.time_i[smp_range_i]))

        smp_range_o = range(self.fr*fr_len_o, (self.fr+1)*fr_len_o)
        print(min(self.time_o[smp_range_o]))
        self.data_o[smp_range_o] = f(self.time_o[smp_range_o])       

    def interpolate(self):
        
        #=====================================================================
        # Interpolate the input data frame by frame
        #=====================================================================
#        fu_l = intp.InterpolatedUnivariateSpline(time_i, data_i[:n_smp_i,0], k = ip)
         # initialize array for output data :
  
        for i in range(self.n_frames):
    
            if i == 0: # first frame, don't try to access elements before first one
                f=intp.interp1d(self.time_i[i*FR_LEN_I:(i+1)*FR_LEN_I+FR_PAD], 
                            self.data_i[i*FR_LEN_I:(i+1)*FR_LEN_I+FR_PAD], 
                                     kind = ip_fram, assume_sorted = True)
                self.data_o[i*self.fr_len_o:(i+1)*self.fr_len_o] \
                    = f(self.time_o[i*self.fr_len_o:(i+1)*self.fr_len_o])
            else:
                f=intp.interp1d(self.time_i[i*FR_LEN_I-FR_PAD:(i+1)*FR_LEN_I+FR_PAD], 
                                self.data_i[i*FR_LEN_I-FR_PAD:(i+1)*FR_LEN_I+FR_PAD], 
                                         kind = ip_fram, assume_sorted = True)
                self.data_o[i*self.fr_len_o:(i+1)*self.fr_len_o] \
                    = f(self.time_o[i*self.fr_len_o:(i+1)*self.fr_len_o])

    
#------------------------------------------------------------------------------
 

  
if __name__ == '__main__':

    app = QtGui.QApplication(sys.argv)
    form = JitterGUI()
    form.show()
   
    app.exec_()