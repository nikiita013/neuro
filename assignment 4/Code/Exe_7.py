
# coding: utf-8

# In[482]:


import matplotlib.pyplot as plt
import numpy as np
import sys
from sympy.solvers import solve
import seaborn as sns
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import signal
from scipy.io import wavfile
import sounddevice as sd


# In[483]:


N1 = 3 + 1
N2 = 6 + 1
N3 = 12 + 1
N4 = 22 +  1

ci1 = np.logspace(np.log10(100), np.log10(8000), N1, endpoint=True, base=10)
ci2 = np.logspace(np.log10(100), np.log10(8000), N2, endpoint=True, base=10)
ci3 = np.logspace(np.log10(100), np.log10(8000), N3, endpoint=True, base=10)
ci4 = np.logspace(np.log10(100), np.log10(8000), N4, endpoint=True, base=10)



# In[484]:


plt.plot(np.log10(ci4), 'o')
plt.xlabel('Electrode Number', fontsize=7)
plt.ylabel('End Frequency (log scale)', fontsize=7)
plt.savefig('elec.pdf')
plt.show()


# In[485]:


def getButterworthFilter(ci, f_nyquist,filename):
    a = []
    b = []
    w = []
    h = []

    ci_norm = ci/f_nyquist
    for i in range(len(ci_norm)-1):
        a.append([])
        b.append([])
        w.append([])
        h.append([])
        lab = "Filter " + str(i+1)
    #     print(ci4_norm[i],' ',ci4_norm[i+1])
        b[i], a[i] = signal.butter(2, [ci_norm[i],ci_norm[i+1]], 'bandpass', analog=True)
    #     b[i], a[i] = signal.butter(2, [ci4_norm[i],ci4_norm[i+1]], 'band')
        w[i], h[i] = signal.freqs(b[i], a[i])
        plt.plot(w[i]*f_nyquist, 20 * np.log10(abs(h[i])), label=lab)

    plt.xscale('log')
#     plt.ylim(-4,0)
#     plt.xlim(7500,9500)
    plt.title('Butterworth filter frequency response')
    plt.xlabel('Frequency [radians / second]')
    plt.ylabel('Amplitude [dB]')
    plt.margins(0, 0.1)
    plt.legend()
    plt.grid(which='both', axis='both')
    # plt.axvline(100, color='green') # cutoff frequency
    plt.savefig(filename)
    plt.show()
    return b,a


# In[487]:


b12, a12 = getButterworthFilter(ci3,8000,'12_butterworth.pdf')
b22, a22 = getButterworthFilter(ci4,8000,'22_butterworth.pdf')
b3, a3 = getButterworthFilter(ci1,8000,'3_butterworth.pdf')


# In[488]:


def PlotChannels(channels, n, fs, filename):
    plt.rc('font', size=3)
    plt.rc('figure', titlesize=5)
    plt.rc('xtick', labelsize=5)
    plt.rc('ytick', labelsize=5)
    t = np.linspace(0, n/fs, n)
    for i in range(1,len(channels)+1):
        lab = "Signal Channel " + str(i)
        ax = plt.subplot(6,2,i)
        ax.plot(t,channels[i-1]*10**5, label=lab)
        ax.set_title(lab, color='red')
        ax.set_xlabel('Seconds')
        ax.set_ylabel('Amplitude ($10^{5}$)')
        ax.set_xticks(np.arange(0,n/fs),(n/fs)/10)
        ax.grid(True)
    plt.subplots_adjust(wspace=0.5, hspace=2.5)
    plt.savefig(filename, dpi=300)
    plt.show()


# In[489]:


def Sound(file,b,a,N):
    plt.rcParams.update(plt.rcParamsDefault)
    fs, data = wavfile.read(file+'.wav')
    n = len(data)
    t = np.linspace(0, n/fs, n)
    plt.plot(t,(data))
    plt.xlabel('Seconds (s)')
    plt.ylabel('Amplitude')
#     plt.xlim(1,1.02)
#     plt.xticks(np.arange(0,n/fs),(n/fs)/10)
    plt.grid(True)
    plt.savefig(file+"_signal.pdf", dpi=300)
    plt.show()
    
    Pxx, freqs, bins, im = plt.specgram(data, NFFT=441, Fs=fs, noverlap=220)
    plt.xlabel('t (s)')
    plt.ylabel('Frequency (Hz)')
    cbar = plt.colorbar(im)
    cbar.set_label('Amplitude')
    
    plt.savefig(file+'_original.pdf')
    plt.show()
    channels = []
    joint_signal = 0
    
    for i in range(0,N-1):
        channels.append([])
        channels[i] = signal.filtfilt(b[i], a[i], data)
        joint_signal = joint_signal + channels[i]
        
    PlotChannels(channels, n, fs, file+'_channels.pdf')
    plt.rcParams.update(plt.rcParamsDefault)
    plt.magnitude_spectrum(jd, Fs=fs, scale='dB', color='C1')
    plt.savefig(file+'_spectrum.pdf')
    plt.show()
#     plt.figure(figsize=(10, 10), dpi=100)
    Pxx, freqs, bins, im = plt.specgram(joint_signal, NFFT=441, Fs=fs, noverlap=220)
    plt.xlabel('t (s)')
    plt.ylabel('Frequency (Hz)')
    cbar = plt.colorbar(im)
    cbar.set_label('Amplitude')
    plt.savefig(file+'_joint.pdf')
    plt.show()
    
    return channels, joint_signal, data
    


# In[490]:


jc, js, jd = Sound('Joycelyne', b12, a12, N3)


# In[352]:


sd.play(js)
min(js)


# In[491]:


Sound('Krishna', b12, a12, N3)


# In[492]:


Sound('Mahamantra', b12, a12, N3)


# In[493]:


Sound('Hello', b12, a12, N3)


# In[494]:


Sound('Helloh', b12, a12, N3)

