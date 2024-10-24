import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft
import scipy.io
from scipy import signal

# Encode image

fs = 4000       

f, t, Zxx = stft(current , fs=fs, nperseg=254, noverlap=60)

x_real =  (np.real(Zxx))

x_imag = np.imag(Zxx)

array1 = x_real.astype(np.float32)

array2 = x_imag.astype(np.float32)

array2 = array2  / (math.ceil(np.max(abs(array1))))

array1 = array1  / (math.ceil(np.max(abs(array1))))

rgb_image = np.dstack((array1, array2))

       
# Decode image        

istft_rdphase = array1 + 1j * array2

_, xrec = signal.istft(istft_rdphase, fs, nperseg=254, noverlap=60)

plt.plot(xrec)

 


 

def plot_ts_data(data_img) :
    fig, ax = plt.subplots(1,2)
    fs = 4000
    istft_rdphase = data_img[:,:,0] + 1j * data_img[:,:,1]
    _, xrec = signal.istft(istft_rdphase, fs, nperseg=254, noverlap=60)
    ax[1].plot(xrec)
    ax[0].imshow(abs(0.5*data_img[:,:,0])+abs(0.5*data_img[:,:,1]), cmap='viridis', interpolation='nearest')
    ax[0].axis('off')
    ax[1].axis('off')