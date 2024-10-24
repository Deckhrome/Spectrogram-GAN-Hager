import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import istft
from PIL import Image
from scipy import signal
import math
import scipy.io


def plot_ts_data(data_img):
    fig, ax = plt.subplots(1,2)
    fs = 4000
    print(data_img.shape)
    istft_rdphase = data_img[:,:,0] + 1j * data_img[:,:,1]
    _, xrec = signal.istft(istft_rdphase, fs, nperseg=254, noverlap=60)
    ax[1].plot(xrec)
    ax[0].imshow(abs(0.5*data_img[:,:,0])+abs(0.5*data_img[:,:,1]), cmap='viridis', interpolation='nearest')
    ax[0].axis('off')
    ax[1].axis('off')
    plt.show()

# Exemple d'utilisation
matrix_file_path = '/home/hager/Desktop/Hager/Data/matrix/2017-04-23_SequenceTrace_00100_4_data_170248_current.npy'
image_file_path = '/home/hager/Desktop/Hager/Data/Images/2017-04-23_SequenceTrace_00100_4_data_170248_current.npy' 

# Open the image file
data_img = np.load(image_file_path)
plot_ts_data(data_img)

# Then plot the original .mat file 
mat_data = scipy.io.loadmat(matrix_file_path)
plt.plot(mat_data['current'])
plt.show()
