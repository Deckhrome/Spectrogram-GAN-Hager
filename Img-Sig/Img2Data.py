import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import istft
from PIL import Image

def inverse_rgb_image_to_signal(image_path, fs=4000, nperseg=38*10, target_signal_length=1900):
    """
    Reconstruit un signal temporel à partir d'une image PNG (spectrogramme en RGB).
    
    :param image_path: Chemin de l'image PNG
    :param fs: Fréquence d'échantillonnage utilisée pour la reconstruction
    :param nperseg: Nombre de points par segment utilisé dans le STFT original
    :param target_signal_length: Longueur souhaitée du signal reconstruit
    :return: Signal temporel reconstruit
    """
    # Charger l'image PNG
    img = Image.open(image_path)
    img_array = np.array(img)

    # Extraire les trois canaux RGB
    R_channel = img_array[:, :, 0] / 255.0  # Canal rouge (normalisé entre 0 et 1)
    G_channel = img_array[:, :, 1] / 255.0  # Canal vert
    B_channel = img_array[:, :, 2] / 255.0  # Canal bleu
    
    # Reconstruire les parties réelle, imaginaire, et l'angle de phase
    real_part = R_channel  # Canal rouge représente la partie réelle normalisée
    imag_part = G_channel  # Canal vert représente la partie imaginaire normalisée
    phase_angle = B_channel * np.pi  # Canal bleu représente l'angle de phase (normalisé entre -pi et pi)

    # Reconstruire la matrice complexe à partir des canaux RGB
    Zxx = real_part + 1j * imag_part * np.exp(1j * phase_angle)

    # Appliquer l'ISTFT pour retrouver le signal temporel
    _, signal_reconstructed = istft(Zxx, fs, nperseg=nperseg)

    # Ajuster la longueur du signal reconstruit si nécessaire
    if len(signal_reconstructed) > target_signal_length:
        signal_reconstructed = signal_reconstructed[:target_signal_length]
    elif len(signal_reconstructed) < target_signal_length:
        # Si le signal est plus court que la longueur cible, on le remplit de zéros
        signal_reconstructed = np.pad(signal_reconstructed, (0, target_signal_length - len(signal_reconstructed)), mode='constant')

    return signal_reconstructed

# Exemple d'utilisation
matrix_file_path = '/home/hager/Desktop/Hager/Data/matrix/01-09-2016_SequenceTrace_00007_2_data_119490.mat'
image_file_path = '/home/hager/Desktop/Hager/SpectrogramGAN/sampleSpectrograms/Images/01-09-2016_SequenceTrace_00007_2_data_119490_voltage.png' 
reconstructed_signal = inverse_rgb_image_to_signal(image_file_path)

# open .mat and build the signal to plot
import scipy.io
import numpy as np
import matplotlib.pyplot as plt

# Charger le fichier .mat contenant le signal
mat = scipy.io.loadmat(matrix_file_path)
signal = mat['voltage']
x_value = np.arange(0, len(signal))   

# Tracer le signal original
plt.figure(figsize=(10, 6))
plt.plot(x_value, signal)
plt.title('Signal temporel original')
plt.xlabel('Temps (échantillons)')
plt.ylabel('Amplitude')
plt.show()

# Tracer le signal reconstruit
plt.figure(figsize=(10, 6))
plt.plot(reconstructed_signal)
plt.title('Signal temporel reconstruit à partir de l\'image PNG')
plt.xlabel('Temps (échantillons)')
plt.ylabel('Amplitude')
plt.show()

# Sauvegarder le signal reconstruit dans un fichier texte
np.savetxt("reconstructed_signal_from_image.txt", reconstructed_signal)
