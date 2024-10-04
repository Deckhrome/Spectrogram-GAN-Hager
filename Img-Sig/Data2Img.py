# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 09:38:00 2024
Refactored by ChatGPT
"""

import os
import requests
from zipfile import ZipFile, BadZipFile
import shutil
import re
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft
import pickle
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def unzip_all_files(input_folder='downloads', output_folder='extracted'):
    """Unzips multipart zip files in a given folder, combines, and extracts them."""
    os.makedirs(output_folder, exist_ok=True)

    # List and filter the zip parts based on the pattern file.zip.number
    zip_files = sorted([file for file in os.listdir(input_folder) if re.match(r'^.*\.zip\.\d+$', file)])
    
    if not zip_files:
        logging.error(f"No multipart zip files found in {input_folder}")
        return
    
    combined_file_path = os.path.join(output_folder, 'combined.zip')
    
    # Combine all zip parts into a single file
    with open(combined_file_path, 'wb') as combined_file:
        for zip_file in zip_files:
            zip_file_path = os.path.join(input_folder, zip_file)
            with open(zip_file_path, 'rb') as part_file:
                shutil.copyfileobj(part_file, combined_file)
    
    # Unzip the combined file
    try:
        with ZipFile(combined_file_path, 'r') as zip_ref:
            zip_ref.extractall(output_folder)
        logging.info(f"Successfully extracted combined file to {output_folder}")
    except BadZipFile:
        logging.error("Error: Combined file is not a valid zip file")
    finally:
        # Optional: Remove the combined file
        os.remove(combined_file_path)


def proxy_activate():
    """Set up proxy configuration, if needed."""
    os.environ['HTTP_PROXY'] = ''
    os.environ['HTTPS_PROXY'] = ''
    logging.info("Proxy activated.")


def download_files(zip_urls, download_path='downloads'):
    """Downloads files from the provided URLs to a specified folder."""
    os.makedirs(download_path, exist_ok=True)
    for url in zip_urls:
        file_name = os.path.join(download_path, os.path.basename(url))
        
        if os.path.exists(file_name):
            logging.info(f"{file_name} already exists, skipping download.")
            continue
        
        try:
            logging.info(f"Downloading {url} to {file_name}")
            response = requests.get(url)
            response.raise_for_status()
            
            with open(file_name, 'wb') as file:
                file.write(response.content)
            logging.info(f"Downloaded {file_name}")
        except requests.RequestException as e:
            logging.error(f"Failed to download {url}: {e}")


def generate_rgb_image(Zxx):
    """Generates an RGB image from STFT data by normalizing real, imaginary, and phase components without repeating."""
    x_real = np.real(Zxx)
    x_imag = np.imag(Zxx)
    x_angle = np.angle(Zxx) / np.pi

    # Normalize each array
    def normalize(arr):
        return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))
    
    array1 = normalize(x_real)
    array2 = normalize(x_imag)
    array3 = normalize(x_angle)

    # Stack normalized components into an RGB image
    return np.dstack((array1, array2, array3))

def ImgFrMat(mat_file, output_folder):
    """Processes .mat files and converts time-series data into compact RGB images."""
    mat_data = loadmat(mat_file)
    fs = 4000
    name_val = ['current', 'voltage']

    for name in name_val:
        try:
            signal = np.squeeze(mat_data.get(name))
            if signal is None:
                logging.warning(f"{name} not found in {mat_file}")
                continue
            
            # Perform STFT with fewer segments to reduce image size
            f, t, Zxx = stft(signal, fs=fs, nperseg=128)  # Adjust nperseg for smaller size
            rgb_image = generate_rgb_image(Zxx)
            
            # Plot and save image without enlarging the figure
            plt.imshow(rgb_image)
            plt.axis('off')
            
            # Save the image with reduced dimensions (optional: adjust DPI for further compression)
            mat_file_base = os.path.splitext(os.path.basename(mat_file))[0]
            image_file_path = os.path.join(output_folder, f"{mat_file_base}_{name}.png")
            plt.savefig(image_file_path, bbox_inches='tight', pad_inches=0, transparent=True, dpi=72)  # Adjust DPI if needed
            plt.close()
            
            logging.info(f"Processed and saved image for {name} as {image_file_path}")
        except Exception as e:
            logging.error(f"Failed to process {name} in {mat_file}: {e}")

from tqdm import tqdm

if __name__ == "__main__":
    list_link = [
        "https://zhsaafddbodweu.blob.core.windows.net/sdia/SDIA.zip.001",
        "https://zhsaafddbodweu.blob.core.windows.net/sdia/SDIA.zip.002",
        "https://zhsaafddbodweu.blob.core.windows.net/sdia/SDIA.zip.003",
        "https://zhsaafddbodweu.blob.core.windows.net/sdia/SDIA.zip.004",
        "https://zhsaafddbodweu.blob.core.windows.net/sdia/SDIA.zip.005",
        "https://zhsaafddbodweu.blob.core.windows.net/sdia/SDIA.zip.006"
    ]
    
    input_folder = '/home/hager/Desktop/Hager/Data/Zipfiles'
    output_folder = '/home/hager/Desktop/Hager/SpectrogramGAN/sampleSpectrograms/Images'

    proxy_activate()  # Activate the proxy if needed
    #download_files(list_link, input_folder)  # Download files
    #unzip_all_files(input_folder, output_folder)  # Unzip files
    
    # Process .mat files
    mat_folder = '/home/hager/Desktop/Hager/Data/matrix'
    mat_files = [file for file in os.listdir(mat_folder) if file.lower().endswith('.mat')]
    
    for mat_file in (mat_files):
        ImgFrMat(os.path.join(mat_folder, mat_file), output_folder)