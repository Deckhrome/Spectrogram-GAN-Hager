import os
import shutil

source_dir = 'sampleSpectrograms' 

img_dir = 'sampleSpectrograms/Images'

current_dir = os.path.join(source_dir, 'current')
voltage_dir = os.path.join(source_dir, 'voltage')

os.makedirs(current_dir, exist_ok=True)
os.makedirs(voltage_dir, exist_ok=True)

# Parcourir les fichiers dans le dossier source
for filename in os.listdir(img_dir):
    if filename.endswith('.npy'):
        # Trier les fichiers contenant "current" dans leur nom
        if 'current' in filename.lower():
            shutil.move(os.path.join(img_dir, filename), os.path.join(current_dir, filename))
        # Trier les fichiers contenant "voltage" dans leur nom
        elif 'voltage' in filename.lower():
            shutil.move(os.path.join(img_dir, filename), os.path.join(voltage_dir, filename))

print("Fichiers triés avec succès.")