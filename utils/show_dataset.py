import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import sys 
import os

# Aggiungi il percorso della cartella principale al sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from Config import Config

# Percorso di esempio
image_path = Config.IMAGES_DIR + "/BRATS_457.nii.gz"
label_path = Config.MASKS_DIR + "/BRATS_457.nii.gz"

# Caricamento delle immagini e delle etichette
image = nib.load(image_path).get_fdata()
label = nib.load(label_path).get_fdata()

# Visualizzazione di una fetta 2D
slice_idx = image.shape[2] // 2  # Scegli la fetta centrale
# Normalizzazione dell'immagine in un range da 0 a 255
image_normalized = (image[:, :, slice_idx] - np.min(image[:, :, slice_idx])) / (np.max(image[:, :, slice_idx]) - np.min(image[:, :, slice_idx])) * 255
image_normalized = image_normalized.astype(np.uint8)

# Esamina i valori unici della maschera nella fetta
unique_values = np.unique(label[:, :, slice_idx])
print(f"Valori unici nella maschera: {unique_values}")

# Associa ciascun valore a una categoria
category_map = {
    0: "background",
    1: "edema",
    2: "non-enhancing tumor",
    3: "enhancing tumor"
}

# Determina la categoria associata ai valori unici trovati nella fetta
categories_in_slice = [category_map[val] for val in unique_values if val in category_map]
print(f"Categorie presenti nella fetta: {categories_in_slice}")

# Visualizzazione
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Immagine RMN (FLAIR)")
plt.imshow(image_normalized, cmap="gray")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Etichetta")
plt.imshow(label[:, :, slice_idx], cmap="jet")
plt.axis("off")

plt.show()