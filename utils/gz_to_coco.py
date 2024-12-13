import os
import nibabel as nib
import numpy as np
import json
import cv2  # Per salvare immagini 2D
from skimage.measure import regionprops
from skimage.morphology import label
import sys

# Aggiungi il percorso della cartella principale al sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from Config import Config

# Categorie
categories = {
    0: "background",
    1: "edema",
    2: "non-enhancing tumor",
    3: "enhancing tumor"
}

# Caricamento delle immagini .nii.gz
def load_nii(nii_path):
    nii_img = nib.load(nii_path)
    return nii_img.get_fdata(), nii_img.affine

# Funzione per creare la segmentazione da una maschera
def get_segmentation(mask):
    # Usa il label delle regioni per estrarre poligoni da ogni area segmentata
    labels = label(mask)
    segmentations = []
    for region in regionprops(labels):
        if region.area > 100:  # Filtra per evitare piccole aree
            contour = region.coords
            segmentations.append((region.label, contour.flatten().tolist()))  # Include anche il label
    return segmentations

# Funzione per salvare una slice come immagine
def save_slice(slice_data, image_id, output_dir):
    slice_image = np.uint8(slice_data * 255)
    image_filename = f"{output_dir}/image_{image_id}.png"
    cv2.imwrite(image_filename, slice_image)
    return image_filename

# Funzione principale per generare il file COCO
def generate_coco_json(nii_file, mask_file, output_dir):
    # Carica il file NIfTI e la maschera
    nii_data, affine = load_nii(nii_file)
    mask_data, _ = load_nii(mask_file)
    
    coco_data = {
        "images": [],
        "annotations": [],
        "categories": [
            {"id": 0, "name": "background", "supercategory": "Medical"},
            {"id": 1, "name": "edema", "supercategory": "Medical"},
            {"id": 2, "name": "non-enhancing tumor", "supercategory": "Medical"},
            {"id": 3, "name": "enhancing tumor", "supercategory": "Medical"}
        ]
    }
    
    image_id = 0
    annotation_id = 0
    for z in range(nii_data.shape[2]):
        # Prendi la slice
        slice_image = nii_data[:, :, z]
        slice_mask = mask_data[:, :, z]
        # Salva l'immagine come PNG
        image_filename = save_slice(slice_image, image_id, Config.TRAINING_IMAGES_DIR)

        # Aggiungi immagine al JSON
        coco_data["images"].append({
            "id": image_id,
            "file_name": image_filename,
            "width": slice_image.shape[1],
            "height": slice_image.shape[0]
        })

        # Esamina i valori unici della maschera per determinare la categoria
        unique_values = np.unique(slice_mask)
        categories_in_slice = [categories[val] for val in unique_values if val in categories]
        
        # Estrai la segmentazione
        segmentations = get_segmentation(slice_mask)
        for label_id, segmentation in segmentations:
            # Associa il category_id in base alla categoria
            category_name = categories.get(label_id, "background")  # Default a "background" se non trovato
            category_id = [key for key, value in categories.items() if value == category_name][0]
            
            coco_data["annotations"].append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": category_id,  # Usa il category_id per assegnare la categoria
                "segmentation": [segmentation],
                "bbox": [
                    min(segmentation[::2]), 
                    min(segmentation[1::2]), 
                    max(segmentation[::2]) - min(segmentation[::2]), 
                    max(segmentation[1::2]) - min(segmentation[1::2])
                ],
                "bbox_mode": 0,
                "area": len(segmentation) / 2,  # Area approssimativa
                "iscrowd": 0
            })
            annotation_id += 1

        image_id += 1

    # Salva il file JSON
    with open(f"{output_dir}/coco_annotations.json", "w") as f:
        json.dump(coco_data, f, indent=4)

# Funzione per eseguire il processo su tutte le immagini e maschere
def process_all_images_and_masks(images_dir, masks_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Elenco delle immagini .nii.gz in imagesTr
    image_files = [f for f in os.listdir(images_dir) if f.endswith('.nii.gz')]
    
    for image_file in image_files:
        # Costruisci il percorso completo dei file immagine e maschera
        nii_file = os.path.join(images_dir, image_file)
        mask_file = os.path.join(masks_dir, image_file)  # Si assume che la maschera abbia lo stesso nome
        
        if os.path.exists(mask_file):
            print(f"Elaborando {image_file}...")
            generate_coco_json(nii_file, mask_file, output_dir)
        else:
            print(f"Maschera non trovata per {image_file}. Saltando...")



process_all_images_and_masks(Config.IMAGES_DIR, Config.MASKS_DIR, Config.OUTPUT_DIR)