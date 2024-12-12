import json
import os
import random

def split_coco_annotations(input_file, output_train_file, output_val_file, val_ratio=0.2):
    with open(input_file, 'r') as f:
        coco_data = json.load(f)

    # Shuffle delle immagini
    images = coco_data['images']
    random.shuffle(images)

    # Calcolo della divisione
    val_size = int(len(images) * val_ratio)
    train_images = images[val_size:]
    val_images = images[:val_size]

    # Creazione mapping immagine_id -> immagine
    image_id_to_image = {img['id']: img for img in coco_data['images']}

    # Filtra annotazioni per training e validation
    train_image_ids = {img['id'] for img in train_images}
    val_image_ids = {img['id'] for img in val_images}

    train_annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] in train_image_ids]
    val_annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] in val_image_ids]

    # Creazione dei file finali
    train_coco = {
        "images": train_images,
        "annotations": train_annotations,
        "categories": coco_data['categories']
    }
    
    val_coco = {
        "images": val_images,
        "annotations": val_annotations,
        "categories": coco_data['categories']
    }

    # Salvataggio dei file
    with open(output_train_file, 'w') as f:
        json.dump(train_coco, f)
    with open(output_val_file, 'w') as f:
        json.dump(val_coco, f)

    print(f"File train salvato in {output_train_file}")
    print(f"File val salvato in {output_val_file}")

# Parametri
script_dir = os.path.dirname(__file__)
input_file = os.path.join(script_dir, "output", "coco_annotations.json")  # File COCO completo
output_train_file = os.path.join(script_dir, "custom_dataset", "train_annotations.json")
output_val_file = os.path.join(script_dir, "custom_dataset", "val_annotations.json")

split_coco_annotations(input_file, output_train_file, output_val_file)