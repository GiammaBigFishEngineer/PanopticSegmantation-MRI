import os
import json

def my_dataset_function():
    # Percorso al file delle annotazioni COCO
    script_dir = os.path.dirname(__file__)
    dataset_dir = "output"
    json_file = os.path.join(script_dir, dataset_dir, "coco_annotations.json")

    # Carica il file JSON delle annotazioni
    with open(json_file) as f:
        coco_data = json.load(f)

    # Estrai le informazioni rilevanti dal file COCO
    dataset_dicts = []
    for image_info in coco_data['images']:
        record = {}
        record['file_name'] = os.path.join(script_dir, image_info['file_name'])
        record['image_id'] = image_info['id']
        record['height'] = image_info['height']
        record['width'] = image_info['width']
        
        # Aggiungi le annotazioni per ogni immagine
        annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] == image_info['id']]
        record['annotations'] = annotations
        dataset_dicts.append(record)
    
    return dataset_dicts