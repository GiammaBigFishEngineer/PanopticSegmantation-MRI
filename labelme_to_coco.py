import os
import json
from labelme import utils
import numpy as np

def labelme_to_coco(labelme_folder, output_file):
    images = []
    annotations = []
    categories = [{"id": 0, "name": "Brain", "supercategory": "Medical"}]
    ann_id = 0
    
    for img_id, filename in enumerate(os.listdir(labelme_folder)):
        if filename.endswith('.json'):
            labelme_path = os.path.join(labelme_folder, filename)
            with open(labelme_path) as f:
                label_data = json.load(f)
            
            # Image info
            image_info = {
                "id": img_id,
                "file_name": label_data["imagePath"],
                "width": label_data["imageWidth"],
                "height": label_data["imageHeight"]
            }
            images.append(image_info)
            
            # Parse annotations
            for shape in label_data["shapes"]:
                points = np.array(shape["points"]).flatten().tolist()
                x_coords = [p[0] for p in shape["points"]]
                y_coords = [p[1] for p in shape["points"]]
                bbox = [min(x_coords), min(y_coords), max(x_coords) - min(x_coords), max(y_coords) - min(y_coords)]
                area = bbox[2] * bbox[3]
                
                ann = {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": 0,
                    "segmentation": [points],
                    "bbox": bbox,
                    "area": area,
                    "iscrowd": 0,
                    "bbox_mode": 0
                }
                annotations.append(ann)
                ann_id += 1
    
    coco_output = {
        "images": images,
        "annotations": annotations,
        "categories": categories
    }
    
    with open(output_file, 'w') as f:
        json.dump(coco_output, f)

# Usalo per convertire i file

script_dir = os.path.dirname(__file__)
# Percorso relativo alla cartella JSON
labelme_folder = os.path.join(script_dir, "labelme", "json_files")
output_file = os.path.join(script_dir, "output", "coco_annotations.json")
labelme_to_coco(labelme_folder, output_file)