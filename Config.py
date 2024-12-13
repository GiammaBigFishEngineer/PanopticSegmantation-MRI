import os


class Config:
    BASE_DIR = os.path.dirname(__file__)
    IMAGES_DIR = os.path.join(BASE_DIR, "Task01_BrainTumour/imagesTr")
    MASKS_DIR = os.path.join(BASE_DIR, "Task01_BrainTumour/labelsTr")
    OUTPUT_DIR = os.path.join(BASE_DIR, "output")
    TRAINING_IMAGES_DIR = os.path.join(BASE_DIR, "custom_dataset")