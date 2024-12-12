import os
import sys
import cv2
from generate_dataset import my_dataset_function
from detectron2.utils.visualizer import ColorMode

"""
Su alcune versioni di macOS e Linux, il metodo di avvio del multiprocessing può
causare problemi con DataLoader. Puoi tentare di impostare il flag start_method
su “fork” (questo è particolarmente utile su macOS):
"""
import torch.multiprocessing as mp
mp.set_start_method('fork', force=True)


# Aggiungi il percorso della cartella detectron2
sys.path.append(os.path.abspath("/detectron2_repo"))

from detectron2_repo.detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2_repo.detectron2.config import get_cfg
from detectron2_repo.detectron2.utils.visualizer import Visualizer
from detectron2.data import DatasetCatalog, MetadataCatalog

script_dir = os.path.dirname(__file__)

# 1. Registrazione del dataset in formato COCO

DatasetCatalog.register("brain_mri_train", lambda: my_dataset_function())
DatasetCatalog.register("brain_mri_val", lambda: my_dataset_function())

# Imposta le classi (se non l'hai già fatto)
MetadataCatalog.get("brain_mri_train").set(thing_classes=["brain"])


# 2. Configurazione del modello
cfg = get_cfg()
cfg.DATASETS.TRAIN = ("brain_mri_train",)
cfg.DATASETS.TEST = ("brain_mri_val",)
cfg.MODEL.WEIGHTS = "detectron2://ImageNetPretrained/MSRA/R-50.pkl"  # Modello pre-addestrato
cfg.DATALOADER.NUM_WORKERS = 0
cfg.SOLVER.IMS_PER_BATCH = 2 #immagini per batch
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 80 # Numero di ROI per immagine
cfg.SOLVER.BASE_LR = 0.00025 #learning rate
cfg.SOLVER.MAX_ITER = 500  # iterazioni massime
cfg.SOLVER.WARMUP_ITERS = 50  # aumenta gradualmente il LR nelle prime x iterazioni fino a valore
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # Cambia in base al numero delle classi
cfg.MODEL.DEVICE = "cpu"  # Imposta l'uso della CPU
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3 #Soglia di confidenza

# 3. Addestramento
# Verifica se il modello addestrato esiste già
model_path = os.path.join(script_dir, "output", "model_final.pth")

if os.path.exists(model_path):
    print("Modello già addestrato trovato, caricando il modello...")
    cfg.MODEL.WEIGHTS = model_path
    # Carica il modello addestrato
    predictor = DefaultPredictor(cfg)
else:
    # 3. Addestramento
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

    # Imposta il modello addestrato
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # Salva il modello addestrato


# 4. Inferenza
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # Modello addestrato
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Soglia per l'inferenza
predictor = DefaultPredictor(cfg)

# Carica un'immagine MRI per l'inferenza
image_path = os.path.join(script_dir, "test", "test3.jpg")
image = cv2.imread(image_path)
outputs = predictor(image)
print(outputs)
# Visualizza i risultati con bordi rossi per le istanze rilevate
v = Visualizer(image[:,:,::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
v = v.draw_instance_predictions(outputs['instances'].to('cpu'))
img = v.get_image()[:, :, ::-1]
cv2.imwrite('output.jpg', img)
