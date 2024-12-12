# SEGMENTAZIONE PANOTTICA IMMAGINI MEDICHE 

## Attenzione:
Per la segmentazione medica, potrebbe essere necessario:
	•	Preprocessare le immagini MRI: Converti il formato DICOM in PNG/JPEG o usa pacchetti come pydicom per leggere direttamente i file DICOM.
	•	Aumentare i dati: Usa tecniche di data augmentation specifiche per immagini mediche.

### COSTRUZIONE DEL DATASET
Dopo la selezione delle immagini adeguate usare labelme per generare le etichette dei dati

### DOPO LA GENERAZIONE DI LABELME E LA CONVERSIONE 
1. Comprensione di train_annotations.json e val_annotations.json
	•	train_annotations.json: Contiene le annotazioni per le immagini di training.
	•	val_annotations.json: Contiene le annotazioni per le immagini di validazione.
	•	Entrambi i file devono seguire il formato COCO e contenere:
	•	images: Le informazioni sulle immagini (nome file, dimensioni, ID).
	•	annotations: Le annotazioni associate (maschere, bounding box, ecc.).
	•	categories: La definizione delle categorie.

2. Divisione del Dataset

Per creare i file train_annotations.json e val_annotations.json:
	1.	Dividi il dataset:
	•	Separare le immagini in due insiemi:
	•	Training set: Circa il 80-90% delle immagini.
	•	Validation set: Circa il 10-20% delle immagini.
	2.	Mantenere la coerenza degli ID:
	•	Aggiorna gli ID delle immagini e delle annotazioni per ciascun file.

3. Script Python per Dividere il Dataset
Usare uno script per dividere automaticamente il dataset COCO in due file


### Istruzioni d'uso complessive 
1. Inserisci immagini dentro labelme
2. Salva tutti i file json per ogni immagine in labelme/json_files
3. esegui labelme_to_coco.py
4. esegui create_annotation.py
5. esegui script.py

### Dati Mostrati durante l'addestramento
	•	eta: ETA (Estimated Time of Arrival): tempo stimato rimanente per completare l’addestramento.
	•	iter: Iterazione corrente: l’addestramento è arrivato alla i iterazione.
	•	total_loss: Perdita totale: misura complessiva dell’errore del modello alla 19ª iterazione.Una perdita più bassa indica che il modello si sta avvicinando a una migliore approssimazione dei dati.
