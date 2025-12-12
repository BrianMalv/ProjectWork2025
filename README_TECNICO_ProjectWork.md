# README TECNICO – Project Work ML
Smistamento recensioni hotel & Analisi Sentiment

## 1. Prerequisiti tecnici
- Windows 10/11
- Python 3.8+ (solo per creare l'eseguibile)
- Librerie: pandas, numpy, scikit-learn, matplotlib, pyinstaller

## 2. Installazione dipendenze
```
pip install pandas numpy scikit-learn matplotlib pyinstaller
```

## 3. Esecuzione come script
```
python project_work_hotel_tkinter.py
```

## 4. Creazione eseguibile
```
pyinstaller --onefile --windowed  project_work_hotel_tkinter.py
```
L'eseguibile si trova in:
```
dist/project_work_hotel_tkinter.exe
```

## 5. Funzionamento
- Generazione dataset sintetico (500 recensioni)
- Addestramento modelli ML
- Dashboard Tkinter
- Predizione singola o tramite CSV

## 6. File generati
- reviews_synthetic.csv
- results/confusion_matrix_department.png
- results/confusion_matrix_sentiment.png
- *_predictions_timestamp.csv

## 7. Requisiti CSV input
Il file deve contenere almeno:
```
title, body
```

## 8. Troubleshooting
- Se SmartScreen blocca l'eseguibile: cliccare "Ulteriori informazioni" → "Esegui comunque".
- Se il CSV non viene accettato, verificare le colonne richieste.

## 9. Contenuto didattico
- TF-IDF (ngrams 1–2)
- Logistic Regression (reparto e sentiment)
- Valutazione: accuracy, F1 macro, confusion matrix
