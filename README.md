# Projet Data Science : Détection et Analyse de Résidences (RP vs RS)

## Présentation du Projet
Ce projet a été réalisé dans le cadre du cours de Data Science à ESIEE Paris. 
L'objectif principal est d'analyser les courbes de charge électriques (données ouvertes Enedis RES2-6-9kVA) pour identifier, classer et simuler les comportements des clients en distinguant les **Résidences Principales (RP)** des **Résidences Secondaires (RS)**.

## L'Équipe
* **Enzo MASSENGO** : Feature Engineering & Préparation des données
* **Alexandre MUTH** : Clustering (K-Means) & Identification RP/RS
* **Léo WIMART** : Classification & Prévision (Forecasting)
* **Khadija WAHHABI** : Génération de courbes & Dashboard Streamlit

## Structure du Dépôt
Voici l'organisation de nos fichiers et de notre code :

```text
📁 projet_energie/
│
├── 📁 data/                  # Dossier (ignoré par Git) contenant les CSV d'origine
├── 📁 outputs/               # Fichiers générés par l'ingénierie des caractéristiques
│   ├── features_clients_pour_clustering.csv
│   ├── serie_temporelle_journaliere.csv
│   └── serie_temporelle_brute.csv
│
├── data_preparation.py       # Script de nettoyage, FFT et Feature Engineering
├── clustering_model.py       # Algorithme K-Means et ACP
├── classification_forecasting.py # Modèles prédictifs (LogReg, PyTorch, ARIMA...)
├── generation.py             # Modèle de génération conditionnelle (RP/RS)
├── app.py                    # Script principal du Dashboard Streamlit
│
├── requirements.txt          # Liste des bibliothèques nécessaires
└── README.md                 # Documentation du projet
