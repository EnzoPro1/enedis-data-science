# Documentation : Préparation des données et Feature Engineering

Cette section détaille le pipeline de traitement des données de consommation électrique Enedis (RES2-6-9kVA) afin de préparer le terrain pour les modèles de Machine Learning (Clustering, Classification et Prévision).

## 1. Objectif
Transformer une série temporelle brute (pas de 30 minutes) en variables explicatives pertinentes (features) permettant de distinguer les Résidences Principales (RP) des Résidences Secondaires (RS).

## 2. Nettoyage et Conversions
* **Formatage Temporel :** Conversion de la colonne `horodate` en objet `datetime` avec gestion du fuseau horaire (UTC).
* **Conversion d'Énergie :** Les données initiales étant en puissance (kW) au pas de 30 minutes, l'énergie consommée a été calculée en multipliant par 0.5 pour obtenir des Kilowattheures (kWh).
* **Nettoyage :** Traitement des valeurs manquantes et des valeurs infinies générées lors des calculs de ratios.

## 3. Création de Variables (Feature Engineering)
Pour permettre à l'algorithme de clustering (K-Means) de comparer efficacement les clients, les données ont été agrégées pour obtenir **une ligne par client**.

Les variables suivantes ont été créées :
* **Statistiques descriptives :** Consommation totale, moyenne, médiane, écart-type, minimum et maximum.
* **Marqueurs temporels :** Création de booléens pour la Nuit (22h-6h), le Week-end (Sam-Dim), et la saison de chauffe / Hiver (Novembre à Mars).
* **Ratios de consommation :** 
  * Ratio Nuit / Jour
  * Ratio Week-end / Semaine
  * Ratio Hiver / Reste de l'année

*Justification : Les ratios permettent de normaliser le comportement des clients indépendamment de la taille de leur logement, ce qui est crucial pour repérer les RS (souvent caractérisées par des pics le week-end).*

## 4. Analyses Avancées
* **Transformée de Fourier (FFT) :** Application de la FFT sur la série de chaque client pour extraire l'amplitude maximale des cycles. Cela aide à capter mathématiquement les rythmes d'occupation (ex: retours réguliers le week-end).
* **Analyse en Composantes Principales (ACP) :** Réduction de la dimensionnalité après standardisation (`StandardScaler`) pour visualiser les comportements sur un plan 2D et identifier visuellement les clusters avant l'application du K-Means.

## 5. Fichiers Générés
Ce pipeline produit trois datasets distincts pour alimenter la suite du projet :
1. `features_clients_pour_clustering.csv` : Matrice client pour le K-Means et la classification.
2. `serie_temporelle_journaliere_pour_forecasting.csv` : Données agrégées au pas journalier pour les modèles de prédiction.
3. `serie_temporelle_brute_pour_generation.csv` : Série nettoyée au pas de 30 min pour le générateur de courbes.