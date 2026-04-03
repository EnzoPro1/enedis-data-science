# Étape 2 : Clustering et Identification des Résidences (K-Means)

**Auteur : Alexandre MUTH
**Script associé :** `02_Clustering_KMeans.py`  
**Rôle dans le projet :** Générer la vérité terrain (labels RP/RS) pour l'étape de classification.

---

## Objectif du Script
L'objectif de cette étape est d'utiliser un algorithme d'apprentissage non supervisé (**K-Means**) pour segmenter les clients en deux groupes distincts en fonction de leurs habitudes de consommation électrique. 
Une fois les clusters créés, une règle métier est appliquée pour identifier quel cluster correspond aux **Résidences Principales (RP)** et lequel correspond aux **Résidences Secondaires (RS)**.

---

## Fichiers et Dépendances

### Entrée (Input)
* `features_clients_pour_clustering.csv` : Fichier généré par le Membre 1 (Data & Features). Contient 1 ligne par client avec ses statistiques de consommation (moyennes, ratios, composantes de Fourier).

### Sortie (Output / Livrable)
* `features_clients_avec_labels.csv` : Le dataset final enrichi avec la nouvelle colonne `Label` contenant les valeurs "RP" ou "RS". Ce fichier est le point de départ pour le Membre 3 (Classification).

### Librairies utilisées
* `pandas`, `numpy` : Manipulation des données.
* `scikit-learn` : Standardisation (`StandardScaler`), Clustering (`KMeans`) et évaluation (`silhouette_score`).
* `matplotlib`, `seaborn` : Visualisation (optionnel dans le livrable final mais utile pour l'analyse).

---

## Explication de la Méthodologie (Étape par Étape)

### 1. Nettoyage et Préparation des données
Avant d'appliquer l'algorithme, il est crucial de traiter les éventuelles anomalies mathématiques (divisions par zéro générant des `np.inf`) souvent créées lors du calcul des ratios de consommation. Celles-ci sont remplacées par des `0.0`. L'identifiant client (`ID`) est mis de côté car il n'a pas de valeur prédictive.

### 2. Standardisation (StandardScaler)
Le K-Means repose sur le calcul de distances (euclidiennes) entre les points. Les variables ayant des échelles très différentes (ex: *conso_totale* en millions vs *ratio_nuit_jour* autour de 1), il est impératif de centrer et réduire les données pour qu'aucune variable ne domine artificiellement la formation des clusters.

### 3. Évaluation Expérimentale (Score de Silhouette)
Bien que l'objectif métier impose $K=2$ (pour RP et RS), une boucle d'exploration calcule le **score de silhouette** pour $K \in [2, 6]$. Cela permet de justifier la qualité du clustering et de s'assurer de la cohérence mathématique de la séparation.

### 4. Clustering K-Means ($K=2$)
L'algorithme est instancié avec `n_clusters=2`. Pour garantir une robustesse face à l'initialisation aléatoire des centroïdes, le paramètre `n_init=50` est utilisé (l'algorithme est lancé 50 fois et conserve la meilleure inertie).

### 5. Labellisation (Règle Métier)
L'algorithme K-Means retourne des numéros de clusters arbitraires (0 et 1). Pour attribuer les labels finaux :
* **Hypothèse retenue :** Une Résidence Secondaire (RS) présente une concentration de sa consommation sur les week-ends beaucoup plus forte qu'une Résidence Principale (RP).
* **Application :** Le cluster présentant la **moyenne la plus élevée** pour la variable `ratio_we_semaine` est automatiquement étiqueté **"RS"**. L'autre devient **"RP"**.

---

## Comment exécuter le code ?
1. S'assurer que le fichier `features_clients_pour_clustering.csv` est dans le même répertoire que le script.
2. Lancer la commande suivante dans le terminal :
   ```bash
   python 02_Clustering_KMeans.py