import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
# ==========================================
# 1. Chargement des données
# ==========================================
df = pd.read_csv('features_clients_pour_clustering.csv', sep=';')

# On sépare les features de l'ID
feature_cols = [col for col in df.columns if col != 'ID']
X = df[feature_cols].copy()

# ==========================================
# 2. Préparation et Standardisation 
# ==========================================
# Copié-collé du code du prof pour gérer les divisions par zéro ou valeurs nulles potentielles
X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

# Standardisation classique 
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ==========================================
# 3. Évaluation K-Means : Silhouette (Copié-collé du prof)
# ==========================================
scores = {}
for k in range(2, 7):
    # On utilise n_init=20 dans sa boucle
    km = KMeans(n_clusters=k, n_init=20, random_state=42)
    labels = km.fit_predict(X_scaled)
    scores[k] = silhouette_score(X_scaled, labels)

print("Scores de silhouette :", scores)

# ==========================================
# 4. Application du Clustering final (K=2)
# ==========================================
# ON utilise n_init=50 dans son run final
kmeans = KMeans(n_clusters=2, n_init=50, random_state=42)
df["cluster"] = kmeans.fit_predict(X_scaled)

print("\nAperçu des clusters :")
print(df[["ID", "cluster"]].head())
# ==========================================
# GENERATION DU GRAPHIQUE POUR LA SLIDE 3 (Anonyme)
# ==========================================
print("\nGénération du graphique ACP...")

# 1. On réduit les 17 dimensions à seulement 2 dimensions (PCA1 et PCA2)
pca = PCA(n_components=2)
components = pca.fit_transform(X_scaled)

# 2. On ajoute temporairement ces coordonnées au DataFrame
df['PCA1'] = components[:, 0]
df['PCA2'] = components[:, 1]

# 3. Création de la figure
plt.figure(figsize=(10, 6))
sns.scatterplot(
    x='PCA1', y='PCA2',
    hue='cluster',               # On colore selon le numéro du cluster (0 ou 1)
    palette='viridis',           # Une belle palette de couleurs "neutre"
    data=df,
    s=100, alpha=0.8
)

# 4. Habillage du graphique
plt.title('Projection ACP : Identification de 2 profils distincts', fontsize=14)
plt.xlabel(f'Composante Principale 1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)')
plt.ylabel(f'Composante Principale 2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)')
plt.legend(title='Cluster (K-Means)')
plt.grid(True, linestyle='--', alpha=0.5)

# 5. SAUVEGARDE L'IMAGE SUR TON PC (Prêt à être collé dans ton PowerPoint !)
plt.savefig("slide3_acp_anonyme.png", dpi=300, bbox_inches='tight')
plt.show()
# ==========================================
# 5. Labellisation (RP vs RS)
# ==========================================
# Hypothèse : Le cluster avec le plus fort ratio WE/Semaine est "RS"
moyennes_par_cluster = df.groupby('cluster')['ratio_we_semaine'].mean()
cluster_rs = moyennes_par_cluster.idxmax()

df['Label'] = df['cluster'].apply(lambda x: 'RS' if x == cluster_rs else 'RP')

# ==========================================
# 6. Sauvegarde 
# ==========================================
output_path = "features_clients_avec_labels.csv"

# On retire la colonne technique 'cluster' pour que ce soit propre
output_df = df.drop(columns=['cluster']) 
output_df.to_csv(output_path, index=False, sep=";")

# Affichage exact utilisé dans la page 31 
print(f"\nLe fichier de labels a été sauvegardé avec succès ici : {output_path}")
print("\nAperçu des 5 premières lignes :")
print(output_df[["ID", "Label"]].head())

# ==========================================
# GENERATION DU GRAPHIQUE 1 (Option Zoom)
# ==========================================
print("\nGénération du diagramme en barres (Zoomé)...")

plt.figure(figsize=(8, 5))
sns.barplot(
    x='cluster', 
    y='ratio_we_semaine', 
    data=df, 
    errorbar=None,    
    hue='cluster',
    legend=False,
    palette='viridis' 
)

# On calcule le minimum et le maximum de nos ratios pour ajuster la vue
min_val = df['ratio_we_semaine'].min() * 0.95
max_val = df['ratio_we_semaine'].max() * 1.05
plt.ylim(min_val, max_val) 
# -----------------------------------------------

plt.title('Preuve métier : Ratio Week-end / Semaine par Cluster', fontsize=14)
plt.xlabel('Numéro du Cluster (K-Means)', fontsize=12)
plt.ylabel('Ratio Moyen', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.5)

plt.savefig("slide4_barplot_zoom.png", dpi=300, bbox_inches='tight')
plt.show()

# ==========================================
# GENERATION DU GRAPHIQUE 2 (SLIDE 4) : Le Résultat
# ==========================================
print("Génération du graphique ACP avec les Labels finaux...")

plt.figure(figsize=(10, 6))
sns.scatterplot(
    x='PCA1', y='PCA2',
    hue='Label',                 # on colore par "Label"
    palette={'RP': '#1f77b4', 'RS': '#ff7f0e'}, # Code couleur : Bleu pour RP, Orange pour RS
    data=df,
    s=100, alpha=0.8
)

plt.title('Vérité Terrain : Séparation des RP et RS', fontsize=14)
plt.xlabel(f'Composante Principale 1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)')
plt.ylabel(f'Composante Principale 2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)')
plt.legend(title='Type de Résidence')
plt.grid(True, linestyle='--', alpha=0.5)

# Sauvegarde de l'image
plt.savefig("slide4_acp_labels.png", dpi=300, bbox_inches='tight')
plt.show()