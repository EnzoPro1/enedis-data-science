import pandas as pd
import numpy as np
from scipy.fft import fft

print("1. Chargement des données")
df = pd.read_csv('courbes-de-charge-fictives-experimentales-profil-res2-plage-6-9-kva.csv', sep=';')

print("2. Nettoyage et conversion (kW -> kWh)...")
df['horodate'] = pd.to_datetime(df['horodate'], utc=True)
df['energie_kWh'] = df['valeur'] * 0.5

print("3. Création des variables temporelles...")
df['heure'] = df['horodate'].dt.hour
df['jour_semaine'] = df['horodate'].dt.dayofweek
df['mois'] = df['horodate'].dt.month

df['est_nuit'] = df['heure'].isin(list(range(22, 24)) + list(range(0, 6)))
df['est_weekend'] = df['jour_semaine'] >= 5
df['est_hiver'] = df['mois'].isin([11, 12, 1, 2, 3])

print("4. Agrégation par client (Statistiques descriptives)")
df_features = df.groupby('ID').agg(
    conso_totale=('energie_kWh', 'sum'),
    conso_moyenne=('energie_kWh', 'mean'),
    conso_mediane=('energie_kWh', 'median'),
    conso_ecart_type=('energie_kWh', 'std'),
    conso_min=('energie_kWh', 'min'),
    conso_max=('energie_kWh', 'max')
).reset_index()

print("5. Création des Ratios (Jour/Nuit, Semaine/WE, Hiver/Été)")
pivot_nuit = df.pivot_table(index='ID', columns='est_nuit', values='energie_kWh', aggfunc='mean')
pivot_nuit.columns = ['conso_moy_jour', 'conso_moy_nuit']

pivot_we = df.pivot_table(index='ID', columns='est_weekend', values='energie_kWh', aggfunc='mean')
pivot_we.columns = ['conso_moy_semaine', 'conso_moy_weekend']

pivot_hiver = df.pivot_table(index='ID', columns='est_hiver', values='energie_kWh', aggfunc='mean')
pivot_hiver.columns = ['conso_moy_hors_hiver', 'conso_moy_hiver']

# Fusion des pivots
df_features = df_features.merge(pivot_nuit, on='ID').merge(pivot_we, on='ID').merge(pivot_hiver, on='ID')

# Calcul des ratios finaux
df_features['ratio_nuit_jour'] = df_features['conso_moy_nuit'] / df_features['conso_moy_jour'].replace(0, np.nan)
df_features['ratio_we_semaine'] = df_features['conso_moy_weekend'] / df_features['conso_moy_semaine'].replace(0, np.nan)
df_features['ratio_hiver_reste'] = df_features['conso_moy_hiver'] / df_features['conso_moy_hors_hiver'].replace(0, np.nan)

print("6. Application de la Transformée de Fourier")
def calculer_fourier(serie_conso):
    serie_centree = serie_conso - serie_conso.mean()
    transformee = np.abs(fft(serie_centree.values))
    n = len(serie_conso)
    return pd.Series({
        'fourier_amplitude_max': np.max(transformee[1:n//2]),
        'fourier_energie_variations': np.sum(transformee[1:n//2]**2)
    })

df_fourier = df.groupby('ID')['energie_kWh'].apply(calculer_fourier).unstack().reset_index()
df_clustering = df_features.merge(df_fourier, on='ID')

# Nettoyage final des infinis et NaN
df_clustering.replace([np.inf, -np.inf], np.nan, inplace=True)
df_clustering.fillna(0, inplace=True)

print("7. Préparation des données journalières / Forecasting")
df['date_jour'] = df['horodate'].dt.date
df_forecasting = df.groupby(['ID', 'date_jour']).agg(
    conso_journaliere_kWh=('energie_kWh', 'sum')
).reset_index()

print("8. Exportation des 3 fichiers CSV")
df_clustering.to_csv('features_clients_pour_clustering.csv', index=False, sep=';')
df_forecasting.to_csv('serie_temporelle_journaliere_pour_forecasting.csv', index=False, sep=';')
df[['ID', 'horodate', 'energie_kWh', 'est_nuit', 'est_weekend', 'est_hiver']].to_csv('serie_temporelle_brute_pour_generation.csv', index=False, sep=';')

print("Terminé")