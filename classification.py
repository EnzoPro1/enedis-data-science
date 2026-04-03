import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report

# 1. Chargement du nouveau fichier avec les labels de K-Means
df = pd.read_csv('features_clients_avec_labels.csv', sep=';')

# 2. Séparation de X et y
X = df.drop(['ID', 'Label'], axis=1)
y = df['Label']

# 3. Séparation Entraînement (80%) / Test (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Standardisation
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Entraînement du Réseau de Neurones
nn = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=500, random_state=42)
nn.fit(X_train_scaled, y_train)

# 6. Prédictions
y_pred = nn.predict(X_test_scaled)

# 7. Affichage des résultats
print("--- Rapport de Classification ---")
print(classification_report(y_test, y_pred))

# 8. Création de la belle Matrice de Confusion
cm = confusion_matrix(y_test, y_pred, labels=['RS', 'RP'])

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['RS', 'RP'], 
            yticklabels=['RS', 'RP'])
plt.title('Matrice de Confusion - Réseau de Neurones')
plt.xlabel('Prédictions du modèle')
plt.ylabel('Réalité')
plt.show()
