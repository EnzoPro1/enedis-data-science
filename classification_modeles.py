import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings('ignore')

# 1. Chargement et préparation des données
df_ts = pd.read_csv('serie_temporelle_journaliere_pour_forecasting.csv', sep=';')
df_ts['date_jour'] = pd.to_datetime(df_ts['date_jour'])
# On fait la moyenne par jour pour lisser un peu
ts = df_ts.groupby('date_jour')['conso_journaliere_kWh'].mean().dropna()

# Les 30 derniers jours pour tester
train = ts.iloc[:-30]
test = ts.iloc[-30:]

# 2. Modèle Régression Linéaire
X_train_reg = np.arange(len(train)).reshape(-1, 1)
X_test_reg = np.arange(len(train), len(train) + len(test)).reshape(-1, 1)
reg = LinearRegression()
reg.fit(X_train_reg, train)
pred_reg = reg.predict(X_test_reg)

# 3. Modèle ARIMA
arima = ARIMA(train, order=(5, 1, 0))
arima_fit = arima.fit()
pred_arima = arima_fit.forecast(steps=len(test))

# 4. Création de la courbe interactive avec Plotly
fig = go.Figure()

# Ajout de l'historique (les 60 jours avant le test pour pas trop charger)
fig.add_trace(go.Scatter(x=train.index[-60:], y=train.values[-60:], mode='lines', name='Historique (Train)', line=dict(color='black', width=2)))

# Ajout de la réalité
fig.add_trace(go.Scatter(x=test.index, y=test.values, mode='lines', name='Réalité (Test)', line=dict(color='blue', width=2)))

# Ajout de la prédiction Régression Linéaire
fig.add_trace(go.Scatter(x=test.index, y=pred_reg, mode='lines', name='Régression Linéaire', line=dict(color='red', dash='dash')))

# Ajout de la prédiction ARIMA
fig.add_trace(go.Scatter(x=test.index, y=pred_arima, mode='lines', name='ARIMA (5,1,0)', line=dict(color='green', dash='dot')))

# Mise en forme du graphique
fig.update_layout(
    title='Prévision de la Consommation Électrique (Forecasting)',
    xaxis_title='Date',
    yaxis_title='Consommation Moyenne (kWh)',
    template='plotly_white',
    legend=dict(x=0.01, y=0.99)
)

# Afficher la courbe
fig.show()
