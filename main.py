import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ÉTAPE 1: CHARGEMENT DES DONNÉES
print("ÉTAPE 1: CHARGEMENT DES DONNÉES")
print("-" * 50)

# Charger les données (nous supposons que le fichier CSV est dans le répertoire courant)
# index_col=0 indique que la première colonne doit être utilisée comme index
# parse_dates=True convertit les dates en objets datetime
data = pd.read_csv('energy_data1.csv', index_col=0, parse_dates=True)

# Afficher les 5 premières lignes du jeu de données
print("Aperçu du jeu de données:")
print(data.head())

# Afficher les informations sur le jeu de données
print("\nInformations sur le jeu de données:")
print(f"Nombre de lignes: {data.shape[0]}")
print(f"Nombre de colonnes: {data.shape[1]}")
print("\nTypes de données:")
print(data.dtypes)

# Vérifier s'il y a des valeurs manquantes
print("\nValeurs manquantes par colonne:")
print(data.isnull().sum())

# ÉTAPE 2: ANALYSE EXPLORATOIRE DES DONNÉES
print("\nÉTAPE 2: ANALYSE EXPLORATOIRE DES DONNÉES")
print("-" * 50)

# Statistiques descriptives
print("Statistiques descriptives:")
print(data.describe())

# Visualisation de la distribution de la consommation énergétique
plt.figure(figsize=(10, 6))
plt.hist(data['consommation_kWh'], bins=20, color='skyblue', edgecolor='black')
plt.title('Distribution de la consommation énergétique')
plt.xlabel('Consommation (kWh)')
plt.ylabel('Fréquence')
plt.grid(True, alpha=0.3)
plt.savefig('distribution_consommation.png')
print("Figure 'distribution_consommation.png' sauvegardée.")

# Visualisation de la consommation énergétique au fil du temps
plt.figure(figsize=(12, 6))
plt.plot(data.index, data['consommation_kWh'], color='blue')
plt.title('Consommation énergétique au fil du temps')
plt.xlabel('Date')
plt.ylabel('Consommation (kWh)')
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('consommation_temps.png')
print("Figure 'consommation_temps.png' sauvegardée.")

# Matrice de corrélation
plt.figure(figsize=(10, 8))
correlation_matrix = data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Matrice de corrélation')
plt.tight_layout()
plt.savefig('correlation_matrix.png')
print("Figure 'correlation_matrix.png' sauvegardée.")

# Consommation moyenne par heure
plt.figure(figsize=(12, 6))
hourly_consumption = data.groupby('heure')['consommation_kWh'].mean()
hourly_consumption.plot(kind='bar', color='skyblue')
plt.title('Consommation moyenne par heure')
plt.xlabel('Heure')
plt.ylabel('Consommation moyenne (kWh)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('consommation_par_heure.png')
print("Figure 'consommation_par_heure.png' sauvegardée.")

# Consommation en fonction de la présence
plt.figure(figsize=(10, 6))
sns.boxplot(x='presence', y='consommation_kWh', data=data)
plt.title('Consommation énergétique en fonction de la présence')
plt.xlabel('Présence (0=Absent, 1=Présent)')
plt.ylabel('Consommation (kWh)')
plt.grid(True, alpha=0.3)
plt.savefig('consommation_presence.png')
print("Figure 'consommation_presence.png' sauvegardée.")

# ÉTAPE 3: PRÉPARATION DES DONNÉES
print("\nÉTAPE 3: PRÉPARATION DES DONNÉES")
print("-" * 50)

# Création de caractéristiques (features) temporelles cycliques
# Les valeurs cycliques comme l'heure et le jour de la semaine sont mieux représentées
# par des transformations en sinus et cosinus pour préserver leur nature cyclique
print("Création de caractéristiques temporelles cycliques...")
data['sin_heure'] = np.sin(2 * np.pi * data['heure']/24)
data['cos_heure'] = np.cos(2 * np.pi * data['heure']/24)
data['sin_jour'] = np.sin(2 * np.pi * data['jour_semaine']/7)
data['cos_jour'] = np.cos(2 * np.pi * data['jour_semaine']/7)

# Ajout de caractéristiques de décalage (lag features)
print("Ajout de caractéristiques de décalage temporel...")
data['consommation_lag1'] = data['consommation_kWh'].shift(1)  # Consommation 1 heure avant
data['consommation_lag2'] = data['consommation_kWh'].shift(2)  # Consommation 2 heures avant

# Ajout d'une moyenne mobile sur 3 heures
print("Calcul de la moyenne mobile sur 3 heures...")
data['moyenne_mobile_3h'] = data['consommation_kWh'].rolling(window=3).mean()

# Suppression des lignes avec des valeurs manquantes (dues aux décalages)
data_clean = data.dropna()
print(f"Nombre de lignes après nettoyage: {data_clean.shape[0]}")

# Séparation des caractéristiques (X) et de la variable cible (y)
X = data_clean.drop(['consommation_kWh'], axis=1)
y = data_clean['consommation_kWh']

# Division en ensembles d'entraînement et de test (80% / 20%)
# Pour les séries temporelles, on respecte l'ordre chronologique
train_size = int(len(X) * 0.8)
X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

print(f"Taille de l'ensemble d'entraînement: {X_train.shape[0]} observations")
print(f"Taille de l'ensemble de test: {X_test.shape[0]} observations")

# ÉTAPE 4: MODÉLISATION AVEC RANDOM FOREST
print("\nÉTAPE 4: MODÉLISATION AVEC RANDOM FOREST")
print("-" * 50)

print("Entraînement du modèle Random Forest...")
# Initialisation du modèle avec 100 arbres
rf_model = RandomForestRegressor(
    n_estimators=100,  # Nombre d'arbres dans la forêt
    max_depth=None,    # Profondeur maximale des arbres (None = pas de limite)
    min_samples_split=2,  # Nombre minimum d'échantillons requis pour diviser un nœud
    min_samples_leaf=1,   # Nombre minimum d'échantillons requis pour être une feuille
    random_state=42       # Pour la reproductibilité des résultats
)

# Entraînement du modèle
rf_model.fit(X_train, y_train)

# Prédictions sur l'ensemble de test
y_pred = rf_model.predict(X_test)

# ÉTAPE 5: ÉVALUATION DU MODÈLE
print("\nÉTAPE 5: ÉVALUATION DU MODÈLE")
print("-" * 50)

# Calcul des métriques d'évaluation
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("Résultats de l'évaluation du modèle:")
print(f"MAE (Erreur Absolue Moyenne): {mae:.4f} kWh")
print(f"RMSE (Racine de l'Erreur Quadratique Moyenne): {rmse:.4f} kWh")
print(f"R² (Coefficient de détermination): {r2:.4f}")

# Visualisation des prédictions vs valeurs réelles
plt.figure(figsize=(12, 6))
plt.plot(y_test.index, y_test.values, label='Valeurs réelles', color='blue')
plt.plot(y_test.index, y_pred, label='Prédictions', color='red', linestyle='--')
plt.title('Consommation réelle vs prédictions')
plt.xlabel('Date')
plt.ylabel('Consommation (kWh)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('predictions_vs_reel.png')
print("Figure 'predictions_vs_reel.png' sauvegardée.")

# ÉTAPE 6: IMPORTANCE DES CARACTÉRISTIQUES
print("\nÉTAPE 6: IMPORTANCE DES CARACTÉRISTIQUES")
print("-" * 50)

# Extraction et tri des importances de caractéristiques
feature_importances = pd.DataFrame({
    'Caractéristique': X_train.columns,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("Importance des caractéristiques:")
print(feature_importances)

# Visualisation des importances de caractéristiques
plt.figure(figsize=(10, 8))
sns.barplot(x='Importance', y='Caractéristique', data=feature_importances)
plt.title('Importance des caractéristiques dans le modèle Random Forest')
plt.xlabel('Importance')
plt.ylabel('Caractéristique')
plt.tight_layout()
plt.savefig('importance_features.png')
print("Figure 'importance_features.png' sauvegardée.")

# ÉTAPE 7: PRÉDICTION POUR LES PROCHAINES 24 HEURES
print("\nÉTAPE 7: PRÉDICTION POUR LES PROCHAINES 24 HEURES")
print("-" * 50)

def predire_prochaines_heures(modele, derniere_date, df_original, nb_heures=24):
    """
    Fonction pour prédire la consommation énergétique pour les prochaines heures.
    
    Args:
        modele: Le modèle de prédiction entraîné
        derniere_date: La dernière date du jeu de données original
        df_original: Le DataFrame original complet
        nb_heures: Le nombre d'heures à prédire
        
    Returns:
        DataFrame avec les prédictions
    """
    print(f"Prédiction de la consommation pour les prochaines {nb_heures} heures...")
    
    # Créer un DataFrame pour les dates futures
    dates_futures = pd.date_range(start=derniere_date + pd.Timedelta(hours=1), 
                                 periods=nb_heures, freq='H')
    df_futur = pd.DataFrame(index=dates_futures)
    
    # Pour la démonstration, on réutilise les valeurs des dernières 24h 
    # avec une petite variation aléatoire
    reference_data = df_original.iloc[-nb_heures:].reset_index(drop=True)
    
    # Copier les caractéristiques
    for col in ['temperature', 'humidite', 'presence', 'eclairage_naturel']:
        if col == 'presence':
            # Pour la présence, on garde les mêmes motifs
            df_futur[col] = reference_data[col].values
        else:
            # Pour les autres, on ajoute une légère variation
            df_futur[col] = reference_data[col].values + np.random.normal(0, 0.5, size=nb_heures)
    
    # Ajouter jour de la semaine et heure
    df_futur['jour_semaine'] = df_futur.index.dayofweek
    df_futur['heure'] = df_futur.index.hour
    
    # Créer les mêmes transformations que pour l'entraînement
    df_futur['sin_heure'] = np.sin(2 * np.pi * df_futur['heure']/24)
    df_futur['cos_heure'] = np.cos(2 * np.pi * df_futur['heure']/24)
    df_futur['sin_jour'] = np.sin(2 * np.pi * df_futur['jour_semaine']/7)
    df_futur['cos_jour'] = np.cos(2 * np.pi * df_futur['jour_semaine']/7)
    
    # Récupérer les dernières valeurs connues pour les lags
    dernieres_consos = list(df_original['consommation_kWh'].iloc[-3:])
    
    # Prédire de manière itérative
    predictions = []
    
    for i in range(nb_heures):
        # Mettre à jour les lags
        df_futur.loc[df_futur.index[i], 'consommation_lag1'] = dernieres_consos[-1]
        df_futur.loc[df_futur.index[i], 'consommation_lag2'] = dernieres_consos[-2]
        
        # Mettre à jour la moyenne mobile
        df_futur.loc[df_futur.index[i], 'moyenne_mobile_3h'] = np.mean(dernieres_consos[-3:])
        
        # Prédire
        X_pred = df_futur.iloc[[i]]
        pred = modele.predict(X_pred)[0]
        predictions.append(pred)
        
        # Mettre à jour les valeurs pour la prochaine itération
        dernieres_consos.append(pred)
        dernieres_consos.pop(0)
    
    # Créer un DataFrame avec les prédictions
    df_predictions = pd.DataFrame({
        'date': df_futur.index,
        'consommation_prevue': predictions
    })
    
    return df_predictions

# Prédire pour les prochaines 24 heures
predictions_futures = predire_prochaines_heures(rf_model, data.index[-1], data)

# Visualiser les prédictions futures
plt.figure(figsize=(12, 6))
plt.plot(data.index[-48:], data['consommation_kWh'].iloc[-48:], 
         label='Historique (2 derniers jours)', color='blue')
plt.plot(predictions_futures['date'], predictions_futures['consommation_prevue'], 
         label='Prévisions (24h)', color='red', linestyle='--')
plt.axvline(x=data.index[-1], color='black', linestyle='-.', label='Maintenant')
plt.title('Prévision de consommation énergétique pour les prochaines 24 heures')
plt.xlabel('Date')
plt.ylabel('Consommation (kWh)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('predictions_futures.png')
print("Figure 'predictions_futures.png' sauvegardée.")

# ÉTAPE 8: CONCLUSION
print("\nÉTAPE 8: CONCLUSION")
print("-" * 50)

print("Résumé de l'analyse et de la modélisation:")
print(f"1. Le modèle Random Forest a obtenu un R² de {r2:.4f}, expliquant {r2*100:.1f}% de la variance dans la consommation énergétique.")
print(f"2. L'erreur moyenne absolue est de {mae:.4f} kWh.")
print("3. Les caractéristiques les plus importantes sont:", feature_importances['Caractéristique'].iloc[0], "et", feature_importances['Caractéristique'].iloc[1])
print("4. Des prédictions pour les 24 prochaines heures ont été générées et enregistrées.")

print("\nRemarque: Pour une utilisation en production, il est recommandé:")
print("- De collecter plus de données pour capturer les tendances saisonnières")
print("- D'optimiser les hyperparamètres du modèle via une recherche par grille")
print("- De mettre en place un système de réentraînement régulier du modèle")