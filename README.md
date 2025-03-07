# SmartWatt Predictor 🔋

## Description
SmartWatt Predictor est un système d'intelligence artificielle conçu pour prédire la consommation énergétique horaire. En utilisant des techniques avancées de machine learning, notamment Random Forest, le système analyse divers facteurs environnementaux et comportementaux pour fournir des prédictions précises.

## ✨ Fonctionnalités
- Prédiction de consommation sur 24h
- Analyse détaillée des patterns de consommation
- Visualisations interactives
- Évaluation des performances du modèle
- Analyse de l'importance des caractéristiques

## 🛠️ Technologies
- Python 3.x
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn

## 📋 Prérequis
- Python 3.8 ou supérieur
- pip (gestionnaire de paquets Python)
- Git

## 🚀 Installation

1. Clonez le repository :
```bash
git clone https://github.com/BlvckHL/SmartWatt-Predictor.git
cd SmartWatt-Predictor
```

2. Installez les dépendances :
```bash
pip install -r requirements.txt
```


## 📊 Structure du Projet
```
SmartWatt-Predictor/
├── main.py              # Script principal
├── energy_data.csv      # Données d'entraînement
├── requirements.txt     # Dépendances
└── README.md           # Documentation
```

## 💻 Utilisation

Exécutez le script principal :
```bash
python main.py
```
Ensuite, pour lancer le tableau de bord, exécutez :
```bash
python SmartWattDashboard.py
```

## 📈 Visualisations Générées

Le script génère plusieurs visualisations dans le répertoire courant :
- `distribution_consommation.png` - Distribution de la consommation
- `consommation_temps.png` - Évolution temporelle
- `correlation_matrix.png` - Corrélations entre variables
- `consommation_par_heure.png` - Patterns horaires
- `consommation_presence.png` - Impact de la présence
- `predictions_vs_reel.png` - Comparaison prédictions/réel
- `importance_features.png` - Importance des caractéristiques

## 📊 Métriques de Performance

Le modèle est évalué selon plusieurs métriques :
- MAE (Mean Absolute Error)
- RMSE (Root Mean Square Error)
- R² (Coefficient de détermination)

## 🔄 Format des Données d'Entrée

Le fichier `energy_data.csv` doit contenir les colonnes suivantes :
- `temperature` - Température ambiante
- `humidite` - Taux d'humidité
- `presence` - Détection de présence (0/1)
- `eclairage_naturel` - Niveau d'éclairage naturel
- `consommation_kWh` - Consommation énergétique
- `jour_semaine` - Jour de la semaine (0-6)
- `heure` - Heure de la journée (0-23)

## 📖 Guide Détaillé du Fonctionnement

### 1. Les Données Utilisées 📊
Le système analyse les données suivantes :
- Température ambiante
- Taux d'humidité
- Présence de personnes (détection binaire)
- Niveau d'éclairage naturel
- Horodatage (heure et jour)
- Historique de consommation électrique

### 2. Processus de Fonctionnement 🔄

#### A. Phase de Préparation
- Chargement des données historiques
- Création de visualisations :
  * Graphiques de tendances
  * Analyses de corrélation
  * Patterns de consommation journaliers

#### B. Phase d'Apprentissage
Le modèle Random Forest :
- Analyse les données historiques
- Détecte les patterns récurrents
- Établit des corrélations entre variables

#### C. Phase de Prédiction
Capacités du système :
- Génération de prédictions sur 24h
- Estimation des consommations futures
- Évaluation de la fiabilité des prédictions

### 3. Applications Pratiques 💡
Utilisations possibles :
- Planification énergétique
- Optimisation de la consommation
- Réduction des coûts
- Amélioration de l'efficacité

### 4. Avantages du Système 🌟
- **Précision** : Analyse multi-factorielle
- **Clarté** : Visualisations intuitives
- **Évolutivité** : Apprentissage continu

### 5. Pour les Débutants 🎓
Imaginez ce système comme une "météo de la consommation électrique" : 
- Il analyse le passé
- Comprend les tendances
- Prédit les besoins futurs

## 🚧 Améliorations Futures
- [ ] Optimisation des hyperparamètres
- [ ] Interface utilisateur web
- [ ] API REST pour les prédictions
- [ ] Support de données en temps réel
- [ ] Intégration de variables météorologiques

## 👤 Auteur
[Linerol]
[Hanim]

## 📄 Licence
Ce projet est sous licence MIT. Voir le fichier LICENSE pour plus de détails.
