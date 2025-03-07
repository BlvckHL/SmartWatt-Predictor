import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Charger les données
energy_data = pd.read_csv('energy_data1.csv', index_col=0, parse_dates=True)
predictions_futures = pd.read_csv('predictions_futures.csv', parse_dates=['date'])

# Initialiser l'application Dash
app = dash.Dash(__name__)

# Layout du dashboard
app.layout = html.Div([
    html.H1("SmartWatt Predictor Dashboard", style={'textAlign': 'center'}),

    # Graphique Historique de Consommation
    dcc.Graph(id='historique-consommation'),
    html.Div([
        html.H3("Interprétation de la Consommation Historique"),
        dcc.Markdown("""
        - Le graphique ci-dessus montre l'évolution de la consommation énergétique au fil du temps.
        - Les pics dans la consommation peuvent être dus à l'utilisation accrue des appareils électroménagers ou des changements de température.
        - Les tendances visibles indiquent des périodes de forte ou faible demande, ce qui peut aider à ajuster les prévisions énergétiques futures.
        """)
    ], style={'padding': '20px'}),

    # Graphique de Prédictions Futures
    dcc.Graph(id='prediction-futures'),
    html.Div([
        html.H3("Interprétation des Prédictions Futures"),
        dcc.Markdown("""
        - Le graphique des prédictions montre les prévisions de consommation énergétique pour les 24 prochaines heures.
        - Les prévisions sont basées sur des modèles qui analysent les tendances historiques, mais peuvent être influencées par des facteurs externes comme la météo.
        - Comparer ces prévisions avec la consommation historique peut aider à anticiper les besoins énergétiques et prendre des décisions plus éclairées.
        """)
    ], style={'padding': '20px'}),

    # Graphique de l'Importance des Features
    dcc.Graph(id='importance-features'),
    html.Div([
        html.H3("Interprétation de l'Importance des Features"),
        dcc.Markdown("""
        - Ce graphique montre l'importance relative des différentes caractéristiques (température, heure de la journée, etc.) dans le modèle de prédiction.
        - Une caractéristique avec une importance élevée signifie qu'elle influence fortement les prévisions de consommation.
        - Comprendre ces influences permet d'ajuster les paramètres du modèle pour améliorer la précision des prédictions.
        """)
    ], style={'padding': '20px'})
])

# Callback pour le graphique Historique de Consommation
@app.callback(
    Output('historique-consommation', 'figure'),
    Input('historique-consommation', 'id')
)
def update_historique(_):
    fig = px.line(energy_data, x=energy_data.index, y='consommation_kWh', title='Consommation Historique')
    return fig

# Callback pour les Prédictions Futures
@app.callback(
    Output('prediction-futures', 'figure'),
    Input('prediction-futures', 'id')
)
def update_predictions(_):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=energy_data.index[-48:], y=energy_data['consommation_kWh'].iloc[-48:],
                             mode='lines', name='Historique', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=predictions_futures['date'], y=predictions_futures['consommation_prevue'],
                             mode='lines', name='Prévisions', line=dict(color='red', dash='dash')))
    fig.update_layout(title='Prédictions des 24 prochaines heures', xaxis_title='Date', yaxis_title='Consommation (kWh)')
    return fig

# Callback pour l'Importance des Features (si nécessaire)
@app.callback(
    Output('importance-features', 'figure'),
    Input('importance-features', 'id')
)
def update_importance(_):
    # Exemple d'un graphique fictif pour l'importance des features, remplace-le par tes données réelles
    importance_data = {'Features': ['Température', 'Heure', 'Présence', 'Luminosité'],
                       'Importance': [0.4, 0.3, 0.2, 0.1]}
    importance_df = pd.DataFrame(importance_data)
    fig = px.bar(importance_df, x='Features', y='Importance', title="Importance des Caractéristiques")
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)
