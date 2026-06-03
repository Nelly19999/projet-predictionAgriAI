# Application Streamlit - Prediction des recoltes au Burundi

Cette application permet de saisir les caracteristiques d'une parcelle agricole et de predire si la recolte sera bonne ou mauvaise.

## Lancer l'application

```bash
pip install -r requirements.txt
streamlit run hello.py
```

Vous pouvez aussi lancer :

```bash
streamlit run app.py
```

## Fonctionnalites

- Formulaire de saisie des donnees agricoles
- Choix du modele : arbre de decision, foret aleatoire, regression logistique ou modele simple
- Prediction en temps reel avec probabilite associee
- Affichage des metriques globales : accuracy, F1 et AUC
- Graphique d'importance des variables ou des coefficients
