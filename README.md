# Kaggle House Prices en R

## Objectif

Prédire `SalePrice`, le prix de vente de chaque maison du jeu de test. Kaggle
évalue les prédictions avec la RMSE calculée sur les logarithmes des prix.

## Télécharger les données

1. Ouvrir <https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data>.
2. Cliquer sur **Join Competition** et accepter les règles.
3. Télécharger puis décompresser les données.
4. Placer dans `house_prices/data/raw/` :
   - `train.csv`
   - `test.csv`
   - `sample_submission.csv`
   - `data_description.txt` (recommandé)

## Exécution dans RStudio

Depuis la console, avec `Projet_R` comme dossier de travail :

```r
setwd("C:/Users/Gorkus/Documents/Codex/Projet_R")
source("house_prices/01_exploration.R")
source("house_prices/02_validation_baseline.R")
source("house_prices/03_validation_croisee.R")
source("house_prices/04_creer_soumission.R")
```

Les commandes `source(...)` doivent être saisies dans la console et non
collées dans le contenu des scripts.

## Parcours prévu

1. Exploration et audit des données
2. Définition de la validation et de la métrique RMSLE/RMSE-log
3. Nettoyage et imputation sans fuite
4. Modèle naïf et régression linéaire
5. Modèles régularisés et modèles d'arbres
6. Explicabilité et analyse des erreurs
7. Entraînement final et création de la soumission Kaggle
