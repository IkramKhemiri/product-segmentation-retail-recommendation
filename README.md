# 🛍️ Product Segmentation & Retail Recommendation System

## 🎯 Plateforme d’intelligence retail: segmentation produits, scoring promos, recommandations

<div align="center">

### 📸 Galerie de l’application
<div style="display: flex; flex-wrap: wrap; justify-content: center; gap: 10px; padding: 20px 0;">
<!-- Galerie unifiée sans titres, même disposition -->
<img src="Capture d'écran 2025-12-10 184319.png" width="280" height="180" alt="">
<img src="Capture d'écran 2025-12-10 184335.png" width="280" height="180" alt="">
<img src="Capture d'écran 2025-12-10 184412.png" width="280" height="180" alt="">
<img src="Capture d'écran 2025-12-10 184429.png" width="280" height="180" alt="">
<img src="Capture d'écran 2025-12-10 184448.png" width="280" height="180" alt="">
<img src="Capture d'écran 2025-12-10 184505.png" width="280" height="180" alt="">
<img src="Capture d'écran 2025-12-10 184631.png" width="280" height="180" alt="">
<img src="Capture d'écran 2025-12-10 184659.png" width="280" height="180" alt="">
<img src="Capture d'écran 2025-12-10 184745.png" width="280" height="180" alt="">
</div>

</div>

> Suite unifiée pour piloter le retail: segmentation produits/clients, scoring promotionnel, recommandations, saisonnalité et benchmark d’algorithmes.

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)

---

## 📚 Sommaire
- Présentation
- Fonctionnalités principales
- Architecture & Stack
- Données attendues
- Installation rapide (Windows)
- Guide d’utilisation
- Performance & Configuration
- Qualité & Validation
- Roadmap
- Dépannage (FAQ)
- Licence & Contact

---

## 🧭 Présentation

Ce projet propose une application Streamlit prête à l’emploi pour:
- Nettoyage et agrégation des données retail
- Segmentation produits/clients par clustering
- Scoring promotionnel pondéré (stock, vitesse, marge, retours)
- Recommandations (best sellers, promo, upsell/cross-sell)
- Visualisations avancées (saisonnalité, tendances, heatmaps)
- Comparaison d’algorithmes et sélection automatique du meilleur

Points forts:
- Segments métier: Best Sellers, Slow Movers, Collections Limitées, Promo Forte
- Bench clustering: K-Means, DBSCAN, GMM, OPTICS, K-Medoids
- Robustesse aux données manquantes et colonnes variables
- Échantillonnage automatique pour éviter les erreurs de taille côté frontend

---

## ✨ Fonctionnalités principales

- Tableau de bord
  - Treemap catégories vs pression promo
  - Ventes par saison, distributions de prix
  - Scatter interactif (prix vs rating/velocity)

- Segmentation
  - Découpage par quantiles + règles métier
  - Compteurs synthétiques par segment
  - Visualisation et export des segments

- Scoring Promotionnel
  - Score pondéré (Sur-stock, Ventes lentes, Marge, Qualité/Retours)
  - Décomposition du score par produit
  - Classement en niveaux de promo (Faible, Modérée, Forte)

- Recommandations
  - Best sellers orientés valeur
  - Candidats promo (déstockage intelligent)
  - Upsell/cross-sell (qualité/marge/velocity)

- Insights Clients
  - Catégories top/bas par segment client
  - Saisonnalité et tendances mensuelles
  - Heatmaps segments x catégories

- Comparaison d’Algorithmes
  - Metrics: Silhouette, Davies-Bouldin, Calinski-Harabasz
  - Visualisations des clusters et projections PCA/UMAP
  - Sauvegarde/chargement de pickles de résultats

---

## 🏗️ Architecture & Stack

Structure recommandée:
- app_streamlit.py — Application Streamlit
- ProductSegmentationRetailRecommendationSystem.py — EDA + pipelines clustering
- data/retail_data.csv — Données source (transactions + produits)
- models/produits_comparison_results.pkl — Clustering produits
- models/clustering_results_clients_*.pkl — Clustering clients
- assets/images/ — Captures d’écran

Stack:
- Frontend: Streamlit, Plotly
- Backend: Python 3.9+
- Data: Pandas, NumPy
- ML: Scikit-learn, scikit-learn-extra (K-Medoids)
- Manifold: PCA, t-SNE, UMAP (optionnel)
- Metrics: Silhouette, DB, CH
- Sérialisation: Pickle

---

## 📈 Données attendues

Colonnes conseillées:
- Identifiants: product_id, customer_id, transaction_id
- Produits: product_category, unit_price, product_stock, product_rating, product_return_rate
- Transactions: quantity, discount_applied, transaction_date, season
- Ventes: total_sales (sinon calcul = quantity × unit_price)

Notes:
- Gestion des NaN/valeurs invalides
- Détection auto de `total_sales` sinon calcul à la volée
- Agrégations multi-vues (produit vs transaction)

---

## 🚀 Installation rapide (Windows)

```bash
# Créer et activer l’environnement
py -m venv .venv
.\.venv\Scripts\activate

# Installer dépendances
pip install -r requirements.txt

# Lancer l’application
streamlit run app_streamlit.py
```

requirements.txt (exemple):
- streamlit
- pandas
- numpy
- scikit-learn
- plotly
- seaborn
- matplotlib
- scikit-learn-extra  # optionnel (K-Medoids)
- umap-learn          # optionnel

---

## 💻 Guide d’utilisation

1) Filtres & Pondérations
- Ajuster catégories, saisons, prix, stock, rating, discount, retours
- Régler les poids du score promo: Sur-stock, Ventes lentes, Marge, Qualité

2) Segmentation & Promotion
- Visualiser segments et compteurs
- Décomposer le score pour expliquer les décisions

3) Recommandations
- Parcourir best sellers et upsell
- Identifier les candidats promo pour déstockage

4) Insights Clients
- Top produits par saison
- Heatmaps segments x catégories
- Tendances mensuelles

5) Benchmark Clustering
- Charger pickles produits/clients
- Comparer les algorithmes, visualiser clusters
- Mapper labels sur la vue transactionnelle

---

## ⚙️ Performance & Configuration

Problèmes de taille (MessageSizeError):
- Ne pas envoyer > 200MB au frontend
- Stratégies intégrées:
  - Échantillonnage des grands DataFrames avant `st.plotly_chart`/`st.dataframe`
  - Limitation Top N (20/30) pour bar/treemap
- Config locale (optionnelle): `.streamlit/config.toml`
  - `[server] maxMessageSize = 200`

Bonnes pratiques:
- Pré-calculer agrégations lourdes
- Limiter points en scatter
- Stocker pickles/images lourdes dans `models/` et `assets/`

---

## 🧪 Qualité & Validation

Tests suggérés:
- Calcul du score promo (pondérations, normalisation MinMax)
- Pipelines features (remplissage NaN, typage, standardisation)
- Sampling pour grands DataFrames

Validation:
- Cohérence segments vs métriques (vente, marge, retours)
- Stabilité Silhouette/DB à travers runs
- Vérification de l’impact des pondérations sur le score promo

---

## 🔮 Roadmap

- Explainable AI: SHAP/LIME sur score promo et clusters
- API REST: recommandations en temps réel
- Apprentissage incrémental: mise à jour continue des clusters
- RFM/CLV/Cohortes: enrichir profils clients
- Association rules: bundles/cross-sell intégrés
- Gouvernance data: traçabilité, qualité, catalogage

---

## 🛠️ Dépannage (FAQ)

- Erreur `MessageSizeError`:
  - Échantillonner (ex: 10–20k lignes), limiter Top N sur graphiques
- `NameError: df_tmp`:
  - Utiliser `df_master`/`df_local` dans les agrégations et graphiques
- Graphiques vides:
  - Vérifier colonnes (`product_category`, `total_sales` ou `quantity+unit_price`)
- Lenteur:
  - Activer sampling, réduire points scatter, éviter les recalculs lourds

---

## 📜 Licence

Projet destiné à des usages académiques et analytics retail. Adapter la licence selon votre besoin (MIT recommandée pour open-source).

---

## 📞 Contact

- Auteur: Ikram Khemiri
- Support: voir logs Streamlit et notifications interface
- Suggestions: ouvrir une issue sur le repository


