# app_streamlit.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Configuration de la page
st.set_page_config(
    page_title="Système de Segmentation Retail",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style CSS personnalisé
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
    }
    .section-header {
        color: #1f77b4;
        border-bottom: 2px solid #1f77b4;
        padding-bottom: 0.5rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Titre principal
st.markdown('<h1 class="main-header">🏪 Tableau de Bord - Segmentation Retail</h1>', unsafe_allow_html=True)

# Sidebar - Filtres et navigation
st.sidebar.title("🎛️ Navigation & Filtres")

# Chargement des données
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("retail_data.csv")
        return df
    except:
        # Création de données d'exemple si le fichier n'existe pas
        np.random.seed(42)
        n_samples = 10000
        data = {
            'customer_id': range(n_samples),
            'product_id': np.random.randint(1000, 2000, n_samples),
            'age': np.random.randint(18, 70, n_samples),
            'gender': np.random.choice(['Male', 'Female'], n_samples),
            'total_sales': np.random.exponential(100, n_samples),
            'product_category': np.random.choice(['Vêtements', 'Chaussures', 'Accessoires', 'Électronique', 'Maison'], n_samples),
            'product_rating': np.random.uniform(3, 5, n_samples),
            'quantity': np.random.randint(1, 10, n_samples),
            'unit_price': np.random.uniform(10, 200, n_samples),
            'discount_applied': np.random.uniform(0, 0.5, n_samples),
            'product_stock': np.random.randint(0, 1000, n_samples),
            'product_return_rate': np.random.uniform(0, 0.2, n_samples),
            'season': np.random.choice(['Printemps', 'Été', 'Automne', 'Hiver'], n_samples),
            'region': np.random.choice(['Nord', 'Sud', 'Est', 'Ouest'], n_samples),
            'loyalty_program': np.random.choice(['Standard', 'Premium', 'VIP'], n_samples)
        }
        return pd.DataFrame(data)

df = load_data()

# Filtres dans la sidebar
st.sidebar.markdown("### 🔍 Filtres des Données")

# Filtre par catégorie
categories = ['Toutes'] + list(df['product_category'].unique())
selected_category = st.sidebar.selectbox('Catégorie Produit', categories)

# Filtre par saison
saisons = ['Toutes'] + list(df['season'].unique())
selected_season = st.sidebar.selectbox('Saison', saisons)

# Filtre par région
regions = ['Toutes'] + list(df['region'].unique())
selected_region = st.sidebar.selectbox('Région', regions)

# Filtre par programme de fidélité
loyalty_programs = ['Tous'] + list(df['loyalty_program'].unique())
selected_loyalty = st.sidebar.selectbox('Programme Fidélité', loyalty_programs)

# Application des filtres
filtered_df = df.copy()
if selected_category != 'Toutes':
    filtered_df = filtered_df[filtered_df['product_category'] == selected_category]
if selected_season != 'Toutes':
    filtered_df = filtered_df[filtered_df['season'] == selected_season]
if selected_region != 'Toutes':
    filtered_df = filtered_df[filtered_df['region'] == selected_region]
if selected_loyalty != 'Tous':
    filtered_df = filtered_df[filtered_df['loyalty_program'] == selected_loyalty]

# Navigation
st.sidebar.markdown("### 📊 Navigation")
page = st.sidebar.radio("Sections", [
    "📈 Vue d'Ensemble", 
    "👥 Segmentation Clients", 
    "📦 Segmentation Produits",
    "🎯 Système de Recommandation",
    "📋 Exploration des Données"
])

# SECTION 1: VUE D'ENSEMBLE
if page == "📈 Vue d'Ensemble":
    st.markdown('<h2 class="section-header">📈 Vue d\'Ensemble et Métriques Clés</h2>', unsafe_allow_html=True)
    
    # Métriques principales
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_sales = filtered_df['total_sales'].sum()
        st.metric("💰 Chiffre d'Affaires Total", f"${total_sales:,.0f}")
    
    with col2:
        avg_sales = filtered_df['total_sales'].mean()
        st.metric("📊 Vente Moyenne", f"${avg_sales:.2f}")
    
    with col3:
        total_customers = filtered_df['customer_id'].nunique()
        st.metric("👥 Clients Uniques", f"{total_customers:,}")
    
    with col4:
        total_products = filtered_df['product_id'].nunique()
        st.metric("📦 Produits Uniques", f"{total_products:,}")
    
    # Deuxième ligne de métriques
    col5, col6, col7, col8 = st.columns(4)
    
    with col5:
        avg_rating = filtered_df['product_rating'].mean()
        st.metric("⭐ Rating Moyen", f"{avg_rating:.2f}")
    
    with col6:
        return_rate = filtered_df['product_return_rate'].mean()
        st.metric("🔄 Taux de Retour", f"{return_rate:.2%}")
    
    with col7:
        avg_discount = filtered_df['discount_applied'].mean()
        st.metric("🎫 Discount Moyen", f"{avg_discount:.1%}")
    
    with col8:
        stock_level = filtered_df['product_stock'].sum()
        st.metric("📦 Stock Total", f"{stock_level:,}")
    
    # Graphiques principaux
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Ventes par Catégorie")
        sales_by_category = filtered_df.groupby('product_category')['total_sales'].sum().sort_values(ascending=False)
        fig = px.bar(sales_by_category, x=sales_by_category.index, y=sales_by_category.values,
                    color=sales_by_category.values, color_continuous_scale='viridis')
        fig.update_layout(xaxis_title="Catégorie", yaxis_title="Ventes Total")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 👥 Répartition des Clients")
        gender_dist = filtered_df['gender'].value_counts()
        fig = px.pie(values=gender_dist.values, names=gender_dist.index, 
                    title="Répartition par Genre")
        st.plotly_chart(fig, use_container_width=True)
    
    # Troisième ligne de graphiques
    col3, col4 = st.columns(2)
    
    with col3:
        st.markdown("### 📈 Performance par Saison")
        seasonal_sales = filtered_df.groupby('season')['total_sales'].mean()
        fig = px.line(x=seasonal_sales.index, y=seasonal_sales.values,
                     markers=True, title="Ventes Moyennes par Saison")
        fig.update_layout(xaxis_title="Saison", yaxis_title="Ventes Moyennes")
        st.plotly_chart(fig, use_container_width=True)
    
    with col4:
        st.markdown("### 🏆 Top Catégories par Rating")
        top_categories = filtered_df.groupby('product_category')['product_rating'].mean().nlargest(5)
        fig = px.bar(top_categories, x=top_categories.index, y=top_categories.values,
                    color=top_categories.values, color_continuous_scale='plasma')
        fig.update_layout(xaxis_title="Catégorie", yaxis_title="Rating Moyen")
        st.plotly_chart(fig, use_container_width=True)

# SECTION 2: SEGMENTATION CLIENTS
elif page == "👥 Segmentation Clients":
    st.markdown('<h2 class="section-header">👥 Analyse et Segmentation des Clients</h2>', unsafe_allow_html=True)
    
    # Création de segments clients simples
    filtered_df['client_segment'] = pd.qcut(filtered_df['total_sales'], q=3, 
                                          labels=['Low-Value', 'Medium-Value', 'High-Value'])
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Distribution des Segments Clients")
        segment_dist = filtered_df['client_segment'].value_counts()
        fig = px.pie(values=segment_dist.values, names=segment_dist.index,
                    title="Répartition des Segments Clients")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 💰 Ventes par Segment")
        sales_by_segment = filtered_df.groupby('client_segment')['total_sales'].agg(['mean', 'sum'])
        fig = px.bar(sales_by_segment, x=sales_by_segment.index, y=sales_by_segment['mean'],
                    title="Ventes Moyennes par Segment")
        fig.update_layout(xaxis_title="Segment Client", yaxis_title="Ventes Moyennes")
        st.plotly_chart(fig, use_container_width=True)
    
    # Analyse détaillée par segment
    st.markdown("### 🔍 Analyse Détaillée par Segment")
    
    selected_segment = st.selectbox("Sélectionnez un segment:", filtered_df['client_segment'].unique())
    
    segment_data = filtered_df[filtered_df['client_segment'] == selected_segment]
    
    col3, col4, col5 = st.columns(3)
    
    with col3:
        avg_age = segment_data['age'].mean()
        st.metric("📅 Âge Moyen", f"{avg_age:.1f} ans")
    
    with col4:
        avg_rating = segment_data['product_rating'].mean()
        st.metric("⭐ Rating Moyen", f"{avg_rating:.2f}")
    
    with col5:
        loyalty_dist = segment_data['loyalty_program'].value_counts().index[0]
        st.metric("🎯 Programme Fidélité Principal", loyalty_dist)
    
    # Comportement d'achat par segment
    col6, col7 = st.columns(2)
    
    with col6:
        st.markdown("### 🛒 Comportement d'Achat")
        metrics_segment = segment_data.groupby('product_category')['quantity'].sum().nlargest(5)
        fig = px.bar(metrics_segment, x=metrics_segment.index, y=metrics_segment.values,
                    title=f"Top Catégories Achetées - {selected_segment}")
        st.plotly_chart(fig, use_container_width=True)
    
    with col7:
        st.markdown("### 💳 Méthodes de Paiement Préférées")
        # Simulation de données de paiement
        payment_methods = ['Carte Credit', 'PayPal', 'Espèce', 'Virement']
        payment_dist = np.random.dirichlet(np.ones(4)) * 100
        fig = px.pie(values=payment_dist, names=payment_methods, 
                    title="Répartition des Méthodes de Paiement")
        st.plotly_chart(fig, use_container_width=True)

# SECTION 3: SEGMENTATION PRODUITS
elif page == "📦 Segmentation Produits":
    st.markdown('<h2 class="section-header">📦 Analyse et Segmentation des Produits</h2>', unsafe_allow_html=True)
    
    # Création de segments produits
    filtered_df['product_segment'] = pd.qcut(filtered_df['total_sales'], q=3,
                                           labels=['Low-Sales', 'Medium-Sales', 'High-Sales'])
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📈 Performance des Produits par Catégorie")
        category_performance = filtered_df.groupby('product_category').agg({
            'total_sales': 'sum',
            'product_rating': 'mean',
            'product_return_rate': 'mean'
        }).round(3)
        
        fig = go.Figure(data=[
            go.Bar(name='Ventes Total', x=category_performance.index, y=category_performance['total_sales']),
            go.Bar(name='Rating Moyen', x=category_performance.index, y=category_performance['product_rating'] * 10000),
            go.Bar(name='Taux Retour', x=category_performance.index, y=category_performance['product_return_rate'] * 100000)
        ])
        fig.update_layout(barmode='group', title="Performance par Catégorie (échelles ajustées)")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 🎯 Segments de Produits")
        product_segment_dist = filtered_df['product_segment'].value_counts()
        fig = px.pie(values=product_segment_dist.values, names=product_segment_dist.index,
                    title="Répartition des Segments Produits")
        st.plotly_chart(fig, use_container_width=True)
    
    # Analyse stock vs ventes
    st.markdown("### 📊 Analyse Stock vs Ventes")
    
    col3, col4 = st.columns(2)
    
    with col3:
        stock_vs_sales = filtered_df.groupby('product_category').agg({
            'product_stock': 'mean',
            'total_sales': 'mean'
        }).reset_index()
        
        fig = px.scatter(stock_vs_sales, x='product_stock', y='total_sales',
                        size='total_sales', color='product_category',
                        hover_name='product_category', title="Stock vs Ventes par Catégorie")
        st.plotly_chart(fig, use_container_width=True)
    
    with col4:
        st.markdown("### 🔄 Taux de Retour par Catégorie")
        returns_by_category = filtered_df.groupby('product_category')['product_return_rate'].mean().sort_values()
        fig = px.bar(returns_by_category, x=returns_by_category.values, y=returns_by_category.index,
                    orientation='h', title="Taux de Retour Moyen par Catégorie",
                    color=returns_by_category.values, color_continuous_scale='reds')
        st.plotly_chart(fig, use_container_width=True)
    
    # Top produits
    st.markdown("### 🏆 Top 10 Produits par Performance")
    
    top_products = filtered_df.groupby('product_id').agg({
        'total_sales': 'sum',
        'product_rating': 'mean',
        'quantity': 'sum',
        'product_category': 'first'
    }).nlargest(10, 'total_sales')
    
    st.dataframe(top_products.style.background_gradient(cmap='Blues'), use_container_width=True)

# SECTION 4: SYSTÈME DE RECOMMANDATION
elif page == "🎯 Système de Recommandation":
    st.markdown('<h2 class="section-header">🎯 Système de Recommandation Intelligent</h2>', unsafe_allow_html=True)
    
    # Calcul des scores de promotion
    def calculate_promotion_scores(df):
        df_scores = df.copy()
        
        # Scores simples pour la démonstration
        df_scores['score_stock'] = df_scores['product_stock'] / df_scores['product_stock'].max()
        df_scores['score_sales'] = df_scores['total_sales'] / df_scores['total_sales'].max()
        df_scores['score_rating'] = df_scores['product_rating'] / 5.0
        df_scores['score_return'] = 1 - df_scores['product_return_rate']
        
        # Score global de promotion
        df_scores['promotion_score'] = (
            0.3 * df_scores['score_stock'] +
            0.25 * (1 - df_scores['score_sales']) +  # Moins de ventes = plus de promotion nécessaire
            0.25 * df_scores['score_rating'] +
            0.2 * df_scores['score_return']
        )
        
        # Classification
        conditions = [
            df_scores['promotion_score'] >= 0.7,
            df_scores['promotion_score'] >= 0.4,
            df_scores['promotion_score'] < 0.4
        ]
        choices = ['🔴 Promotion Forte', '🟡 Promotion Modérée', '🟢 Promotion Faible']
        df_scores['promotion_level'] = np.select(conditions, choices, default='🟢 Promotion Faible')
        
        return df_scores
    
    df_with_scores = calculate_promotion_scores(filtered_df)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Distribution des Niveaux de Promotion")
        promotion_dist = df_with_scores['promotion_level'].value_counts()
        fig = px.pie(values=promotion_dist.values, names=promotion_dist.index,
                    title="Répartition des Recommandations de Promotion")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 🎯 Scores de Promotion par Catégorie")
        promotion_by_category = df_with_scores.groupby('product_category')['promotion_score'].mean().sort_values(ascending=False)
        fig = px.bar(promotion_by_category, x=promotion_by_category.index, y=promotion_by_category.values,
                    color=promotion_by_category.values, color_continuous_scale='RdYlGn_r')
        fig.update_layout(xaxis_title="Catégorie", yaxis_title="Score de Promotion Moyen")
        st.plotly_chart(fig, use_container_width=True)
    
    # Recommandations détaillées
    st.markdown("### 💡 Recommandations Détaillées par Produit")
    
    # Filtre pour voir les produits par niveau de promotion
    promotion_level = st.selectbox("Niveau de Promotion:", 
                                 ['Tous', '🔴 Promotion Forte', '🟡 Promotion Modérée', '🟢 Promotion Faible'])
    
    if promotion_level != 'Tous':
        recommendations = df_with_scores[df_with_scores['promotion_level'] == promotion_level]
    else:
        recommendations = df_with_scores
    
    # Affichage des recommandations
    st.markdown(f"#### 📋 Produits Recommandés ({len(recommendations)} produits)")
    
    # Top 20 produits pour la démonstration
    top_recommendations = recommendations.nlargest(20, 'promotion_score')[
        ['product_id', 'product_category', 'unit_price', 'product_rating', 
         'promotion_score', 'promotion_level']
    ]
    
    st.dataframe(top_recommendations.style.background_gradient(subset=['promotion_score'], cmap='RdYlGn_r'),
                use_container_width=True)
    
    # Analyse des opportunités
    st.markdown("### 📈 Opportunités Commerciales")
    
    col3, col4 = st.columns(2)
    
    with col3:
        # Produits à fort potentiel de promotion
        high_potential = df_with_scores[
            (df_with_scores['promotion_score'] > 0.6) & 
            (df_with_scores['product_rating'] > 4.0)
        ]
        st.metric("🎯 Produits Haut Potentiel", len(high_potential))
    
    with col4:
        # Produits nécessitant une attention
        need_attention = df_with_scores[
            (df_with_scores['promotion_score'] > 0.7) & 
            (df_with_scores['product_rating'] < 3.5)
        ]
        st.metric("⚠️ Produits à Revoir", len(need_attention))

# SECTION 5: EXPLORATION DES DONNÉES
else:
    st.markdown('<h2 class="section-header">📋 Exploration Avancée des Données</h2>', unsafe_allow_html=True)
    
    # Sélection des variables pour l'analyse
    col1, col2 = st.columns(2)
    
    with col1:
        numerical_vars = filtered_df.select_dtypes(include=[np.number]).columns.tolist()
        x_axis = st.selectbox("Variable X:", numerical_vars, index=0)
    
    with col2:
        y_axis = st.selectbox("Variable Y:", numerical_vars, index=min(1, len(numerical_vars)-1))
    
    # Graphique interactif
    col3, col4 = st.columns(2)
    
    with col3:
        color_by = st.selectbox("Colorier par:", ['None'] + filtered_df.select_dtypes(include=['object']).columns.tolist())
    
    with col4:
        chart_type = st.selectbox("Type de Graphique:", ['Scatter', 'Line', 'Bar', 'Histogram'])
    
    # Génération du graphique
    st.markdown("### 📊 Graphique Interactif")
    
    try:
        if chart_type == 'Scatter':
            if color_by != 'None':
                fig = px.scatter(filtered_df, x=x_axis, y=y_axis, color=color_by,
                               hover_data=['product_category', 'season'],
                               title=f"{y_axis} vs {x_axis} par {color_by}")
            else:
                fig = px.scatter(filtered_df, x=x_axis, y=y_axis,
                               hover_data=['product_category', 'season'],
                               title=f"{y_axis} vs {x_axis}")
        
        elif chart_type == 'Line':
            # Agrégation pour les line charts
            agg_data = filtered_df.groupby(x_axis)[y_axis].mean().reset_index()
            fig = px.line(agg_data, x=x_axis, y=y_axis, title=f"Évolution de {y_axis} par {x_axis}")
        
        elif chart_type == 'Bar':
            agg_data = filtered_df.groupby(x_axis)[y_axis].mean().nlargest(20).reset_index()
            fig = px.bar(agg_data, x=x_axis, y=y_axis, title=f"{y_axis} par {x_axis} (Top 20)")
        
        elif chart_type == 'Histogram':
            fig = px.histogram(filtered_df, x=x_axis, title=f"Distribution de {x_axis}")
        
        st.plotly_chart(fig, use_container_width=True)
    
    except Exception as e:
        st.error(f"Erreur lors de la génération du graphique: {e}")
    
    # Analyse de corrélation
    st.markdown("### 🔗 Matrice de Corrélation")
    
    # Sélection des variables numériques pour la corrélation
    numeric_cols_for_corr = st.multiselect(
        "Sélectionnez les variables pour la corrélation:",
        numerical_vars,
        default=numerical_vars[:5] if len(numerical_vars) >= 5 else numerical_vars
    )
    
    if len(numeric_cols_for_corr) >= 2:
        corr_matrix = filtered_df[numeric_cols_for_corr].corr()
        
        fig = px.imshow(corr_matrix,
                       text_auto=True,
                       aspect="auto",
                       color_continuous_scale='RdBu_r',
                       title="Matrice de Corrélation")
        st.plotly_chart(fig, use_container_width=True)
    
    # Données brutes avec filtres
    st.markdown("### 📋 Données Brutes")
    
    # Filtres supplémentaires pour les données brutes
    col5, col6 = st.columns(2)
    
    with col5:
        show_columns = st.multiselect(
            "Colonnes à afficher:",
            filtered_df.columns.tolist(),
            default=filtered_df.columns.tolist()[:8]
        )
    
    with col6:
        n_rows = st.slider("Nombre de lignes à afficher:", 10, 100, 20)
    
    if show_columns:
        st.dataframe(filtered_df[show_columns].head(n_rows), use_container_width=True)
    
    # Téléchargement des données filtrées
    st.markdown("### 💾 Export des Données")
    
    csv = filtered_df.to_csv(index=False)
    st.download_button(
        label="📥 Télécharger les données filtrées (CSV)",
        data=csv,
        file_name=f"retail_data_filtered_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )

# Pied de page
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        📊 Tableau de Bord de Segmentation Retail - Développé avec Streamlit<br>
        🎯 Système de Recommandation Intelligent - © 2024
    </div>
    """, 
    unsafe_allow_html=True
)