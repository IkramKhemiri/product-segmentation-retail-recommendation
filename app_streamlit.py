import os
import math
import glob
import numpy as np
import pandas as pd
import streamlit as st
from typing import Dict
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
import plotly.graph_objects as go
import calendar

# ==== INTERPRETATION (LLM GRATUIT OPTIONNEL) =================================
# Fallback règle-based + tentative d'utilisation d'un modèle open-source gratuit local.
# Trois modes:
# 1) Heuristics (par défaut)
# 2) Transformers (si package transformers et poids HF téléchargés: mistral-7b-instruct)
# 3) Ollama (si installé: `ollama run mistral`)
# Choix via variable d'environnement: INTERP_MODE = heuristics | hf | ollama
INTERP_MODE = os.getenv("INTERP_MODE", "heuristics").lower()

@st.cache_resource(show_spinner=False)
def _load_hf_pipeline():
    if INTERP_MODE != "hf":
        return None
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
        model_name = os.getenv("INTERP_MODEL", "mistral-7b-instruct-v0.2")
        tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        mdl = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype="auto"
        )
        return pipeline("text-generation", model=mdl, tokenizer=tok)
    except Exception as e:
        st.warning(f"HF pipeline indisponible (fallback heuristics): {e}")
        return None

@st.cache_resource(show_spinner=False)
def _ollama_available():
    if INTERP_MODE != "ollama":
        return False
    try:
        import subprocess, json
        r = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=5)
        return r.returncode == 0 and "mistral" in r.stdout.lower()
    except Exception:
        return False

class ExplanationEngine:
    base_rules = {
        "treemap_promo": "Treemap des catégories par poids commercial et pression promotionnelle.",
        "season_bar": "Barres des ventes par saison pour planification des stocks.",
        "price_rating_scatter": "Dispersion prix vs qualité/rotation produit.",
        "segmentation_scatter": "Projection multivariée des produits selon attributs clés.",
        "promotion_decomp": "Décomposition pondérée du score promotion pour un produit individuel.",
        "top_products_bar": "Classement des produits leaders sur la métrique choisie.",
        "heat_client_prod": "Affinité entre groupes clients et catégories produit.",
        "dow_sales": "Performance par jour de semaine.",
        "cluster_value_clients": "Contribution de chaque cluster client à la valeur.",
        "cluster_scatter_clients": "Projection proxy clients colorée par cluster.",
        "cluster_value_products": "Valeur agrégée par cluster produit.",
        "cluster_scatter_products": "Projection produits prix vs rating/vitesse colorée par cluster.",
        "correlation_heatmap": "Carte des corrélations entre variables numériques.",
        "missing_bar": "Vue du pourcentage de valeurs manquantes par colonne.",
        "numeric_box": "Dispersion et outliers des variables numériques.",
        "numeric_hist": "Forme de distribution (symétrie, queues).",
        "categorical_dist": "Fréquences des catégories.",
        "cluster_comparison_table": "Comparaison algorithmes de clustering sur métriques qualité."
    }

    def summarize_numeric(self, df: pd.DataFrame):
        out = {}
        num = df.select_dtypes(include=[np.number])
        for c in num.columns:
            s = num[c].dropna()
            if s.empty: continue
            out[c] = {"mean": float(s.mean()), "std": float(s.std()), "skew": float(s.skew())}
        return out

    def find_cor_pairs(self, df: pd.DataFrame, thr=0.8):
        num = df.select_dtypes(include=[np.number])
        if num.shape[1] < 2: return []
        corr = num.corr().abs()
        pairs=[]
        for i,c1 in enumerate(corr.columns):
            for j,c2 in enumerate(corr.columns):
                if j<=i: continue
                v=corr.loc[c1,c2]
                if v>=thr: pairs.append((c1,c2,float(v)))
        return sorted(pairs,key=lambda x:-x[2])[:5]

    def cluster_profile(self, df, labels):
        if labels is None or len(labels)!=len(df): return ""
        df2 = df.copy()
        df2["_cl"]=labels
        num=df2.select_dtypes(include=[np.number]).drop(columns=["_cl"], errors="ignore")
        res=[]
        for cl,g in df2.groupby("_cl"):
            means=g[num.columns].mean().round(2).to_dict()
            top=list(means.items())[:4]
            res.append(f"{cl}: " + ", ".join([f"{k}={v}" for k,v in top]))
        return " | ".join(res[:6])

    def heuristic(self, chart_id, df, extra):
        base=self.base_rules.get(chart_id,"Visualisation.")
        parts=[base]
        if df is not None and not df.empty:
            stats=self.summarize_numeric(df)
            if chart_id=="correlation_heatmap":
                pairs=self.find_cor_pairs(df)
                if pairs:
                    parts.append("Corrélations fortes: " + "; ".join([f"{a}-{b}({v:.2f})" for a,b,v in pairs]))
            if chart_id.startswith("cluster_") and extra.get("labels") is not None:
                prof=self.cluster_profile(df, extra["labels"])
                if prof: parts.append("Profils: "+prof)
        if "silhouette" in extra:
            s=extra["silhouette"]
            qual="excellente" if s>=0.7 else "bonne" if s>=0.5 else "moyenne" if s>=0.3 else "faible"
            parts.append(f"Silhouette {s:.3f} ({qual}).")
        return " ".join(parts)

    def llm_generate(self, prompt):
        # HF pipeline
        if INTERP_MODE=="hf":
            pipe=_load_hf_pipeline()
            if pipe:
                txt=pipe(prompt, max_new_tokens=120, temperature=0.2, do_sample=False)[0]["generated_text"]
                return txt.split("###")[-1].strip()
        # Ollama
        if INTERP_MODE=="ollama" and _ollama_available():
            try:
                import subprocess, json
                cmd=["ollama","run","mistral",prompt]
                r=subprocess.run(cmd,capture_output=True,text=True,timeout=25)
                if r.returncode==0:
                    return r.stdout.strip()
            except Exception:
                pass
        return None

    def gen(self, chart_id, df=None, extra=None):
        df = df if isinstance(df, pd.DataFrame) else pd.DataFrame()
        extra = extra or {}
        heuristic_text = self.heuristic(chart_id, df, extra)
        if INTERP_MODE=="heuristics":
            return heuristic_text
        # Compose prompt
        prompt = (
            "Tu es un assistant data. Résume le graphique.\n"
            f"Type: {chart_id}\n"
            f"Heuristique: {heuristic_text}\n"
            f"Colonnes: {', '.join(df.columns.tolist()[:12])}\n"
            "Répond en une phrase concise en français sans listes."
        )
        llm = self.llm_generate(prompt)
        if llm and len(llm.split())>3:
            return llm
        return heuristic_text

engine = ExplanationEngine()
def explain(chart_id, df_ref=None, extra=None):
    st.caption(engine.gen(chart_id, df_ref, extra))

# ==== FIN INTERPRETATION =====================================================

# --------------------------------------------------
# CONFIG + THEME
# --------------------------------------------------
st.set_page_config(
    page_title="Retail Fashion Intelligence (One-Page)",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)
px.defaults.template = "plotly_white"

st.markdown("""
<style>
.main {background: linear-gradient(140deg,#f8fafc 0%,#eef3f9 55%,#e7edf5 100%);}
h1,h2,h3,h4 {font-family: 'Segoe UI',sans-serif;font-weight:600;}
.section-box{
  background:#ffffff;
  padding:18px 22px;
  border-radius:18px;
  margin-bottom:18px;
  border:1px solid #d9e2ec;
  box-shadow:0 2px 4px rgba(0,0,0,0.05);
}
.metric-card{
  padding:14px 18px;
  border-radius:14px;
  background:linear-gradient(135deg,#ffffff 0%,#f2f5fa 100%);
  border:1px solid #d8dee7;
  box-shadow:0 2px 4px rgba(0,0,0,0.06);
}
.badge{
  display:inline-block;
  padding:3px 9px;
  border-radius:12px;
  background:#2563eb;
  color:#fff;
  font-size:11px;
  font-weight:600;
  margin-left:6px;
}
.sep {height:2px;background:linear-gradient(90deg,#2563eb,#7e57c2,#ec407a);margin:28px 0;border-radius:2px;}
.filter-box{
  background:#ffffff;
  padding:12px 16px;
  border-radius:14px;
  border:1px solid #d9e2ec;
  margin-bottom:12px;
}
.stTextInput>div>div>input {border-radius:8px;}
.qabox{
  background:#fefefe;
  border:1px solid #d9e2ec;
  padding:12px 16px;
  border-radius:12px;
  margin-bottom:10px;
  font-size:13px;
}
.algocard{
  background:#ffffff;
  border:1px solid #d9e2ec;
  padding:10px 14px;
  border-radius:12px;
  font-size:12px;
  margin-bottom:8px;
}
</style>
""", unsafe_allow_html=True)

HERE = os.path.dirname(__file__)
DATA_PATH = os.path.join(HERE, "retail_data.csv")
CLIENT_PKLS = glob.glob(os.path.join(HERE, "clustering_results_clients_*.pkl"))
PRODUCT_PKLS = glob.glob(os.path.join(HERE, "produits_comparison_results.pkl"))
ASSOCIATION_RULES_CSV = os.path.join(HERE, "association_rules.csv")
CROSS_SELL_PAIRS_CSV = os.path.join(HERE, "cross_sell_pairs.csv")

# --------------------------------------------------
# HELPERS
# --------------------------------------------------
@st.cache_data(show_spinner=True)
def load_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return pd.read_csv(path)

def safe_num(s, default=0.0):
    return pd.to_numeric(s, errors="coerce").fillna(default)

def ensure_col(df: pd.DataFrame, col: str, default):
    if col not in df.columns:
        df[col] = default
    return df

@st.cache_data(show_spinner=True)
def build_products_view(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c,v in [
        ("product_category","Unknown"),("unit_price",0.0),("quantity",0.0),
        ("discount_applied",0.0),("product_stock",0.0),("product_return_rate",0.0),
        ("product_rating",np.nan),("season","All Seasons")
    ]:
        df = ensure_col(df,c,v)
    for c in ["unit_price","quantity","discount_applied","product_stock","product_return_rate","product_rating"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if "total_sales" not in df.columns:
        df["total_sales"] = df["quantity"].fillna(0)*df["unit_price"].fillna(0)
    else:
        df["total_sales"] = safe_num(df["total_sales"],0)
    agg_keys = ["product_id"] if "product_id" in df.columns else ["product_category","unit_price","season"]
    agg = df.groupby(agg_keys, dropna=False).agg(
        product_name=("product_name","first") if "product_name" in df.columns else ("product_category","first"),
        product_category=("product_category","first"),
        season=("season","first"),
        unit_price=("unit_price","mean"),
        quantity=("quantity","sum"),
        discount_applied=("discount_applied","mean"),
        product_stock=("product_stock","mean"),
        product_return_rate=("product_return_rate","mean"),
        product_rating=("product_rating","mean"),
        total_sales=("total_sales","sum")
    ).reset_index()
    agg["discount_applied"] = agg["discount_applied"].clip(0,1).fillna(0)
    agg["product_return_rate"] = agg["product_return_rate"].clip(0,1).fillna(0)
    med_rating = agg["product_rating"].median()
    agg["product_rating"] = agg["product_rating"].fillna(med_rating if not math.isnan(med_rating) else 3.0)
    qty = pd.to_numeric(agg["quantity"], errors="coerce").fillna(0)
    stock = pd.to_numeric(agg["product_stock"], errors="coerce")
    safe_stock = stock.replace(0,np.nan)
    vel = qty.divide(safe_stock).where(stock>0,0).replace([np.inf,-np.inf],0).fillna(0)
    agg["sales_velocity"] = vel
    agg["estimated_margin"] = agg["unit_price"]*(1-agg["discount_applied"])

    def minmax(s: pd.Series):
        s = s.replace([np.inf,-np.inf],np.nan).fillna(s.median() if not math.isnan(s.median()) else 0)
        mn,mx = float(s.min()), float(s.max())
        return pd.Series(0.5,index=s.index) if mx-mn<1e-12 else (s-mn)/(mx-mn)
    agg["sales_norm"] = minmax(agg["total_sales"])
    agg["stock_norm"] = minmax(agg["product_stock"])
    agg["velocity_norm"] = minmax(agg["sales_velocity"])
    agg["margin_norm"] = minmax(agg["estimated_margin"])
    agg["rating_norm"] = minmax(agg["product_rating"])
    agg["return_norm_inv"] = 1 - minmax(agg["product_return_rate"])

    def safe_qcut(x,q=3,labels=None,default="Medium"):
        try: return pd.qcut(x.rank(method="first"), q=q, labels=labels)
        except: return pd.Series([default]*len(x), index=x.index)
    agg["segment_sales"] = safe_qcut(agg["total_sales"],3,["Low-Sales","Medium-Sales","High-Sales"])
    agg["segment_velocity"] = safe_qcut(agg["sales_velocity"],3,["Slow","Medium","Fast"])
    agg["limited_collection"] = (agg["product_stock"]<=agg["product_stock"].quantile(0.25)) & (agg["unit_price"]>=agg["unit_price"].quantile(0.75))
    agg["best_seller"] = (agg["segment_sales"].astype(str)=="High-Sales")
    agg["slow_mover"] = (agg["segment_velocity"].astype(str)=="Slow")
    return agg

def compute_scores(df_prod,w_over,w_slow,w_margin,w_quality,margin_factor):
    d = df_prod.copy()
    d["estimated_margin_adj"] = d["estimated_margin"]*margin_factor
    d["margin_norm"] = MinMaxScaler().fit_transform(d[["estimated_margin_adj"]])
    overstock_factor = d["stock_norm"]*(1-d["velocity_norm"])
    quality_factor = d["rating_norm"]*d["return_norm_inv"]
    d["promotion_score"] = (
        w_over*overstock_factor +
        w_slow*(1-d["velocity_norm"]) +
        w_margin*d["margin_norm"] +
        w_quality*d["return_norm_inv"]
    )
    d["upsell_score"] = (
        0.35*d["velocity_norm"] + 0.30*d["margin_norm"] +
        0.20*quality_factor + 0.15*d["return_norm_inv"]
    )
    bins = [-np.inf,0.4,0.7,np.inf]
    labels = ["Promotion Faible","Promotion Modérée","Promotion Forte"]
    try:
        d["promotion_level"] = pd.cut(d["promotion_score"], bins=bins, labels=labels)
    except:
        d["promotion_level"] = "Promotion Modérée"
    return d

def kpis(df_prod: pd.DataFrame)->Dict[str,float]:
    return {
        "revenue": float(df_prod.get("total_sales",pd.Series(dtype=float)).sum()),
        "units": float(df_prod.get("quantity",pd.Series(dtype=float)).sum()) if "quantity" in df_prod.columns else np.nan,
        "avg_price": float(df_prod.get("unit_price",pd.Series(dtype=float)).mean()) if "unit_price" in df_prod.columns else np.nan,
        "avg_rating": float(df_prod.get("product_rating",pd.Series(dtype=float)).mean()) if "product_rating" in df_prod.columns else np.nan,
        "stock": float(df_prod.get("product_stock",pd.Series(dtype=float)).sum()) if "product_stock" in df_prod.columns else np.nan
    }

@st.cache_data(show_spinner=True)
def load_pickles():
    import pickle
    client_obj = None
    product_obj = None
    if CLIENT_PKLS:
        try:
            with open(sorted(CLIENT_PKLS)[-1],"rb") as f:
                client_obj = pickle.load(f)
        except: client_obj=None
    if PRODUCT_PKLS:
        try:
            with open(sorted(PRODUCT_PKLS)[-1],"rb") as f:
                product_obj = pickle.load(f)
        except: product_obj=None
    return client_obj, product_obj

@st.cache_data(show_spinner=True)
def load_optional_csvs():
    rules = pd.read_csv(ASSOCIATION_RULES_CSV) if os.path.exists(ASSOCIATION_RULES_CSV) else None
    pairs = pd.read_csv(CROSS_SELL_PAIRS_CSV) if os.path.exists(CROSS_SELL_PAIRS_CSV) else None
    return rules, pairs

# --------------------------------------------------
# PREPARE TRANSACTIONAL VIEW (for seller insights)
# --------------------------------------------------
@st.cache_data(show_spinner=True)
def build_transactions_view(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    # Ensure base cols
    for c, v in [
        ("quantity", 0.0),
        ("unit_price", 0.0),
        ("product_category", "Unknown"),
        ("product_name", "Unknown"),
        ("season", "Unknown"),
        ("customer_id", "NA"),
    ]:
        if c not in d.columns:
            d[c] = v
    # Total sales
    if "total_sales" not in d.columns:
        d["total_sales"] = pd.to_numeric(d["quantity"], errors="coerce").fillna(0) * pd.to_numeric(d["unit_price"], errors="coerce").fillna(0)

    # Dates
    if "transaction_date" in d.columns:
        d["transaction_date"] = pd.to_datetime(d["transaction_date"], errors="coerce")
    else:
        d["transaction_date"] = pd.NaT

    # Month, Year, Day of week (robust fallbacks)
    d["year"] = d["transaction_date"].dt.year
    if "month_of_year" in d.columns:
        d["month_num"] = pd.to_numeric(d["month_of_year"], errors="coerce").clip(1, 12).fillna(1).astype(int)
    else:
        d["month_num"] = d["transaction_date"].dt.month.fillna(1).astype(int)
    d["month_name"] = d["month_num"].map(lambda m: ["Jan", "Fév", "Mar", "Avr", "Mai", "Juin", "Juil", "Aoû", "Sep", "Oct", "Nov", "Déc"][m-1] if 1 <= m <= 12 else "NA")
    if "day_of_week" in d.columns:
        # 0-6 or names; normalize to names
        dow = d["day_of_week"]
        if pd.api.types.is_numeric_dtype(dow):
            d["day_name"] = dow.map({0:"Lun",1:"Mar",2:"Mer",3:"Jeu",4:"Ven",5:"Sam",6:"Dim"})
        else:
            d["day_name"] = dow.astype(str).str[:3].str.title()
    else:
        d["day_name"] = d["transaction_date"].dt.dayofweek.map({0:"Lun",1:"Mar",2:"Mer",3:"Jeu",4:"Ven",5:"Sam",6:"Dim"})

    # Age groups (if age exists)
    if "age" in d.columns:
        d["age_group"] = pd.cut(pd.to_numeric(d["age"], errors="coerce"),
                                bins=[-np.inf, 24, 34, 44, 54, 64, np.inf],
                                labels=["<25","25-34","35-44","45-54","55-64","65+"])
    else:
        d["age_group"] = "NA"

    # Simple client monetary segments (Low/Medium/High) on total_sales per customer
    try:
        cust_sales = d.groupby("customer_id", dropna=False)["total_sales"].sum()
        seg = pd.qcut(cust_sales.rank(method="first"), q=3, labels=["Low-Value","Medium-Value","High-Value"])
        seg = seg.reindex(cust_sales.index)
        d = d.merge(seg.rename("customer_segment"), left_on="customer_id", right_index=True, how="left")
    except Exception:
        d["customer_segment"] = "Medium-Value"

    return d

# --------------------------------------------------
# DATA LOAD
# --------------------------------------------------
try:
    df_raw = load_csv(DATA_PATH)
except Exception as e:
    st.error(f"Chargement impossible: {e}")
    st.stop()

df_prod = build_products_view(df_raw)
df_tx = build_transactions_view(df_raw)  # NEW: transactional view for seller insights
client_pkg, product_pkg = load_pickles()
rules_df, pairs_df = load_optional_csvs()

# --------------------------------------------------
# SIDEBAR FILTERS
# --------------------------------------------------
st.sidebar.markdown("### 🎛️ Filtres")
with st.sidebar.form("filters_form"):
    # Catégories
    if "product_category" in df_prod.columns:
        cat_all = sorted(df_prod["product_category"].astype(str).dropna().unique())
        st.markdown("**Catégories**")
        all_cat_flag = st.checkbox("Toutes les catégories", value=True, key="all_cat_flag")
        if all_cat_flag:
            sel_cats = cat_all
        else:
            sel_cats = []
            cols_cat = st.columns(2)
            for i, cval in enumerate(cat_all):
                if cols_cat[i % 2].checkbox(cval, key=f"cat_{cval}"):
                    sel_cats.append(cval)
            if not sel_cats:  # fallback si aucune cochée
                sel_cats = cat_all
    else:
        sel_cats = []

    # Saisons
    if "season" in df_prod.columns:
        season_all = sorted(df_prod["season"].astype(str).dropna().unique())
        st.markdown("**Saisons**")
        all_season_flag = st.checkbox("Toutes les saisons", value=True, key="all_season_flag")
        if all_season_flag:
            season_choice_list = season_all
        else:
            season_choice_list = []
            cols_sea = st.columns(2)
            for i, sval in enumerate(season_all):
                if cols_sea[i % 2].checkbox(sval, key=f"season_{sval}"):
                    season_choice_list.append(sval)
            if not season_choice_list:
                season_choice_list = season_all
    else:
        season_choice_list = []

    def safe_range(s: pd.Series, pad=1.0):
        s = pd.to_numeric(s, errors="coerce").replace([np.inf,-np.inf],np.nan).dropna()
        if s.empty: return 0.0,1.0
        mn,mx=float(s.min()),float(s.max())
        return (mn,mn+pad) if mx-mn<1e-9 else (mn,mx)

    pmin,pmax = safe_range(df_prod["unit_price"]) if "unit_price" in df_prod.columns else (0,1)
    price_sel = st.slider("Prix", float(round(pmin,2)), float(round(max(pmax,pmin+1),2)),
                          (float(round(pmin,2)), float(round(max(pmax,pmin+1),2))))
    smin,smax = safe_range(df_prod["product_stock"]) if "product_stock" in df_prod.columns else (0,1)
    stock_sel = st.slider("Stock", float(smin), float(max(smax,smin+1)),
                          (float(smin), float(max(smax,smin+1))))
    if "product_rating" in df_prod.columns:
        rmin,rmax = safe_range(df_prod["product_rating"])
        rating_sel = st.slider("Rating", float(min(0.0,rmin)), float(max(5.0,rmax)),
                               (float(min(0.0,rmin)), float(max(5.0,rmax))))
    else:
        rating_sel=None
    if "discount_applied" in df_prod.columns:
        dmin,dmax = safe_range(df_prod["discount_applied"])
        discount_sel = st.slider("Remise", float(dmin), float(dmax), (float(dmin), float(dmax)))
    else:
        discount_sel=None
    if "product_return_rate" in df_prod.columns:
        retmin,retmax = safe_range(df_prod["product_return_rate"])
        return_sel = st.slider("Taux retour", float(retmin), float(retmax), (float(retmin), float(retmax)))
    else:
        return_sel=None

    st.markdown("#### Paramètres Promotion")
    w_over = st.slider("Poids Sur-stock",0.0,1.0,0.30,0.05)
    w_slow = st.slider("Poids Ventes lentes",0.0,1.0,0.25,0.05)
    w_margin = st.slider("Poids Marge",0.0,1.0,0.25,0.05)
    w_qual = st.slider("Poids Qualité/Retours",0.0,1.0,0.20,0.05)
    sumw = max(w_over+w_slow+w_margin+w_qual,1e-9)
    w_over,w_slow,w_margin,w_qual = [w/sumw for w in (w_over,w_slow,w_margin,w_qual)]
    margin_factor = st.slider("Facteur marge",0.1,2.0,0.6,0.05)
    text_search = st.text_input("Recherche texte (nom/id/catégorie)")
    submitted = st.form_submit_button("Appliquer")

# Application des filtres (mise à jour saison multiple)
flt = df_prod.copy()
if sel_cats:
    flt = flt[flt["product_category"].isin(sel_cats)]
if season_choice_list and "season" in flt.columns:
    flt = flt[flt["season"].isin(season_choice_list)]
if "unit_price" in flt.columns:
    flt = flt[flt["unit_price"].between(price_sel[0], price_sel[1])]
if "product_stock" in flt.columns:
    flt = flt[flt["product_stock"].between(stock_sel[0], stock_sel[1])]
if rating_sel and "product_rating" in flt.columns:
    flt = flt[flt["product_rating"].between(rating_sel[0], rating_sel[1])]
if discount_sel and "discount_applied" in flt.columns:
    flt = flt[flt["discount_applied"].between(discount_sel[0], discount_sel[1])]
if return_sel and "product_return_rate" in flt.columns:
    flt = flt[flt["product_return_rate"].between(return_sel[0], return_sel[1])]

scored = compute_scores(flt,w_over,w_slow,w_margin,w_qual,margin_factor)
if text_search.strip():
    mask = pd.Series(False,index=scored.index)
    for c in ["product_name","product_id","product_category"]:
        if c in scored.columns:
            mask |= scored[c].astype(str).str.contains(text_search, case=False, na=False)
    scored = scored[mask]

# --------------------------------------------------
# HEADER (context + guide) — REPLACE CE BLOC
st.markdown("""
<div style="background:linear-gradient(145deg,#0d47a1 0%,#2563eb 45%,#7e57c2 75%,#ec407a 100%);
           padding:36px 42px;border-radius:26px;color:#fff;position:relative;overflow:hidden;">
  <div style="position:absolute;inset:0;
      background:radial-gradient(circle at 78% 22%,rgba(255,255,255,0.18),transparent 60%);
      pointer-events:none;"></div>
  <h1 style="margin:0;font-size:40px;letter-spacing:.5px;">Retail Fashion Intelligence</h1>
  <p style="margin:10px 0 18px;font-size:15px;line-height:1.5;max-width:880px;">
    Pilotage unifié des décisions retail: segmentation produits & clients, scoring promotion, 
    recommandations, cross-sell, analyse saisonnalité, comparaison d’algorithmes, et insights opérationnels.
  </p>
  <div style="display:flex;flex-wrap:wrap;gap:10px;font-size:12px;">
    <div class="badge" style="background:#1e293b;">Segmentation</div>
    <div class="badge" style="background:#4338ca;">Promotion</div>
    <div class="badge" style="background:#be185d;">Recommandations</div>
    <div class="badge" style="background:#0f766e;">Saisonnalité</div>
    <div class="badge" style="background:#7c2d12;">Clustering</div>
    <div class="badge" style="background:#047857;">Comparaison Algorithmes</div>
  </div>
  <hr style="margin:26px 0;border:0;height:2px;
      background:linear-gradient(90deg,#ffffff55,#ffffffdd,#ffffff55);"/>
  <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:14px;font-size:12px;">
    <div style="background:#ffffff12;padding:10px 14px;border-radius:12px;">1. Filtres dynamiques</div>
    <div style="background:#ffffff12;padding:10px 14px;border-radius:12px;">2. Segments clés</div>
    <div style="background:#ffffff12;padding:10px 14px;border-radius:12px;">3. Scores promotion</div>
    <div style="background:#ffffff12;padding:10px 14px;border-radius:12px;">4. Recommandations</div>
    <div style="background:#ffffff12;padding:10px 14px;border-radius:12px;">5. Insights vendeur</div>
    <div style="background:#ffffff12;padding:10px 14px;border-radius:12px;">6. Clustering & Comparaison</div>
  </div>
</div>
""", unsafe_allow_html=True)

# KPIs
k = kpis(scored)
kcols = st.columns(5)
for col,label,val in [
    (kcols[0],"Produits filtrés",f"{len(scored):,}"),
    (kcols[1],"Chiffre d'affaires",f"{k['revenue']:,.0f}"),
    (kcols[2],"Unités vendues",f"{0 if np.isnan(k['units']) else k['units']:,.0f}"),
    (kcols[3],"Prix moyen",f"{0 if np.isnan(k['avg_price']) else k['avg_price']:,.2f}"),
    (kcols[4],"Stock total",f"{0 if np.isnan(k['stock']) else k['stock']:,.0f}")
]:
    col.markdown(f"<div class='metric-card'><b>{label}</b><br><span style='font-size:22px;color:#2563eb;font-weight:600'>{val}</span></div>", unsafe_allow_html=True)

st.markdown("<div class='sep'></div>", unsafe_allow_html=True)

# --------------------------------------------------
# SECTION OVERVIEW
# --------------------------------------------------
st.markdown("## 📊 Vue Générale")
c1,c2 = st.columns([1.4,1])
if "product_category" in scored.columns and len(scored)>0:
    size_col = "quantity" if "quantity" in scored.columns and scored["quantity"].sum()>0 else "total_sales"
    fig_t = px.treemap(
        scored,
        path=["product_category","promotion_level"] if "promotion_level" in scored.columns else ["product_category"],
        values=size_col,
        color="promotion_score",
        color_continuous_scale="RdYlGn_r",
        title="Structure Catégorie & Pression Promotion"
    )
    c1.plotly_chart(fig_t, use_container_width=True); explain("treemap_promo", scored)

if "season" in scored.columns and scored["season"].nunique()>1:
    season_sales = scored.groupby("season")["total_sales"].sum().reset_index().sort_values("total_sales",ascending=False)
    fig_season = px.bar(season_sales, x="season", y="total_sales", color="season", text_auto=True,
                        title="Ventes par Saison", color_discrete_sequence=px.colors.qualitative.Set3)
    c2.plotly_chart(fig_season, use_container_width=True); explain("season_bar", season_sales)
else:
    if "unit_price" in scored.columns:
        fig_price = px.histogram(scored,x="unit_price",nbins=40,title="Distribution Prix",color_discrete_sequence=["#2563eb"])
        c2.plotly_chart(fig_price, use_container_width=True)

if "unit_price" in scored.columns:
    x_axis = "unit_price"
    y_axis = "product_rating" if "product_rating" in scored.columns else "velocity_norm"
    fig_sc = px.scatter(
        scored.head(6000), x=x_axis,y=y_axis,
        size="quantity" if "quantity" in scored.columns else None,
        color="segment_sales",
        hover_data=[c for c in ["product_id","product_name","product_category","season","total_sales","sales_velocity"] if c in scored.columns],
        title="Carte Produits — Prix vs Rating/Vitesse",
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    st.plotly_chart(fig_sc, use_container_width=True); explain("price_rating_scatter", scored)

st.markdown("<div class='sep'></div>", unsafe_allow_html=True)

# --------------------------------------------------
# SECTION SEGMENTATION
# --------------------------------------------------
st.markdown("## 🧩 Segmentation Produits")
seg_cols = st.columns([1,1])
color_by = seg_cols[0].selectbox("Couleur par", [c for c in ["segment_sales","segment_velocity","promotion_level","product_category","season"] if c in scored.columns])
size_by = seg_cols[0].selectbox("Taille par", [c for c in ["quantity","product_stock","total_sales"] if c in scored.columns])
x_axis = "unit_price" if "unit_price" in scored.columns else "margin_norm"
y_axis = "product_rating" if "product_rating" in scored.columns else "velocity_norm"
fig_seg = px.scatter(
    scored.head(6000), x=x_axis, y=y_axis, color=color_by,
    size=size_by if size_by else None,
    hover_data=[c for c in ["product_id","product_name","product_category","season","total_sales","sales_velocity"] if c in scored.columns],
    title=f"{x_axis} vs {y_axis}",
    color_discrete_sequence=px.colors.qualitative.Vivid
)
seg_cols[0].plotly_chart(fig_seg, use_container_width=True); explain("segmentation_scatter", scored)

seg_metric_col = seg_cols[1]
seg_count_df = pd.DataFrame({
    "Segment":["Best Sellers","Ventes lentes","Collections limitées","Promo Forte"],
    "Count":[
        int(scored.get("best_seller",pd.Series(dtype=bool)).sum()) if "best_seller" in scored.columns else 0,
        int(scored.get("slow_mover",pd.Series(dtype=bool)).sum()) if "slow_mover" in scored.columns else 0,
        int(scored.get("limited_collection",pd.Series(dtype=bool)).sum()) if "limited_collection" in scored.columns else 0,
        int((scored["promotion_level"]=="Promotion Forte").sum()) if "promotion_level" in scored.columns else 0
    ]
})
fig_seg_bar = px.bar(seg_count_df,x="Segment",y="Count",text_auto=True,title="Compteurs Segments Clés",
                     color="Segment",color_discrete_sequence=px.colors.qualitative.Pastel)
seg_metric_col.plotly_chart(fig_seg_bar, use_container_width=True); explain("top_products_bar", seg_count_df)

st.markdown("<div class='sep'></div>", unsafe_allow_html=True)

# --------------------------------------------------
# SECTION PROMOTION
# --------------------------------------------------
st.markdown("## 🎯 Promotion & Décomposition Score")
top_promo = scored.sort_values("promotion_score", ascending=False).head(60)
show_cols = [c for c in ["product_id","product_name","product_category","season","unit_price","product_stock","sales_velocity","product_rating","product_return_rate","promotion_score","promotion_level"] if c in top_promo.columns]
st.dataframe(top_promo[show_cols], use_container_width=True, height=380)
if len(scored)>0:
    idx = st.number_input("Index produit (filtré)",0,len(scored)-1,0)
    row = scored.iloc[int(idx)]
    overstock_factor = float(row.get("stock_norm",0)*(1-row.get("velocity_norm",0)))
    comps = {
        "Sur-stock": w_over*overstock_factor,
        "Ventes lentes": w_slow*(1-row.get("velocity_norm",0)),
        "Marge (norm)": w_margin*row.get("margin_norm",0),
        "Retours faibles": w_qual*row.get("return_norm_inv",0)
    }
    fig_comp = go.Figure(data=[go.Bar(x=list(comps.keys()),y=list(comps.values()),
                                      marker_color=["#D62728","#FF7F0E","#1F77B4","#2CA02C"])])
    fig_comp.update_layout(title=f"Décomposition Score Promotion — {row.get('product_name','')}",yaxis_title="Contribution pondérée")
    st.plotly_chart(fig_comp,use_container_width=True); explain("promotion_decomp", pd.DataFrame([comps]))

st.markdown("<div class='sep'></div>", unsafe_allow_html=True)

# --------------------------------------------------
# SECTION RECOMMANDATIONS
# --------------------------------------------------
st.markdown("## 🚀 Recommandations Produits")
colR1,colR2,colR3 = st.columns(3)
top_sellers = scored.sort_values("total_sales",ascending=False).head(25) if "total_sales" in scored.columns else scored.head(25)
top_upsell = scored.sort_values("upsell_score",ascending=False).head(25) if "upsell_score" in scored.columns else scored.head(25)
top_promo_small = scored.sort_values("promotion_score",ascending=False).head(25)
cols_sell = [c for c in ["product_id","product_name","product_category","season","total_sales","quantity","unit_price","product_rating"] if c in top_sellers.columns]
cols_promo = [c for c in ["product_id","product_name","product_category","season","product_stock","sales_velocity","estimated_margin_adj","promotion_score","promotion_level"] if c in top_promo_small.columns]
cols_up = [c for c in ["product_id","product_name","product_category","season","upsell_score","unit_price","product_rating","product_return_rate"] if c in top_upsell.columns]
colR1.markdown("#### Meilleures ventes")
colR1.dataframe(top_sellers[cols_sell],use_container_width=True,height=340)
colR2.markdown("#### Candidats promotion")
colR2.dataframe(top_promo_small[cols_promo],use_container_width=True,height=340)
colR3.markdown("#### Candidats upsell")
colR3.dataframe(top_upsell[cols_up],use_container_width=True,height=340)

st.markdown("<div class='sep'></div>", unsafe_allow_html=True)

# --------------------------------------------------
# SECTION INSIGHTS VENDEUR (questions métier prêtes à l’emploi)
# --------------------------------------------------
st.markdown("## 🧠 Insights Vendeur (Questions fréquentes)")
with st.expander("Comment lire cette section ?", expanded=True):
    st.markdown("- Voir en un coup d’œil: produit star par saison, meilleures catégories, pics temporels (mois/jour), et profils clients par catégorie.")
    st.markdown("- Chaque visuel inclut une courte explication sous le graphe.")

# Controls for this section
i1, i2, i3, i4 = st.columns([1,1,1,1])
attr_choices = [c for c in ["gender","loyalty_program","income_bracket","region","age_group","customer_segment"] if c in df_tx.columns]
prod_dim = i1.selectbox("Dimension client", attr_choices if attr_choices else ["customer_segment"])
season_filter = i2.selectbox("Saison (insights)", ["Toutes"] + sorted(df_tx["season"].dropna().astype(str).unique().tolist()) if "season" in df_tx.columns else ["Toutes"])
year_list = sorted([int(y) for y in df_tx["year"].dropna().unique()]) if "year" in df_tx.columns else []
year_filter = i3.selectbox("Année", ["Toutes"] + year_list if year_list else ["Toutes"])
metric_sel = i4.selectbox("Métrique", ["total_sales","quantity"])

def apply_tx_filters(d: pd.DataFrame) -> pd.DataFrame:
    x = d.copy()
    if season_filter != "Toutes" and "season" in x.columns:
        x = x[x["season"].astype(str) == str(season_filter)]
    if year_filter != "Toutes" and "year" in x.columns:
        x = x[x["year"] == year_filter]
    return x

txv = apply_tx_filters(df_tx)

# 1) Meilleur produit par saison
st.markdown("### 🏆 Meilleur produit par saison")
if "season" in txv.columns:
    grp = txv.groupby(["season","product_name"], dropna=False)[metric_sel].sum().reset_index()
    best = grp.loc[grp.groupby("season")[metric_sel].idxmax()].sort_values("season")
    # FIX: utiliser le DataFrame renommé aussi dans la figure
    best_display = best.rename(columns={metric_sel: "valeur"})
    st.dataframe(best_display, use_container_width=True, height=240)
    fig_best = px.bar(
        best_display, 
        x="season", y="valeur", color="product_name", text_auto=True,
        title=f"Top produit par saison ({metric_sel})",
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    st.plotly_chart(fig_best, use_container_width=True); explain("top_products_bar", best_display)
    st.caption("Pourquoi: identifie le produit “star” de chaque saison pour ajuster l’assortiment et la mise en avant.")
else:
    st.info("Colonne 'season' absente.")

# 2) Produit le plus acheté (global ou filtré)
st.markdown("### 🥇 Produit le plus acheté")
top_prod = txv.groupby("product_name", dropna=False)[metric_sel].sum().reset_index().sort_values(metric_sel, ascending=False).head(15)
if not top_prod.empty:
    fig_tp = px.bar(top_prod, x="product_name", y=metric_sel, color="product_name", title=f"Top produits ({metric_sel})",
                    color_discrete_sequence=px.colors.qualitative.Pastel)
    fig_tp.update_xaxes(tickangle=45)
    st.plotly_chart(fig_tp, use_container_width=True); explain("top_products_bar", top_prod)
    st.caption("Pourquoi: met en évidence les références locomotives à sécuriser en stock et à promouvoir.")
else:
    st.info("Données insuffisantes pour ce visuel.")

# 3) Catégories de clients qui achètent le plus (par catégorie produit)
st.markdown("### 👥 Profil clients par catégorie produit")
if prod_dim in txv.columns:
    grp2 = txv.groupby([prod_dim,"product_category"], dropna=False)["customer_id"].nunique().reset_index().rename(columns={"customer_id":"clients_uniques"})
    if not grp2.empty:
        fig_heat = px.density_heatmap(grp2, x="product_category", y=prod_dim, z="clients_uniques", color_continuous_scale="Blues",
                                      title="Clients uniques par catégorie client × catégorie produit")
        st.plotly_chart(fig_heat, use_container_width=True); explain("heat_client_prod", grp2)
        st.caption("Pourquoi: relie groupes de clients aux catégories achetées pour cibler les actions marketing.")
    else:
        st.info("Aucune combinaison trouvée.")
else:
    st.info("Dimension client non disponible.")

# 4) Quelle catégorie client achète le plus/moins (global)
st.markdown("### 🔝🔻 Catégories clients: qui achète le plus/moins")
if prod_dim in txv.columns:
    grp3 = txv.groupby(prod_dim, dropna=False)[metric_sel].sum().reset_index().sort_values(metric_sel, ascending=False)
    cA, cB = st.columns(2)
    cA.plotly_chart(px.bar(grp3.head(10), x=prod_dim, y=metric_sel, title=f"Top {prod_dim} ({metric_sel})",
                           color_discrete_sequence=["#2563eb"]), use_container_width=True)
    cB.plotly_chart(px.bar(grp3.tail(10).sort_values(metric_sel), x=prod_dim, y=metric_sel, title=f"Bottom {prod_dim} ({metric_sel})",
                           color_discrete_sequence=["#ef4444"]), use_container_width=True)
    st.caption("Pourquoi: identifie les segments forts/faibles pour prioriser prospection et fidélisation.")

# 5) Saison qui vend le plus + Evolution mensuelle
st.markdown("### 🌤️ Saisons et tendance dans le temps")
if "season" in txv.columns:
    seas = txv.groupby("season", dropna=False)[metric_sel].sum().reset_index().sort_values(metric_sel, ascending=False)
    c1, c2 = st.columns([1,1.2])
    c1.plotly_chart(px.bar(seas, x="season", y=metric_sel, color="season", text_auto=True,
                           title=f"Ventes par saison ({metric_sel})",
                           color_discrete_sequence=px.colors.qualitative.Set3), use_container_width=True)
if "month_num" in txv.columns:
    mth = txv.groupby(["year","month_num","month_name"], dropna=False)[metric_sel].sum().reset_index().sort_values(["year","month_num"])
    mth["periode"] = mth["year"].astype(str) + "-" + mth["month_num"].astype(int).astype(str).str.zfill(2)
    c2.plotly_chart(px.line(mth, x="periode", y=metric_sel, markers=True, color_discrete_sequence=["#7e57c2"],
                            title=f"Evolution mensuelle ({metric_sel})"), use_container_width=True)
st.caption("Pourquoi: planifie stocks et promotions selon la saisonnalité et la tendance mensuelle.")



# 7) Types de produits qui se vendent le plus
st.markdown("### 🧺 Catégories produits les plus performantes")
cat_perf = txv.groupby("product_category", dropna=False)[metric_sel].sum().reset_index().sort_values(metric_sel, ascending=False).head(20)
if not cat_perf.empty:
    fig_cat = px.bar(cat_perf, x="product_category", y=metric_sel, title=f"Top catégories ({metric_sel})",
                     color="product_category", color_discrete_sequence=px.colors.qualitative.Set1)
    fig_cat.update_xaxes(tickangle=45)
    st.plotly_chart(fig_cat, use_container_width=True); explain("top_products_bar", cat_perf)
    st.caption("Pourquoi: priorise les familles à forte contribution et détecte les sous-performances.")
else:
    st.info("Données insuffisantes pour ce visuel.")

# (Helper pour extraire tableaux clustering depuis pickles s'il n'existe pas déjà)
def extract_algo_table(pkg, is_product=False):
    if not pkg or not isinstance(pkg, dict): 
        return pd.DataFrame()
    all_res = pkg.get("all_results") or pkg.get("all_results_products") or pkg.get("all_results")
    if not isinstance(all_res, dict):
        all_res = pkg.get("all_results", {})
    rows=[]
    for name,res in all_res.items():
        if not isinstance(res, dict): 
            continue
        rows.append({
            "Algorithme": name,
            "Silhouette": res.get("silhouette", np.nan),
            "Davies-Bouldin": res.get("davies_bouldin", np.nan),
            "Clusters": res.get("n_clusters", "N/A"),
            "Bruit (%)": f"{res.get('noise_ratio',0)*100:.1f}%",
            "Sample": len(res.get("labels", []))
        })
    df = pd.DataFrame(rows)
    if df.empty: return df
    # Score composite pour tri
    df["Score Composite"] = (
        df["Silhouette"].fillna(0)*2 +
        (1/(1+df["Davies-Bouldin"].fillna(1)))*1.5 +
        (1 - df["Bruit (%)"].str.replace('%','').astype(float)/100)*0.5
    )
    return df.sort_values("Score Composite", ascending=False)

# --------------------------------------------------
# SECTION COMPARAISON CLUSTERING (NOUVELLE)
# --------------------------------------------------
st.markdown("<div class='sep'></div>", unsafe_allow_html=True)
st.markdown("## 🔬 Comparaison Algorithmes Clustering")

tabC, tabP = st.tabs(["Clients", "Produits"])

with tabC:
    client_table = extract_algo_table(client_pkg, is_product=False)
    if client_table.empty:
        st.info("Aucun résultat clustering clients chargé (pickle manquant).")
    else:
        st.markdown("### Tableau comparatif (Clients)")
        st.dataframe(client_table, use_container_width=True, height=300)
        sel_algo_client = st.selectbox("Algorithme à visualiser (clients)", client_table["Algorithme"].tolist())
        sel_res = client_pkg["all_results"].get(sel_algo_client, {})
        # Mapping clusters sur transactions (approximation)
        labels = sel_res.get("labels")
        if labels is not None:
            # Ajuster longueur: assigner au premier n clients uniques
            cust_ids = df_tx["customer_id"].dropna().unique()
            assign_len = min(len(labels), len(cust_ids))
            map_df = pd.DataFrame({"customer_id": cust_ids[:assign_len], "cluster": labels[:assign_len]})
            merged = df_tx.merge(map_df, on="customer_id", how="left")
            merged["cluster"] = merged["cluster"].fillna(-1)
            agg = merged.groupby("cluster")["total_sales"].sum().reset_index()
            figc1 = px.bar(agg, x="cluster", y="total_sales", title=f"Distribution valeur par cluster ({sel_algo_client})",
                           color="cluster", color_discrete_sequence=px.colors.qualitative.Set3)
            st.plotly_chart(figc1, use_container_width=True)
            # Scatter proxy (days_since_last_purchase vs membership_years)
            sc_source = merged.dropna(subset=["days_since_last_purchase","membership_years"]).head(12000)
            figc2 = px.scatter(sc_source, x="days_since_last_purchase", y="membership_years",
                               color="cluster", title=f"Projection proxy clients ({sel_algo_client})",
                               color_discrete_sequence=px.colors.qualitative.Set2,
                               hover_data=["customer_id","customer_segment"] if "customer_segment" in sc_source.columns else None)
            st.plotly_chart(figc2, use_container_width=True)
        else:
            st.info("Labels absents pour cet algorithme clients.")

with tabP:
    product_table = extract_algo_table(product_pkg, is_product=True)
    if product_table.empty:
        st.info("Aucun résultat clustering produits chargé (pickle manquant).")
    else:
        st.markdown("### Tableau comparatif (Produits)")
        st.dataframe(product_table, use_container_width=True, height=300)
        sel_algo_prod = st.selectbox("Algorithme à visualiser (produits)", product_table["Algorithme"].tolist())
        sel_res_p = product_pkg["all_results"].get(sel_algo_prod, {})
        plabels = sel_res_p.get("labels")
        if plabels is not None:
            base_df = scored.copy()
            assign_len = min(len(plabels), len(base_df))
            base_df = base_df.head(assign_len).copy()
            base_df["cluster"] = plabels[:assign_len]
            aggP = base_df.groupby("cluster")["total_sales"].sum().reset_index()
            figp1 = px.bar(aggP, x="cluster", y="total_sales", title=f"Valeur totale par cluster ({sel_algo_prod})",
                           color="cluster", color_discrete_sequence=px.colors.qualitative.Set1)
            st.plotly_chart(figp1, use_container_width=True)
            # Scatter produits (prix vs rating/vitesse)
            y_axis = "product_rating" if "product_rating" in base_df.columns else "velocity_norm"
            figp2 = px.scatter(base_df.head(12000), x="unit_price", y=y_axis, color="cluster",
                               size="quantity" if "quantity" in base_df.columns else None,
                               title=f"Projection produits ({sel_algo_prod})",
                               hover_data=[c for c in ["product_id","product_name","product_category","total_sales"] if c in base_df.columns],
                               color_discrete_sequence=px.colors.qualitative.Dark24)
            st.plotly_chart(figp2, use_container_width=True)
        else:
            st.info("Labels absents pour cet algorithme produits.")