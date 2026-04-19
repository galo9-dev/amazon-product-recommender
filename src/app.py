import streamlit as st
import pandas as pd
from modelo_content import cargar_datos, construir_modelo, recomendar

# ── Configuración ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Amazon Product Recommender",
    page_icon="🛒",
    layout="wide"
)

# ── Carga de datos ────────────────────────────────────────────────────────────
@st.cache_data
def inicializar():
    df = cargar_datos()
    similitud = construir_modelo(df)
    return df, similitud

df, similitud = inicializar()

# ── Interfaz ──────────────────────────────────────────────────────────────────
st.title("🛒 Amazon Product Recommender")
st.markdown("Encontrá productos similares usando Machine Learning")

st.sidebar.header("Filtros")

# Filtro por categoría
categorias = ['Todas'] + sorted(df['main_category'].unique().tolist())
categoria_sel = st.sidebar.selectbox("Categoría", categorias)

# Filtro por precio máximo
precio_max = st.sidebar.slider(
    "Precio máximo (₹)",
    min_value=int(df['discounted_price'].min()),
    max_value=int(df['discounted_price'].max()),
    value=int(df['discounted_price'].max())
)

# Filtro por rating mínimo
rating_min = st.sidebar.slider(
    "Rating mínimo",
    min_value=2.0,
    max_value=5.0,
    value=3.5,
    step=0.1
)

# ── Búsqueda ──────────────────────────────────────────────────────────────────
st.subheader("Buscá un producto")
busqueda = st.text_input("Nombre del producto", placeholder="Ej: Samsung, Cable USB, Mixer...")

if busqueda:
    df_filtrado = df.copy()
    
    if categoria_sel != 'Todas':
        df_filtrado = df_filtrado[df_filtrado['main_category'] == categoria_sel]
    
    df_filtrado = df_filtrado[
        (df_filtrado['discounted_price'] <= precio_max) &
        (df_filtrado['rating'] >= rating_min)
    ]

    resultado = recomendar(busqueda, df_filtrado, similitud)

    if isinstance(resultado, str):
        st.warning(resultado)
    else:
        st.subheader(f"Productos similares a: *{busqueda}*")
        
        for _, row in resultado.iterrows():
            with st.container():
                col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
                col1.write(f"**{row['product_name'][:80]}...**")
                col2.write(f"⭐ {row['rating']}")
                col3.write(f"₹{int(row['discounted_price'])}")
                col4.write(f"📦 {row['main_category']}")
                st.divider()