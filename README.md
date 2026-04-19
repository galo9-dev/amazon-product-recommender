🔗 **Demo en vivo:** https://amazon-appuct-recommender-ikhqtfwbdhvirpweqqmvkf.streamlit.app

# 🛒 Amazon Product Recommender

Sistema de recomendación de productos basado en Machine Learning, construido con Python y Streamlit.

## Descripción

Este proyecto implementa un sistema de recomendación **content-based** usando TF-IDF y similitud coseno sobre un dataset de productos de Amazon India. El usuario puede buscar un producto y obtener recomendaciones similares filtradas por categoría, precio y rating.

## Decisiones técnicas

- **Content-based filtering** con TF-IDF + Cosine Similarity sobre nombre, categoría y descripción del producto.
- **Filtrado colaborativo descartado:** se intentó implementar un modelo colaborativo usando SVD para recomendar productos basándose en el comportamiento de usuarios similares. Sin embargo, el dataset no contiene ratings individuales por usuario — cada fila tiene un rating general del producto y una lista de usuarios que dejaron reseña, pero sin una puntuación por usuario. Esto hace imposible construir una matriz usuario-producto real, generando el problema conocido como *cold start*. La implementación del intento está documentada en `src/modelo_colaborativo.py`.

## Cómo correr el proyecto

**1. Clonar el repositorio**
```bash
git clone https://github.com/tu-usuario/amazon-product-recommender.git
cd amazon-product-recommender
```

**2. Crear y activar el entorno virtual**
```bash
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux
```

**3. Instalar dependencias**
```bash
pip install -r requirements.txt
```

**4. Correr la app**
```bash
streamlit run src/app.py
```

## Stack

- Python 3.11
- Pandas
- Scikit-learn (TF-IDF, SVD, Cosine Similarity)
- Streamlit
- Jupyter Notebook

## 📊 Dataset

[Amazon Sales Dataset](https://www.kaggle.com/datasets/karkavelrajaj/amazon-sales-dataset) — 1465 productos de Amazon India con nombre, categoría, descripción, precio y ratings.