import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ── Carga y limpieza ──────────────────────────────────────────────────────────
def cargar_datos(path='data/amazon.csv'):
    df = pd.read_csv(path)
    
    df['discounted_price'] = df['discounted_price'].str.replace('₹', '').str.replace(',', '').astype(float)
    df['actual_price'] = df['actual_price'].str.replace('₹', '').str.replace(',', '').astype(float)
    df['discount_percentage'] = df['discount_percentage'].str.replace('%', '').astype(float)
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
    df['rating_count'] = df['rating_count'].str.replace(',', '').astype(float)
    df['main_category'] = df['category'].str.split('|').str[0]
    df['texto_modelo'] = df['product_name'] + ' ' + df['main_category'] + ' ' + df['about_product']
    
    return df

# ── Modelo ────────────────────────────────────────────────────────────────────
def construir_modelo(df):
    tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
    matriz = tfidf.fit_transform(df['texto_modelo'])
    similitud = cosine_similarity(matriz)
    return similitud

def recomendar(product_name, df, similitud, n=5):
    # Encontrar todos los productos que matchean
    indices = df[df['product_name'].str.contains(product_name, case=False, na=False)].index.tolist()
    
    if len(indices) == 0:
        return "Producto no encontrado."
    
    # Promediar las similitudes de todos los matches
    puntajes_promedio = np.mean(similitud[indices], axis=0)
    
    # Ordenar y excluir los productos que ya matchearon
    puntajes = list(enumerate(puntajes_promedio))
    puntajes = [(i, p) for i, p in puntajes if i not in indices]
    puntajes = sorted(puntajes, key=lambda x: x[1], reverse=True)[:n]
    
    resultado_indices = [i[0] for i in puntajes]
    return df[['product_name', 'main_category', 'rating', 'discounted_price']].iloc[resultado_indices]

# ── Test ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    df = cargar_datos()
    similitud = construir_modelo(df)
    
    print("=== Recomendaciones ===")
    resultado = recomendar('Samsung', df, similitud)
    print(resultado)