import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

# ── Carga y limpieza ──────────────────────────────────────────────────────────
def cargar_datos(path='data/amazon.csv'):
    df = pd.read_csv(path)
    
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
    df['main_category'] = df['category'].str.split('|').str[0]
    
    # Separar usuarios (cada fila tiene múltiples user_ids separados por coma)
    df_exp = df[['product_id', 'product_name', 'user_id', 'rating']].copy()
    df_exp = df_exp.dropna(subset=['rating'])
    df_exp['user_id'] = df_exp['user_id'].str.split(',')
    df_exp = df_exp.explode('user_id')
    df_exp['user_id'] = df_exp['user_id'].str.strip()
    
    return df, df_exp

# ── Matriz usuario-producto ───────────────────────────────────────────────────
def construir_matriz(df_exp):
    matriz = df_exp.pivot_table(
        index='user_id',
        columns='product_id',
        values='rating',
        aggfunc='mean'
    ).fillna(0)
    
    return matriz

# ── Modelo SVD ────────────────────────────────────────────────────────────────
def construir_modelo(matriz):
    svd = TruncatedSVD(n_components=20, random_state=42)
    matriz_reducida = svd.fit_transform(matriz)
    similitud = cosine_similarity(matriz_reducida)
    return similitud, matriz

# ── Recomendador ──────────────────────────────────────────────────────────────
def recomendar(user_id, df, df_exp, matriz, similitud, n=5):
    if user_id not in matriz.index:
        return "Usuario no encontrado."
    
    # Índice del usuario
    idx = list(matriz.index).index(user_id)
    
    # Usuarios más similares
    puntajes = list(enumerate(similitud[idx]))
    puntajes = sorted(puntajes, key=lambda x: x[1], reverse=True)
    usuarios_similares = [matriz.index[i[0]] for i in puntajes[1:6]]
    
    # Productos que vieron esos usuarios
    productos_vistos = df_exp[df_exp['user_id'].isin(usuarios_similares)]['product_id'].unique()
    
    # Productos que el usuario actual NO vio
    productos_usuario = df_exp[df_exp['user_id'] == user_id]['product_id'].unique()
    productos_nuevos = [p for p in productos_vistos if p not in productos_usuario]
    
    # Devolver info de esos productos
    resultado = df[df['product_id'].isin(productos_nuevos[:n])][['product_name', 'main_category', 'rating', 'discounted_price']]
    resultado['discounted_price'] = resultado['discounted_price'].str.replace('₹', '').str.replace(',', '').astype(float)
    
    return resultado

# ── Test ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    df, df_exp = cargar_datos()
    matriz = construir_matriz(df_exp)
    similitud, matriz = construir_modelo(matriz)
    
    # Buscar un usuario con suficientes interacciones
    usuario_ejemplo = df_exp['user_id'].value_counts().index[5]
    print(f"Usuario: {usuario_ejemplo}")
    print(f"Productos que vio: {len(df_exp[df_exp['user_id'] == usuario_ejemplo])}")
    resultado = recomendar(usuario_ejemplo, df, df_exp, matriz, similitud)
    print(resultado)