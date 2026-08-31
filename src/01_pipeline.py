import pandas as pd
import requests

# ============================================================
# 1. EXTRACCIÓN — Fetch the data
# ============================================================

url_datos = "https://ourworldindata.org/grapher/life-expectancy.csv?v=1&csvType=full&useColumnShortNames=true"

df = pd.read_csv(
    url_datos,
    storage_options={
        "User-Agent": "Our World In Data data fetch/1.0"
    }
)

# ============================================================
# 2. EXTRACCIÓN — Fetch the metadata
# ============================================================

url_metadata = "https://ourworldindata.org/grapher/life-expectancy.metadata.json?v=1&csvType=full&useColumnShortNames=true"

metadata = requests.get(url_metadata).json()

# ============================================================
# 3. EXPLORACIÓN INICIAL
# ============================================================

print("Dimensiones del DataFrame:")
print(df.shape)

print("\nColumnas:")
print(df.columns.tolist())

print("\nTipos de datos:")
print(df.dtypes)

print("\nValores nulos:")
print(df.isnull().sum())

# ============================================================
# 4. LIMPIEZA
# ============================================================

# Eliminar registros que no tengan código de país
df = df.dropna(subset=["code"])

# Eliminar registros que no tengan esperanza de vida
df = df.dropna(subset=["life_expectancy_0"])

# ============================================================
# 5. TRANSFORMACIÓN
# ============================================================

# Convertir el año a entero
df["Year"] = df["year"].astype(int)

# Crear una nueva columna con la esperanza de vida
# redondeada a dos decimales
df["life_expectancy_round"] = df["life_expectancy_0"].round(2)

# ============================================================
# 6. FILTRADO
# ============================================================

# Trabajaremos únicamente con datos a partir del año 2000
df = df[df["year"] >= 2000]

# ============================================================
# 7. ANÁLISIS
# ============================================================

# Promedio de esperanza de vida por país
promedio_pais = (
    df.groupby("entity")["life_expectancy_round"]
      .mean()
      .round(2)
      .sort_values(ascending=False)
)

print("\nPromedio de esperanza de vida por país:")
print(promedio_pais.head(10))

# ============================================================
# 8. RESULTADO
# ============================================================
archivo_salida = "data/processed/promedio_esperanza_vida_por_pais.csv"

promedio_pais.to_csv(
    archivo_salida,
    header=["Average life expectancy"]
)

print("\nPipeline ejecutado correctamente.")
print(f"Archivo generado: {archivo_salida}")