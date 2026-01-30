import streamlit as st
import pandas as pd
from fuzzywuzzy import process
import numpy as np
import os

st.set_page_config(page_title="Cruce Inteligente de Afiliados", layout="wide", page_icon="🔗")

st.title("🔗 Cruce Inteligente de Afiliados ALSUM 2025-2026")

st.header("1️⃣ Carga de Archivos (Automática desde la raíz del repositorio)")

# Calcula la ruta a la raíz del repositorio (un nivel arriba de /pages)
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Define las rutas relativas a los archivos en la raíz
PLAN_ACCION_PATH = os.path.join(ROOT_DIR, "Plan de accion 2026.xlsx")
NUEVOS_AFILIADOS_PATH = os.path.join(ROOT_DIR, "nuevos_afiliados.xlsx")
DIRECTORIO_PATH = os.path.join(ROOT_DIR, "directorio_Afiliados_2025.xlsx")

# Verifica existencia y carga
if not (os.path.exists(PLAN_ACCION_PATH) and os.path.exists(NUEVOS_AFILIADOS_PATH) and os.path.exists(DIRECTORIO_PATH)):
    st.error("❌ No se encontraron todos los archivos requeridos en la raíz del repositorio. Asegúrate de que existan:\n"
             f"- {PLAN_ACCION_PATH}\n- {NUEVOS_AFILIADOS_PATH}\n- {DIRECTORIO_PATH}")
    st.stop()

plan_accion = pd.read_excel(PLAN_ACCION_PATH)
nuevos_afiliados = pd.read_excel(NUEVOS_AFILIADOS_PATH)
directorio = pd.read_excel(DIRECTORIO_PATH)

# 2. Lectura de datos
plan_accion = pd.read_excel(PLAN_ACCION_PATH)
nuevos_afiliados = pd.read_excel(NUEVOS_AFILIADOS_PATH)
directorio = pd.read_excel(DIRECTORIO_PATH)

# 3. Normalización de nombres
def normalizar_nombre(nombre):
    if pd.isna(nombre): return ""
    return (
        str(nombre)
        .lower()
        .replace("s.a.", "")
        .replace("s.a", "")
        .replace("sa", "")
        .replace("compañía", "")
        .replace("compania", "")
        .replace("aseguradora", "")
        .replace("reaseguradora", "")
        .replace("de seguros", "")
        .replace("de reaseguros", "")
        .replace(".", "")
        .replace(",", "")
        .replace("-", " ")
        .replace("&", "y")
        .replace("  ", " ")
        .strip()
    )

plan_accion["nombre_norm"] = plan_accion["Compañía"].apply(normalizar_nombre)
nuevos_afiliados["nombre_norm"] = nuevos_afiliados["Compañía"].apply(normalizar_nombre)
directorio["nombre_norm"] = directorio["Empresa"].apply(normalizar_nombre)

# 4. Fuzzy Matching para cruce de empresas
def fuzzy_merge(df_izq, df_der, key_izq, key_der, threshold=80, limit=1):
    s = df_der[key_der].tolist()
    matches = df_izq[key_izq].apply(
        lambda x: process.extractOne(x, s, score_cutoff=threshold)
    )
    df_izq["match_name"] = matches.apply(lambda x: x[0] if x else np.nan)
    df_izq["match_score"] = matches.apply(lambda x: x[1] if x else np.nan)
    return df_izq

# Cruce plan de acción vs directorio
plan_accion = fuzzy_merge(plan_accion, directorio, "nombre_norm", "nombre_norm", threshold=80)
# Cruce nuevos afiliados vs directorio
nuevos_afiliados = fuzzy_merge(nuevos_afiliados, directorio, "nombre_norm", "nombre_norm", threshold=80)

# 5. KPIs y análisis

st.header("2️⃣ KPIs y Análisis")

# --- Afiliados 2025: miembros vs asociados ---
st.subheader("Afiliados 2025: Miembros vs Asociados")
afiliados_2025 = nuevos_afiliados.copy()
afiliados_2025["Categoria"] = afiliados_2025["Categoria"].str.strip().str.upper()
miembros = afiliados_2025[afiliados_2025["Categoria"] == "MIEMBRO"]
asociados = afiliados_2025[afiliados_2025["Categoria"] == "ASOCIADO"]

st.metric("Total Afiliados 2025", len(afiliados_2025))
st.metric("Miembros", len(miembros))
st.metric("Asociados", len(asociados))

# --- Miembros: aseguradoras vs reaseguradoras ---
st.subheader("Miembros: Aseguradoras vs Reaseguradoras")
def tipo_empresa(row):
    tipo = str(row.get("Tipo_Afiliado", "")).lower()
    if "reasegurad" in tipo:
        return "Reaseguradora"
    elif "asegurad" in tipo:
        return "Aseguradora"
    else:
        return "Otro"

miembros["Tipo_Empresa"] = miembros.apply(tipo_empresa, axis=1)
aseguradoras = miembros[miembros["Tipo_Empresa"] == "Aseguradora"]
reaseguradoras = miembros[miembros["Tipo_Empresa"] == "Reaseguradora"]

st.metric("Miembros - Aseguradoras", len(aseguradoras))
st.metric("Miembros - Reaseguradoras", len(reaseguradoras))

# --- Asociados: por categoría y país ---
st.subheader("Asociados: Por Categoría y País")
if not asociados.empty:
    asociados_cat = asociados.groupby("Categoria").size().reset_index(name="Cantidad")
    st.dataframe(asociados_cat, use_container_width=True)
    # Si hay columna país en directorio, cruzar para obtener país
    asociados = fuzzy_merge(asociados, directorio, "nombre_norm", "nombre_norm", threshold=80)
    asociados_pais = asociados.merge(directorio[["nombre_norm", "País sede operación"]], left_on="match_name", right_on="nombre_norm", how="left")
    asociados_pais_count = asociados_pais.groupby("País sede operación").size().reset_index(name="Cantidad")
    st.dataframe(asociados_pais_count, use_container_width=True)
else:
    st.info("No hay asociados en los nuevos afiliados.")

# --- No afiliados: cantidad por país de aseguradoras y afiliación ---
st.subheader("No Afiliados: Cantidad por País de Aseguradoras y Afiliación")
# Empresas en plan de acción que NO están en nuevos afiliados (por nombre fuzzy)
afiliados_nombres = set(afiliados_2025["match_name"].dropna())
plan_no_afiliados = plan_accion[~plan_accion["match_name"].isin(afiliados_nombres)]

# Determinar aseguradoras por columna 'Compañía'
plan_no_afiliados["Tipo_Empresa"] = plan_no_afiliados["Compañía"].apply(lambda x: "Aseguradora" if "asegurad" in str(x).lower() else ("Reaseguradora" if "reasegurad" in str(x).lower() else "Otro"))
no_afiliados_aseg = plan_no_afiliados[plan_no_afiliados["Tipo_Empresa"] == "Aseguradora"]

# Cantidad por país
no_afiliados_aseg_pais = no_afiliados_aseg.groupby("País").size().reset_index(name="Cantidad")
st.dataframe(no_afiliados_aseg_pais, use_container_width=True)

# Saber cuáles son afiliadas y cuáles no (por fuzzy match con directorio)
no_afiliados_aseg = fuzzy_merge(no_afiliados_aseg, directorio, "nombre_norm", "nombre_norm", threshold=80)
no_afiliados_aseg["Es_Afiliada"] = no_afiliados_aseg["match_name"].notna()
st.dataframe(no_afiliados_aseg[["Compañía", "País", "Es_Afiliada"]], use_container_width=True)

st.success("Análisis completado. Puedes descargar las tablas desde el menú de Streamlit.")