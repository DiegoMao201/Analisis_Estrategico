import streamlit as st
import pandas as pd
from fuzzywuzzy import process
import numpy as np
import os

st.set_page_config(page_title="Cruce Inteligente de Afiliados", layout="wide", page_icon="🔗")

st.title("🔗 Cruce Inteligente de Afiliados ALSUM 2025-2026")

st.header("1️⃣ Carga de Archivos (Automática desde la raíz del repositorio)")

# Siempre busca en el directorio actual de trabajo
ROOT_DIR = os.getcwd()

# Define las rutas relativas a los archivos en la raíz
PLAN_ACCION_PATH = os.path.join(ROOT_DIR, "Plan de acción 2026.xlsx")
NUEVOS_AFILIADOS_PATH = os.path.join(ROOT_DIR, "nuevos_afiliados.xlsx")
DIRECTORIO_PATH = os.path.join(ROOT_DIR, "Directorio_Afiliados_2025.xlsx")

st.write("Buscando archivos en:", ROOT_DIR)
st.write("Archivos encontrados:", os.listdir(ROOT_DIR))

# Verifica existencia y carga
if not (os.path.exists(PLAN_ACCION_PATH) and os.path.exists(NUEVOS_AFILIADOS_PATH) and os.path.exists(DIRECTORIO_PATH)):
    st.error("❌ No se encontraron todos los archivos requeridos en la raíz del repositorio. Asegúrate de que existan:\n"
             f"- {PLAN_ACCION_PATH}\n- {NUEVOS_AFILIADOS_PATH}\n- {DIRECTORIO_PATH}")
    st.stop()

try:
    plan_accion = pd.read_excel(PLAN_ACCION_PATH)
    st.success("Plan de acción 2026.xlsx cargado correctamente.")
except Exception as e:
    st.error(f"Error cargando Plan de acción 2026.xlsx: {e}")
    st.stop()

try:
    nuevos_afiliados = pd.read_excel(NUEVOS_AFILIADOS_PATH)
    st.success("nuevos_afiliados.xlsx cargado correctamente.")
except Exception as e:
    st.error(f"Error cargando nuevos_afiliados.xlsx: {e}")
    st.stop()

try:
    directorio = pd.read_excel(DIRECTORIO_PATH)
    st.success("Directorio_Afiliados_2025.xlsx cargado correctamente.")
except Exception as e:
    st.error(f"Error cargando Directorio_Afiliados_2025.xlsx: {e}")
    st.stop()

st.write("Primeras filas de Plan de acción 2026.xlsx:")
st.dataframe(plan_accion.head())
st.write("Primeras filas de nuevos_afiliados.xlsx:")
st.dataframe(nuevos_afiliados.head())
st.write("Primeras filas de Directorio_Afiliados_2025.xlsx:")
st.dataframe(directorio.head())

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