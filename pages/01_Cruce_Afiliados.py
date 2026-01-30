import streamlit as st
import pandas as pd
import numpy as np
import os

# Importar utilidades propias
import utils

st.set_page_config(page_title="Cruce Inteligente de Afiliados", layout="wide", page_icon="🔗")

st.title("🔗 Cruce Inteligente de Afiliados ALSUM 2025-2026")

st.header("1️⃣ Carga de Archivos (Vía Utils)")

# Definición de rutas usando utils
DATA_FILE = "plan_2026.xlsx"
PLAN_ACCION_PATH = utils.get_file_path(DATA_FILE)
NUEVOS_AFILIADOS_PATH = utils.get_file_path("nuevos_afiliados.xlsx")
DIRECTORIO_PATH = utils.get_file_path("Directorio_Afiliados_2025.xlsx")

# Verificación inicial
if not (os.path.exists(PLAN_ACCION_PATH) and os.path.exists(NUEVOS_AFILIADOS_PATH) and os.path.exists(DIRECTORIO_PATH)):
    st.error("❌ Faltan archivos en el directorio raíz. Verifique: Plan de accion 2026, nuevos_afiliados, Directorio_Afiliados_2025")
    st.stop()

# Carga usando Utils (Usamos load_plan_accion_procesado para el principal para tener nombres normalizados)
try:
    plan_accion, err = utils.load_plan_accion_procesado(PLAN_ACCION_PATH)
    if err: st.warning(f"Advertencia en Plan Acción: {err}")
    else: st.success("Plan de acción cargado y procesado correctamente.")
    
    nuevos_afiliados = utils.load_simple_excel(NUEVOS_AFILIADOS_PATH)
    st.success("nuevos_afiliados.xlsx cargado correctamente.")
    
    directorio = utils.load_simple_excel(DIRECTORIO_PATH)
    st.success("Directorio_Afiliados_2025.xlsx cargado correctamente.")

except Exception as e:
    st.error(f"Error crítico en carga: {e}")
    st.stop()

st.write("Primeras filas de Plan de acción 2026 (Procesado):")
st.dataframe(plan_accion.head())

st.header("2️⃣ KPIs y Análisis")

# --- Afiliados 2025: miembros vs asociados ---
st.subheader("Afiliados 2025: Miembros vs Asociados")
if nuevos_afiliados is not None:
    afiliados_2025 = nuevos_afiliados.copy()
    if "Categoria" in afiliados_2025.columns:
        afiliados_2025["Categoria"] = afiliados_2025["Categoria"].astype(str).str.strip().str.upper()
        miembros = afiliados_2025[afiliados_2025["Categoria"] == "MIEMBRO"]
        asociados = afiliados_2025[afiliados_2025["Categoria"] == "ASOCIADO"]

        st.metric("Total Afiliados 2025", len(afiliados_2025))
        st.metric("Miembros", len(miembros))
        st.metric("Asociados", len(asociados))

        # --- Miembros: aseguradoras vs reaseguradoras ---
        st.subheader("Miembros: Aseguradoras vs Reaseguradoras")
        def tipo_empresa(row):
            tipo = str(row.get("Tipo_Afiliado", "")).lower()
            if "reasegurad" in tipo: return "Reaseguradora"
            elif "asegurad" in tipo: return "Aseguradora"
            else: return "Otro"

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
            
            # Cruce difuso con directorio para obtener país
            # Usamos la función fuzzy_merge de utils
            asociados = utils.fuzzy_merge(asociados, directorio, "nombre_norm", "nombre_norm", threshold=80)
            
            if "País sede operación" in directorio.columns:
                asociados_pais = asociados.merge(directorio[["nombre_norm", "País sede operación"]], left_on="match_name", right_on="nombre_norm", how="left")
                asociados_pais_count = asociados_pais.groupby("País sede operación").size().reset_index(name="Cantidad")
                st.dataframe(asociados_pais_count, use_container_width=True)
            else:
                st.warning("Columna 'País sede operación' no encontrada en Directorio.")
        else:
            st.info("No hay asociados en los nuevos afiliados.")
    else:
        st.error("Columna 'Categoria' no encontrada en nuevos_afiliados.")

# --- No afiliados: cantidad por país de aseguradoras y afiliación ---
st.subheader("No Afiliados: Cantidad por País de Aseguradoras y Afiliación")

# Paso 1: Identificar quiénes del Plan de Acción YA son afiliados
# Usamos fuzzy_merge de utils para cruzar plan_accion con nuevos_afiliados
plan_accion_cruzado = utils.fuzzy_merge(plan_accion, nuevos_afiliados, "nombre_norm", "nombre_norm", threshold=85)

# Filtramos los que NO tuvieron match (score bajo o nulo)
plan_no_afiliados = plan_accion_cruzado[plan_accion_cruzado['match_score'] < 85].copy()

# Determinar aseguradoras por columna 'Compañía'
def clasificar_empresa(val):
    val_str = str(val).lower()
    if "asegurad" in val_str: return "Aseguradora"
    if "reasegurad" in val_str: return "Reaseguradora"
    return "Otro"

plan_no_afiliados["Tipo_Empresa_Calc"] = plan_no_afiliados["Compañía"].apply(clasificar_empresa)
no_afiliados_aseg = plan_no_afiliados[plan_no_afiliados["Tipo_Empresa_Calc"] == "Aseguradora"]

# Cantidad por país
if not no_afiliados_aseg.empty:
    no_afiliados_aseg_pais = no_afiliados_aseg.groupby("País").size().reset_index(name="Cantidad")
    st.dataframe(no_afiliados_aseg_pais, use_container_width=True)

    # Saber cuáles son afiliadas y cuáles no (por fuzzy match con directorio general histórico si fuera necesario)
    # Aquí reutilizamos el directorio cargado
    no_afiliados_aseg = utils.fuzzy_merge(no_afiliados_aseg, directorio, "nombre_norm", "nombre_norm", threshold=80)
    no_afiliados_aseg["Es_Afiliada_Historico"] = no_afiliados_aseg["match_name"].notna()
    
    st.write("Detalle de empresas detectadas como No Afiliadas en el Plan 2026:")
    st.dataframe(no_afiliados_aseg[["Compañía", "País", "Es_Afiliada_Historico", "match_score"]], use_container_width=True)
else:
    st.info("No se encontraron Aseguradoras no afiliadas con los criterios actuales.")

st.success("Análisis cruzado completado.")