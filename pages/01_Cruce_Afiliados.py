import streamlit as st
import pandas as pd
import os
import utils # Importar el archivo utils.py de la raíz

st.set_page_config(page_title="Cruce Inteligente", layout="wide", page_icon="🔗")
st.title("🔗 Cruce Inteligente de Afiliados ALSUM")

# 1. Rutas
PLAN_PATH = utils.get_file_path("plan_2026.xlsx")
NUEVOS_PATH = utils.get_file_path("nuevos_afiliados.xlsx")
DIRECTORIO_PATH = utils.get_file_path("Directorio_Afiliados_2025.xlsx")

# 2. Carga
with st.spinner("Cargando bases de datos..."):
    plan_accion, err = utils.load_plan_accion_procesado(PLAN_PATH, sheet_name="Afiliados")
    nuevos_afiliados = utils.load_excel_sheet(NUEVOS_PATH, sheet_name="Hoja1")
    directorio = utils.load_excel_sheet(DIRECTORIO_PATH, sheet_name="Directorio 2025")

if err or nuevos_afiliados is None:
    st.error("Faltan archivos o hay error de lectura. Verifique que existan 'plan_2026.xlsx' y 'nuevos_afiliados.xlsx'.")
    st.stop()

st.success("Archivos cargados correctamente.")

# 3. Preparación para cruce (Normalización)
# Asegurarnos de tener columnas para comparar
col_plan = 'Compañía'
col_nuevos = 'nombre_norm' # Crearemos esta si no existe

# Si 'nuevos_afiliados' no tiene nombre claro, busca la primera columna de texto
col_nombre_nuevos = 'Nombre' if 'Nombre' in nuevos_afiliados.columns else nuevos_afiliados.columns[0]
nuevos_afiliados['nombre_norm'] = nuevos_afiliados[col_nombre_nuevos].apply(utils.normalize_text)

# En plan de acción
plan_accion['nombre_norm'] = plan_accion['Compañía'].apply(utils.normalize_text)

st.header("📊 Resultado del Cruce")

if st.button("Ejecutar Cruce Fuzzy (Puede tardar unos segundos)"):
    with st.spinner("Realizando inteligencia de nombres..."):
        # Usamos la función de utils que ahora SÍ existe
        # Cruzamos: Plan de acción (Left) vs Nuevos Afiliados (Right)
        merged = utils.fuzzy_merge(
            plan_accion[['Compañía', 'nombre_norm', 'País']].drop_duplicates(), # Optimizamos pasando solo únicos
            nuevos_afiliados[['nombre_norm', 'Categoria']], 
            left_on='nombre_norm', 
            right_on='nombre_norm', 
            threshold=85
        )
        
        # Clasificar resultados
        merged['Estado'] = merged['match_name'].apply(lambda x: "✅ Encontrado" if pd.notna(x) else "❌ No Encontrado")
        
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Total Empresas en Plan", len(merged))
        with c2:
            encontrados = merged[merged['Estado']=="✅ Encontrado"].shape[0]
            st.metric("Coincidencias Detectadas", encontrados)
            
        st.dataframe(merged.sort_values('match_score', ascending=False), use_container_width=True)
        
        st.subheader("No Afiliados Detectados (Oportunidades)")
        oportunidades = merged[merged['Estado'] == "❌ No Encontrado"]
        st.dataframe(oportunidades[['Compañía', 'País']], use_container_width=True)

else:
    st.info("Presiona el botón para iniciar el análisis de coincidencia.")