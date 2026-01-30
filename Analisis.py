import streamlit as st
import pandas as pd
import plotly.express as px
import datetime
import os
from openai import OpenAI

# Importar utilidades propias
import utils

# ==========================================
# 0. CONFIGURACIÓN INICIAL
# ==========================================
st.set_page_config(
    page_title="ALSUM 2026 | Strategic Command", 
    layout="wide", 
    page_icon="🦅",
    initial_sidebar_state="expanded"
)

# ==========================================
# 1. ESTILOS CSS
# ==========================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap');
    html, body, [class*="css"]  { font-family: 'Roboto', sans-serif; }
    h1, h2, h3 { color: #004A8F; font-weight: 700; }
    div[data-testid="metric-container"] {
        background-color: #FFFFFF;
        border: 1px solid #E0E0E0;
        padding: 15px;
        border-radius: 8px;
        border-left: 6px solid #004A8F;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    div.stButton > button {
        background: linear-gradient(135deg, #004A8F 0%, #002a52 100%);
        color: white; 
        font-weight: bold;
        border-radius: 8px; 
        border: none;
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. CARGA DE DATOS
# ==========================================
DATA_FILE = "plan_2026.xlsx"
FULL_PATH = utils.get_file_path(DATA_FILE)

# Carga de datos con indicador visual para evitar sensación de "colgado"
with st.spinner('Cargando datos maestros de ALSUM...'):
    df_final, error = utils.load_plan_accion_procesado(FULL_PATH, sheet_name="Afiliados")

if error:
    st.error(f"❌ {error}")
    st.info(f"Ruta intentada: {FULL_PATH}")
    st.stop()
elif df_final is None or df_final.empty:
    st.error("⚠️ El archivo se cargó pero no contiene datos válidos o está vacío.")
    st.stop()

# ---------------------------------------------------------
# 1. FILTROS GLOBALES (SIDEBAR)
# ---------------------------------------------------------
st.sidebar.image("https://alsum.co/wp-content/uploads/2018/06/Logo-Alsum-Web.png", width=150) # Logo opcional
st.sidebar.markdown("### 🌍 Filtros Macro")

anios_disp = sorted(df_final['Año'].unique())
paises_disp = sorted(df_final['País'].unique())
ramos_disp = sorted(df_final['Ramo'].unique())

filtro_anios = st.sidebar.multiselect("Año", anios_disp, default=anios_disp)
filtro_paises = st.sidebar.multiselect("País", paises_disp, default=paises_disp)
filtro_afiliado = st.sidebar.radio("Afiliación", ["Todos", "Afiliados", "No afiliados"], horizontal=False)
filtro_ramos = st.sidebar.multiselect("Ramo", ramos_disp, default=ramos_disp)

# Aplicar Filtros
df_filtrado = df_final.copy()
    
if filtro_anios: 
    df_filtrado = df_filtrado[df_filtrado['Año'].isin(filtro_anios)]
if filtro_paises: 
    df_filtrado = df_filtrado[df_filtrado['País'].isin(filtro_paises)]
if filtro_afiliado == "Afiliados": 
    df_filtrado = df_filtrado[df_filtrado['AFILIADO'] == 'AFILIADO']
elif filtro_afiliado == "No afiliados": 
    df_filtrado = df_filtrado[df_filtrado['AFILIADO'] == 'NO AFILIADO']
if filtro_ramos: 
    df_filtrado = df_filtrado[df_filtrado['Ramo'].isin(filtro_ramos)]

# ---------------------------------------------------------
# 2. AREA PRINCIPAL
# ---------------------------------------------------------
st.title(f"🚀 Análisis Comercial Lina Marcela Contreras {datetime.date.today().year}")

with st.expander("🛠️ Filtros de Profundización y Configuración", expanded=False):
    c_filt1, c_filt2, c_filt3 = st.columns([2, 1, 1])
    with c_filt1:
        empresas_disponibles = sorted(df_filtrado['Compañía'].unique())
        filtro_empresas = st.multiselect("🏢 Filtrar por Compañías Específicas:", empresas_disponibles)
    with c_filt2:
        metrica_view = st.radio("👁️ Métrica Principal:", ["Primas", "Siniestros"], horizontal=True)
    with c_filt3:
        ver_desglose_anos = st.toggle("📅 Desglosar Años", value=True)
        
    if filtro_empresas:
        df_filtrado = df_filtrado[df_filtrado['Compañía'].isin(filtro_empresas)]

if df_filtrado.empty:
    st.warning("⚠️ No hay datos para los filtros seleccionados.")
    st.stop()

# ---------------------------------------------------------
# 3. KPIs GLOBALES
# ---------------------------------------------------------
escala = 1e9 # Billones
primas_tot = df_filtrado['Primas'].sum()
siniestros_tot = df_filtrado['Siniestros'].sum()
ratio_global = (siniestros_tot / primas_tot) * 100 if primas_tot > 0 else 0
res_tec = primas_tot - siniestros_tot

k1, k2, k3, k4 = st.columns(4)
k1.metric("Volumen Primas (USD)", f"${primas_tot/escala:,.2f}B")
k2.metric("Siniestros Totales (USD)", f"${siniestros_tot/escala:,.2f}B")
k3.metric("Siniestralidad Global", f"{ratio_global:.1f}%", delta=f"{65-ratio_global:.1f}% vs Meta")
k4.metric("Resultado Técnico", f"${res_tec/escala:,.2f}B")
st.markdown("---")

# ---------------------------------------------------------
# 4. PESTAÑAS
# ---------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(["🌎 Mapa", "📦 Productos", "🧠 INFORME IA", "🎯 Profundización"])

# === TAB 1: GEOGRÁFICO ===
with tab1:
    st.subheader("Análisis de Territorio")
    
    if ver_desglose_anos and len(filtro_anios) > 1:
        st.markdown(f"##### 📅 Vista Desglosada por Años ({metrica_view})")
        df_pivot_pais = utils.crear_vista_pivot_anos(df_filtrado, 'País', metrica_view)
        cols_num = [c for c in df_pivot_pais.columns if c != 'País']
        st.dataframe(df_pivot_pais.style.format({col: '${:,.0f}' for col in cols_num}), use_container_width=True, hide_index=True)
        
    pais_chart = df_filtrado.groupby('País')[['Primas', 'Siniestros']].sum().reset_index()
    pais_chart['Siniestralidad'] = pais_chart.apply(lambda x: (x['Siniestros']/x['Primas']*100) if x['Primas']>0 else 0, axis=1)
        
    c1, c2 = st.columns([2, 1])
    with c1:
        fig_map = px.scatter(pais_chart, x='Primas', y='Siniestralidad', size='Primas', color='País',
                             title="Matriz Riesgo vs Volumen")
        fig_map.add_hline(y=65, line_dash="dash", line_color="red")
        st.plotly_chart(fig_map, use_container_width=True)
    with c2:
        st.markdown("**Top Países**")
        st.dataframe(pais_chart[['País', 'Primas', 'Siniestralidad']].sort_values('Primas', ascending=False)
                     .style.format({'Primas':'${:,.0f}', 'Siniestralidad':'{:.1f}%'}), use_container_width=True, hide_index=True)

    st.markdown("### Ranking de Compañías")
    comp_geo = df_filtrado.groupby('Compañía')[['Primas','Siniestros']].sum().reset_index()
    comp_geo['Siniestralidad'] = comp_geo.apply(lambda x: (x['Siniestros']/x['Primas']*100) if x['Primas']>0 else 0, axis=1)
    
    st.dataframe(comp_geo.sort_values('Primas', ascending=False).head(50).style
                 .format({'Primas':'${:,.0f}', 'Siniestros':'${:,.0f}', 'Siniestralidad':'{:.1f}%'})
                 .background_gradient(subset=['Siniestralidad'], cmap='RdYlGn_r', vmin=0, vmax=100),
                 use_container_width=True, hide_index=True)

# === TAB 2: PRODUCTOS ===
with tab2:
    st.subheader("Análisis por Ramo")
    ramo_df = df_filtrado.groupby('Ramo')[['Primas', 'Siniestros']].sum().reset_index()
    ramo_df['Ratio'] = ramo_df.apply(lambda x: (x['Siniestros']/x['Primas']*100) if x['Primas']>0 else 0, axis=1)
    fig_bar = px.bar(ramo_df.sort_values('Primas', ascending=False), x='Ramo', y='Primas', color='Ratio',
                     color_continuous_scale='RdYlGn_r', title="Ramos: Volumen y Siniestralidad")
    st.plotly_chart(fig_bar, use_container_width=True)

# === TAB 3: PDF ===
with tab3:
    st.header("🧠 Generador de Informe")
    foco = st.text_area("🎯 Enfoque del reporte:", placeholder="Ej: Analizar crecimiento en México...")
    btn_gen = st.button("🔥 GENERAR PDF", type="primary")
    
    if btn_gen:
        api_key = utils.get_api_key()
        if not api_key:
            st.error("⚠️ Falta OPENAI_API_KEY.")
        else:
            with st.status("Generando...", expanded=True) as status:
                try:
                    # Lógica simplificada para el ejemplo
                    client = OpenAI(api_key=api_key)
                    prompt = f"Eres un experto en seguros. Resume este desempeño: Primas {primas_tot:,.0f}, Ratio {ratio_global:.1f}%. Foco: {foco}"
                    resp = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role":"user", "content": prompt}])
                    texto = resp.choices[0].message.content
                    
                    pdf = utils.UltimatePDF()
                    pdf.cover_page("PLAN 2026", "ESTRATEGIA")
                    pdf.add_page()
                    pdf.section_title("Diagnóstico IA")
                    pdf.chapter_body(texto)
                    
                    pdf_bytes = bytes(pdf.output(dest='S'))
                    status.update(label="Listo!", state="complete", expanded=False)
                    st.download_button("Descargar PDF", pdf_bytes, "Plan_2026.pdf", "application/pdf")
                except Exception as e:
                    st.error(f"Error: {e}")

# === TAB 4: PROFUNDIZACIÓN ===
with tab4:
    st.header("Correlación Detallada")
    comp_deep = df_filtrado.groupby('Compañía')[['Primas','Siniestros']].sum().reset_index()
    comp_deep = comp_deep[comp_deep['Primas'] > 0]
    
    if not comp_deep.empty:
        fig_deep = px.scatter(comp_deep, x="Primas", y="Siniestros", hover_name="Compañía", trendline="ols")
        st.plotly_chart(fig_deep, use_container_width=True)
    else:
        st.info("Sin datos suficientes.")