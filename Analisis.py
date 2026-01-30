import streamlit as st
import pandas as pd
import plotly.express as px
import datetime
import os
from openai import OpenAI
import openpyxl

# Importar utilidades propias (asegúrate de que utils.py esté en la misma carpeta)
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
# 1. ESTILOS CSS PREMIUM
# ==========================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap');
    html, body, [class*="css"]  { font-family: 'Roboto', sans-serif; }
    .main .block-container { padding-top: 1rem; padding-bottom: 3rem; }
    h1, h2, h3 { color: #004A8F; font-weight: 700; }
    div[data-testid="metric-container"] {
        background-color: #FFFFFF;
        border: 1px solid #E0E0E0;
        padding: 15px;
        border-radius: 8px;
        border-left: 6px solid #004A8F;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        transition: transform 0.2s;
    }
    div[data-testid="metric-container"]:hover { transform: scale(1.02); }
    div.stButton > button {
        background: linear-gradient(135deg, #004A8F 0%, #002a52 100%);
        color: white; 
        font-weight: bold;
        border-radius: 8px; 
        border: none; 
        box-shadow: 0 4px 10px rgba(0, 74, 143, 0.3);
        transition: all 0.3s ease;
        width: 100%;
    }
    div.stButton > button:hover { transform: translateY(-2px); }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. CARGA DE DATOS (Usando Utils)
# ==========================================
DATA_FILE = "plan_2026.xlsx"
FULL_PATH = utils.get_file_path(DATA_FILE)

# Carga de datos (usa hoja "Afiliados")
df_final, error = utils.load_plan_accion_procesado(FULL_PATH, sheet_name="Afiliados")

if error:
    st.error(f"❌ {error}")
    st.write("DataFrame retornado:", df_final)
    st.stop()
elif df_final is None:
    st.error("No se pudo cargar ningún dato. Revisa el archivo y la hoja.")
    st.stop()

# ---------------------------------------------------------
# 1. FILTROS GLOBALES (SIDEBAR - MACRO)
# ---------------------------------------------------------
anios_disp = sorted(df_final['Año'].unique())
paises_disp = sorted(df_final['País'].unique())
ramos_disp = sorted(df_final['Ramo'].unique())

st.sidebar.markdown("### 🌍 Filtros Macro")
filtro_anios = st.sidebar.multiselect("Año", anios_disp, default=anios_disp)
filtro_paises = st.sidebar.multiselect("País", paises_disp, default=paises_disp)
filtro_afiliado = st.sidebar.radio("Afiliación", ["Todos", "Afiliados", "No afiliados"], horizontal=False)
filtro_ramos = st.sidebar.multiselect("Ramo", ramos_disp, default=ramos_disp)

# Aplicar Filtros Sidebar
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
# 2. AREA PRINCIPAL Y FILTROS AVANZADOS (EXPANDER)
# ---------------------------------------------------------
st.title(f"🚀 Análisis Comercial Lina Marcela Contreras {datetime.date.today().year}")

with st.expander("🛠️ Filtros de Profundización y Configuración de Vistas", expanded=False):
    c_filt1, c_filt2, c_filt3 = st.columns([2, 1, 1])
    with c_filt1:
        empresas_disponibles = sorted(df_filtrado['Compañía'].unique())
        filtro_empresas = st.multiselect("🏢 Filtrar por Compañías Específicas:", empresas_disponibles)
    with c_filt2:
        metrica_view = st.radio("👁️ Métrica Principal:", ["Primas", "Siniestros"], horizontal=True)
    with c_filt3:
        ver_desglose_anos = st.toggle("📅 Desglosar Años en Columnas", value=True)
        
    if filtro_empresas:
        df_filtrado = df_filtrado[df_filtrado['Compañía'].isin(filtro_empresas)]

if df_filtrado.empty:
    st.warning("⚠️ No hay datos para los filtros seleccionados. Intenta ampliar la búsqueda.")
    st.stop()

# ---------------------------------------------------------
# 3. KPIs GLOBALES
# ---------------------------------------------------------
escala = 1e9 # Billones
primas_tot = df_filtrado['Primas'].sum()
siniestros_tot = df_filtrado['Siniestros'].sum()
    
# Cálculo seguro de ratio global
ratio_global = (siniestros_tot / primas_tot) * 100 if primas_tot > 0 else 0
res_tec = primas_tot - siniestros_tot

k1, k2, k3, k4 = st.columns(4)
k1.metric("Volumen Primas (USD)", f"${primas_tot/escala:,.2f}B")
k2.metric("Siniestros Totales (USD)", f"${siniestros_tot/escala:,.2f}B")
k3.metric("Siniestralidad Global", f"{ratio_global:.1f}%", delta=f"{65-ratio_global:.1f}% vs Meta (65%)")
k4.metric("Resultado Técnico", f"${res_tec/escala:,.2f}B")
st.markdown("---")

# ---------------------------------------------------------
# 4. PESTAÑAS
# ---------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(["🌎 Mapa de Análisis", "📦 Productos", "🧠 GENERADOR INFORME (PDF)", "🎯 Profundización"])

# === TAB 1: GEOGRÁFICO ===
with tab1:
    st.subheader("Análisis de Territorio")
        
    if ver_desglose_anos and len(filtro_anios) > 1:
        st.markdown(f"##### 📅 Vista Desglosada por Años ({metrica_view})")
        df_pivot_pais = utils.crear_vista_pivot_anos(df_filtrado, 'País', metrica_view)
            
        # Formateo dinámico de columnas
        cols_numericas = [c for c in df_pivot_pais.columns if c != 'País']
        format_dict = {col: '${:,.0f}' for col in cols_numericas}
            
        st.dataframe(
            df_pivot_pais.style.format(format_dict).background_gradient(cmap="Blues", subset=['TOTAL CONSOLIDADO']),
            use_container_width=True, hide_index=True
        )
        
    # Gráfica siempre visible debajo
    pais_df_chart = df_filtrado.groupby('País')[['Primas', 'Siniestros']].sum().reset_index()
    pais_df_chart['Siniestralidad'] = pais_df_chart.apply(
        lambda x: (x['Siniestros']/x['Primas']*100) if x['Primas'] > 0 else 0, axis=1
    )
        
    c1, c2 = st.columns([2, 1])
    with c1:
        fig_map = px.scatter(pais_df_chart, x='Primas', y='Siniestralidad', 
                                   size='Primas', color='País',
                                   hover_name='País',
                                   title="Matriz de Desempeño (Riesgo vs Volumen)")
        fig_map.add_hline(y=65, line_dash="dash", line_color="red", annotation_text="Límite Rentabilidad")
        st.plotly_chart(fig_map, use_container_width=True)
    with c2:
        st.markdown("**Top Países por Volumen**")
        st.dataframe(pais_df_chart[['País', 'Primas', 'Siniestralidad']].sort_values('Primas', ascending=False)
                     .style.format({'Primas':'${:,.0f}', 'Siniestralidad':'{:.1f}%'}), 
                     hide_index=True, use_container_width=True)

    st.markdown("### Ranking de Compañías")
    if ver_desglose_anos and len(filtro_anios) > 1:
        st.info(f"Mostrando desglose anual de **{metrica_view}** por Compañía")
        df_pivot_comp = utils.crear_vista_pivot_anos(df_filtrado, 'Compañía', metrica_view)
            
        cols_vals = [c for c in df_pivot_comp.columns if c != 'Compañía']
        st.dataframe(
            df_pivot_comp.head(50).style
            .format({col: '${:,.0f}' for col in cols_vals})
            .bar(subset=['TOTAL CONSOLIDADO'], color='#004A8F'),
            use_container_width=True, hide_index=True
        )
    else:
        comp_geo = df_filtrado.groupby('Compañía')[['Primas','Siniestros']].sum().reset_index()
        comp_geo['Siniestralidad'] = comp_geo.apply(lambda x: (x['Siniestros']/x['Primas']*100) if x['Primas']>0 else 0, axis=1)
        comp_geo['Resultado Técnico'] = comp_geo['Primas'] - comp_geo['Siniestros']
            
        st.dataframe(
            comp_geo.sort_values('Primas', ascending=False).head(50).style
                .format({'Primas':'${:,.0f}', 'Siniestros':'${:,.0f}', 'Resultado Técnico':'${:,.0f}', 'Siniestralidad':'{:.1f}%'})
                .background_gradient(subset=['Siniestralidad'], cmap='RdYlGn_r', vmin=0, vmax=100),
            use_container_width=True, hide_index=True
        )

# === TAB 2: PRODUCTOS ===
with tab2:
    st.subheader("Análisis por Ramo")
    if ver_desglose_anos and len(filtro_anios) > 1:
        st.markdown("##### Evolución Anual por Producto")
        df_pivot_ramo = utils.crear_vista_pivot_anos(df_filtrado, 'Ramo', metrica_view)
        cols_vals = [c for c in df_pivot_ramo.columns if c != 'Ramo']
        st.dataframe(
            df_pivot_ramo.style.format({col: '${:,.0f}' for col in cols_vals})
            .background_gradient(subset=['TOTAL CONSOLIDADO'], cmap='Greens'),
            use_container_width=True, hide_index=True
        )
    else:
        ramo_df = df_filtrado.groupby('Ramo')[['Primas', 'Siniestros']].sum().reset_index()
        ramo_df['Ratio'] = ramo_df.apply(lambda x: (x['Siniestros']/x['Primas']*100) if x['Primas']>0 else 0, axis=1)
            
        fig_bar = px.bar(ramo_df.sort_values('Primas', ascending=False), 
                           x='Ramo', y='Primas', color='Ratio',
                           color_continuous_scale='RdYlGn_r', 
                           title="Ramos: Volumen y Siniestralidad")
        st.plotly_chart(fig_bar, use_container_width=True)

# === TAB 3: PDF (Usando Utils) ===
with tab3:
    st.header("🧠 Generador de Informe de Conquista 2026")
    st.markdown("Este módulo utiliza **GPT-4** para redactar el plan estratégico basado en los datos filtrados.")
        
    c_ai1, c_ai2 = st.columns([2, 1])
    with c_ai1:
        foco = st.text_area("🎯 Instrucción Especial (Opcional)", placeholder="Ej: Enfocarme en el crecimiento en México y reducir siniestralidad en Autos...")
    with c_ai2:
        st.write("")
        st.write("")
        btn_gen = st.button("🔥 GENERAR INFORME PDF", type="primary")
        
    if btn_gen:
        api_key = utils.get_api_key()
        if not api_key:
            st.error("⚠️ Error: No se encontró la API KEY en las variables de entorno.")
        else:
            with st.status("🛠️ Fabricando tu Plan...", expanded=True) as status:
                # Preparar datos para el prompt
                pais_analisis = df_filtrado.groupby('País')[['Primas', 'Siniestros']].sum().reset_index()
                top_paises = pais_analisis.sort_values('Primas', ascending=False).head(3)['País'].tolist()
                    
                prompt_user = (
                    f"Actúa como consultor estratégico senior. Datos Clave del Dashboard actual: "
                    f"Primas Totales ${primas_tot/1e9:.2f}B USD. Siniestralidad Global {ratio_global:.1f}%. "
                    f"Top Mercados actuales: {', '.join(top_paises)}. "
                    f"Instrucción Adicional del Usuario: {foco}. "
                    "Escribe un diagnóstico ejecutivo de 3 párrafos y 3 estrategias puntuales (bullet points) para el plan 2026. "
                    "Sé directo y profesional."
                )
                    
                status.write("🧠 Redactando estrategia con IA...")
                try:
                    client = OpenAI(api_key=api_key)
                    resp = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role": "user", "content": prompt_user}]
                    )
                    texto_estrategia = resp.choices[0].message.content
                        
                    status.write("📄 Ensamblando PDF profesional...")
                    pdf = utils.UltimatePDF()
                    pdf.cover_page("PLAN DE DOMINACIÓN 2026", "ESTRATEGIA PARA LA EXPANSIÓN")
                    pdf.add_page()
                    pdf.section_title("1. TABLERO DE CONTROL (KPIs)")
                        
                    # Cajas de métricas en el PDF
                    current_y = pdf.get_y()
                    pdf.add_metric_box("PRIMAS (USD)", f"${primas_tot/1e9:.2f}B", 15, current_y + 5)
                    pdf.add_metric_box("SINIESTRALIDAD", f"{ratio_global:.1f}%", 70, current_y + 5)
                    pdf.add_metric_box("RESULTADO TÉC.", f"${res_tec/1e9:.2f}B", 125, current_y + 5)
                        
                    pdf.ln(40)
                    pdf.section_title("2. ESTRATEGIA GENERADA POR IA")
                    pdf.chapter_body(texto_estrategia)
                        
                    pdf_bytes = bytes(pdf.output(dest='S'))
                        
                    status.update(label="✅ Informe listo.", state="complete", expanded=False)
                    st.download_button("📥 DESCARGAR PDF", pdf_bytes, "Plan_2026.pdf", "application/pdf", type="primary")
                    
                except Exception as e:
                    st.error(f"Error en generación: {e}")

# === TAB 4: PROFUNDIZACIÓN ===
with tab4:
    st.header("Análisis de Profundización Total")
    comp_deep = df_filtrado.groupby('Compañía')[['Primas','Siniestros']].sum().reset_index()
        
    # Filtro para quitar compañías minúsculas que ensucian el gráfico
    comp_deep = comp_deep[comp_deep['Primas'] > 0]
        
    c_d1, c_d2 = st.columns([3,1])
    with c_d1:
        if not comp_deep.empty:
            fig_scat_deep = px.scatter(comp_deep, x="Primas", y="Siniestros", 
                                         hover_name="Compañía",
                                         size="Primas",
                                         title="Correlación Primas vs Siniestros por Empresa", 
                                         trendline="ols")
            st.plotly_chart(fig_scat_deep, use_container_width=True)
        else:
            st.info("No hay datos suficientes para el gráfico de dispersión.")
                
    with c_d2:
        st.markdown("**Top Riesgos (Mayor Siniestralidad Absoluta)**")
        top_riesgos = comp_deep.sort_values("Siniestros", ascending=False).head(10)
        st.dataframe(top_riesgos[['Compañía', 'Siniestros']].style.format({'Siniestros':'${:,.0f}'}), 
                     hide_index=True, use_container_width=True)