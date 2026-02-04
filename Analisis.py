import streamlit as st
import pandas as pd
import plotly.express as px
import datetime
import os
from openai import OpenAI

# Importamos tu cerebro robusto
import utils

api_key = utils.get_api_key()

# ==========================================
# 0. CONFIGURACIÓN INICIAL & ESTILOS PREMIUM
# ==========================================
st.set_page_config(
    page_title="ALSUM 2026 | Strategic Command", 
    layout="wide", 
    page_icon="🦅",
    initial_sidebar_state="expanded"
)

# CSS Profesional para que se vea como una App de alto nivel
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap');
    html, body, [class*="css"]  { font-family: 'Roboto', sans-serif; }
    
    /* Encabezados Azules */
    h1, h2, h3 { color: #004A8F; font-weight: 700; }
    
    /* Tarjetas de Métricas (KPIs) */
    div[data-testid="metric-container"] {
        background-color: #FFFFFF;
        border: 1px solid #E0E0E0;
        padding: 15px;
        border-radius: 8px;
        border-left: 6px solid #004A8F;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        transition: transform 0.2s;
    }
    div[data-testid="metric-container"]:hover {
        transform: scale(1.02);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    /* Botones Premium */
    div.stButton > button {
        background: linear-gradient(135deg, #004A8F 0%, #002a52 100%);
        color: white; 
        font-weight: bold;
        border-radius: 8px; 
        border: none;
        height: 50px;
        box-shadow: 0 4px 6px rgba(0, 74, 143, 0.3);
        transition: all 0.3s ease;
    }
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 74, 143, 0.4);
    }
    
    /* Tablas más limpias */
    .dataframe { font-size: 12px !important; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. CARGA DE DATOS ROBUSTA
# ==========================================
# Apuntamos al CSV para velocidad máxima
DATA_FILE = "plan_2026.xlsx"
FULL_PATH = utils.get_file_path(DATA_FILE)

with st.spinner('🚀 Inicializando Motor de Inteligencia de Negocios...'):
    # Especifica el nombre de la hoja, por ejemplo "Plan2026"
    df_final, error = utils.load_plan_accion_procesado(FULL_PATH, sheet_name="plan_2026")

# Manejo de Errores Críticos
if error:
    st.error(f"❌ Error Crítico: {error}")
    st.info("Asegúrate de haber guardado el Excel como 'plan_2026.csv' (Delimitado por comas).")
    st.stop()
elif df_final is None or df_final.empty:
    st.warning("⚠️ El archivo se cargó pero parece estar vacío.")
    st.stop()

# ==========================================
# 3. BARRA LATERAL (FILTROS MACRO)
# ==========================================
st.sidebar.image("https://alsum.co/wp-content/uploads/2018/06/Logo-Alsum-Web.png", width=180)
st.sidebar.markdown("---")
st.sidebar.header("🌍 Filtros Globales")

# Generación dinámica de filtros basada en columnas disponibles
df_filtrado = df_final.copy()

# Filtro AÑO
if 'Año' in df_final.columns:
    # Solo mostrar los años de estudio
    anios_estudio = [2022, 2023, 2024, 2025]
    anios_disp = [a for a in anios_estudio if a in df_final['Año'].unique()]
    filtro_anios = st.sidebar.multiselect("📅 Año Fiscal", anios_disp, default=anios_disp)
    # Filtrar el DataFrame solo por esos años
    df_filtrado = df_filtrado[df_filtrado['Año'].isin(filtro_anios)]
else:
    filtro_anios = []

# Filtro PAÍS
if 'País' in df_final.columns:
    paises_disp = sorted(df_final['País'].unique())
    filtro_paises = st.sidebar.multiselect("🌎 País / Región", paises_disp, default=paises_disp)
    if filtro_paises:
        df_filtrado = df_filtrado[df_filtrado['País'].isin(filtro_paises)]

# Filtro AFILIACIÓN
if 'AFILIADO' in df_final.columns:
    filtro_afiliado = st.sidebar.radio("💎 Estado Afiliación", ["Todos", "Afiliados", "No afiliados"])
    if filtro_afiliado == "Afiliados":
        df_filtrado = df_filtrado[df_filtrado['AFILIADO'] == 'AFILIADO']
    elif filtro_afiliado == "No afiliados":
        df_filtrado = df_filtrado[df_filtrado['AFILIADO'] == 'NO AFILIADO']

# Filtro RAMO (ahora muestra todos los ramos, incluyendo "No reporta")
if 'Ramo' in df_final.columns:
    # Excluir portuarios y petroleros (cualquier mayúscula/minúscula)
    ramos_excluir = ['riesgos portuarios', 'riesgos petroleros']
    ramos_disp = sorted([
        r for r in df_final['Ramo'].dropna().unique()
        if r.strip().lower() not in ramos_excluir
    ])
    filtro_ramos = st.sidebar.multiselect("📦 Ramo / Producto", ramos_disp, default=ramos_disp)
    if filtro_ramos:
        df_filtrado = df_filtrado[df_filtrado['Ramo'].isin(filtro_ramos)]

# Filtro CATEGORÍA
if 'Categoria' in df_final.columns:
    tipos_disp = sorted(df_final['Categoria'].dropna().unique())
    filtro_tipo = st.sidebar.multiselect("💠 Tipo de Afiliado", tipos_disp, default=tipos_disp)
    if filtro_tipo:
        df_filtrado = df_filtrado[df_filtrado['Categoria'].isin(filtro_tipo)]

# ==========================================
# 4. ÁREA PRINCIPAL
# ==========================================
st.title(f"🚀 Dashboard Estratégico {datetime.date.today().year}")
st.markdown("Visión 360° del desempeño comercial y técnico.")

# Filtros de Profundización (Expander)
with st.expander("🛠️ Herramientas de Análisis Profundo", expanded=False):
    c_f1, c_f2 = st.columns([2, 1])
    with c_f1:
        if 'Compañía' in df_final.columns:
            comps = sorted(df_filtrado['Compañía'].unique())
            sel_comp = st.multiselect("Filtrar por Compañía específica:", comps)
            if sel_comp:
                df_filtrado = df_filtrado[df_filtrado['Compañía'].isin(sel_comp)]
    # Elimina el bloque de métricas
    # with c_f2:
    #     metrica_focus = st.selectbox(
    #         "Métrica para Tablas:",
    #         ["Primas", "Siniestros", "Resultado Técnico"],
    #         index=0
    #     )

# Verificar si quedaron datos después de filtrar
if df_filtrado.empty:
    st.warning("⚠️ No hay datos que coincidan con tus filtros. Intenta ampliar la selección.")
    st.stop()

# --- KPIs GLOBALES (ENCABEZADO) ---
primas_tot = df_filtrado['Primas'].sum()
siniestros_tot = df_filtrado['Siniestros'].sum()
res_tec = df_filtrado['Resultado Técnico'].sum()
ratio_global = (siniestros_tot / primas_tot * 100) if primas_tot > 0 else 0.0

# NUEVO: Suma de "No reporta"
primas_no_reporta = df_filtrado[df_filtrado['Ramo'].str.lower() == 'no reporta']['Primas'].sum() if 'Ramo' in df_filtrado.columns else 0

# === BLOQUE DE MANEJO DE "NO REPORTA" ===

# 1. Separar registros "No reporta" (Tipo y Ramo)
mask_no_reporta = (
    (df_filtrado['Tipo'].str.lower() == 'no reporta') &
    (df_filtrado['Ramo'].str.lower() == 'no reporta')
)
df_no_reporta = df_filtrado[mask_no_reporta]
df_normales = df_filtrado[~mask_no_reporta]

# 2. KPIs (solo registros normales)
primas_tot = df_normales['Primas'].sum()
siniestros_tot = df_normales['Siniestros'].sum()
res_tec = df_normales['Resultado Técnico'].sum()
ratio_global = (siniestros_tot / primas_tot * 100) if primas_tot > 0 else 0.0

# 3. KPI "No reporta" (suma de USD, Primas y Siniestros si existen)
no_reporta_total = 0
if not df_no_reporta.empty:
    # Suma todas las columnas numéricas relevantes
    cols_sum = [col for col in ['USD', 'Primas', 'Siniestros'] if col in df_no_reporta.columns]
    no_reporta_total = df_no_reporta[cols_sum].sum().sum()

# 4. KPIs en pantalla
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("💰 Primas Totales (USD)", f"${primas_tot:,.0f}", delta="Volumen")
k2.metric("🔥 Siniestros Totales", f"${siniestros_tot:,.0f}", delta="Costo", delta_color="inverse")
k3.metric("📉 Siniestralidad", f"{ratio_global:.1f}%", delta=f"{65-ratio_global:.1f}% vs Meta (65%)")
k4.metric("📈 Resultado Técnico", f"${res_tec:,.0f}")
k5.metric("No Reporta (USD)", f"${no_reporta_total:,.0f}")

st.markdown("---")

# ==========================================
# 5. PESTAÑAS DETALLADAS
# ==========================================
tab1, tab2, tab3, tab4 = st.tabs([
    "🗺️ Mapa & Territorio", 
    "📦 Productos (Ramos)", 
    "🤖 Generador IA (PDF)", 
    "🔬 Profundización (Data)"
])

# === TAB 1: GEOGRÁFICO ===
with tab1:
    st.subheader("Análisis de Mercados")
    
    # Preparar datos
    pais_df = df_filtrado.groupby('País')[['Primas', 'Siniestros']].sum().reset_index()
    pais_df['Siniestralidad'] = (pais_df['Siniestros'] / pais_df['Primas'] * 100).fillna(0)
    # Antes de graficar, asegúrate que 'Primas' sea >= 0
    pais_df['Primas_plot'] = pais_df['Primas'].abs()
    
    c1, c2 = st.columns([2, 1])
    
    with c1:
        # Gráfico de Burbujas (Scatter)
        fig_map = px.scatter(
            pais_df, 
            x='Primas', 
            y='Siniestralidad',
            size='Primas_plot',  # Usar la columna corregida
            color='País',
            hover_name='País',
            title="Matriz de Desempeño: Volumen vs Rentabilidad",
            labels={'Primas': 'Volumen de Primas (USD)', 'Siniestralidad': '% Siniestralidad'}
        )
        # Línea de referencia del 65%
        fig_map.add_hline(y=65, line_dash="dash", line_color="red", annotation_text="Límite Rentable (65%)")
        st.plotly_chart(fig_map, use_container_width=True)
        
    with c2:
        st.markdown("### Top Mercados")
        # Tabla estilizada
        st.dataframe(
            pais_df.sort_values('Primas', ascending=False)
            .style.format({'Primas': '${:,.0f}', 'Siniestros': '${:,.0f}', 'Siniestralidad': '{:.1f}%'})
            .background_gradient(subset=['Siniestralidad'], cmap='RdYlGn_r', vmin=40, vmax=100),
            use_container_width=True,
            hide_index=True
        )

    st.markdown("### 🏆 Ranking de Compañías")

    # Agrupar y mostrar todas las columnas relevantes
    pivot_comp = df_normales.groupby(['Compañía', 'País']).agg({
        'Primas': 'sum',
        'Siniestros': 'sum',
        'Resultado Técnico': 'sum'
    }).reset_index()
    pivot_comp['Siniestralidad'] = (pivot_comp['Siniestros'] / pivot_comp['Primas'] * 100).fillna(0)
    pivot_comp['Compañía'] = pivot_comp['Compañía'].astype(str).str.upper().str.strip()

    # Suma de "No reporta" por compañía
    if not df_no_reporta.empty:
        no_reporta_sum = df_no_reporta.groupby(['Compañía', 'País'])[['USD', 'Primas', 'Siniestros']].sum().sum(axis=1).reset_index(name='No reporta')
        pivot_comp = pd.merge(pivot_comp, no_reporta_sum, on=['Compañía', 'País'], how='left')
        pivot_comp['No reporta'] = pivot_comp['No reporta'].fillna(0)
    else:
        pivot_comp['No reporta'] = 0

    # Formateo dinámico de columnas numéricas
    st.dataframe(
        pivot_comp.sort_values('Primas', ascending=False).style
            .format({
                'Primas': '${:,.0f}',
                'Siniestros': '${:,.0f}',
                'Resultado Técnico': '${:,.0f}',
                'Siniestralidad': '{:,.0f}%',
                'No reporta': '${:,.0f}'
            }),
        use_container_width=True,
        hide_index=True
    )

    # Ejemplo debajo de la gráfica de burbujas en TAB 1
    contexto = pais_df.head(10).to_string()
    prompt = "Analiza la gráfica de matriz de desempeño de países."
    analisis = utils.analisis_ia_3_puntos(api_key, prompt, contexto)
    st.info(analisis)

    # Ejemplo debajo de la tabla de compañías
    contexto = pivot_comp.head(10).to_string()
    prompt = "Analiza la tabla de ranking de compañías."
    analisis = utils.analisis_ia_3_puntos(api_key, prompt, contexto)
    st.info(analisis)

# === TAB 2: PRODUCTOS ===
with tab2:
    st.subheader("Desempeño por Ramo de Negocio")
    
    ramo_df = df_filtrado.groupby('Ramo')[['Primas', 'Siniestros']].sum().reset_index()
    ramo_df['Ratio'] = (ramo_df['Siniestros'] / ramo_df['Primas'] * 100).fillna(0)
    
    # Gráfico de Barras con color por Siniestralidad
    fig_bar = px.bar(
        ramo_df.sort_values('Primas', ascending=False),
        x='Ramo',
        y='Primas',
        color='Ratio',
        color_continuous_scale='RdYlGn_r',
        title="Volumen por Ramo (Color = Siniestralidad)",
        text_auto='.2s'
    )
    st.plotly_chart(fig_bar, use_container_width=True)
    
    st.markdown("#### Detalle Anual por Ramo")
    pivot_ramo = utils.crear_vista_pivot_anos(df_filtrado, 'Ramo', valor='Primas')
    st.dataframe(
    pivot_ramo.style.format({
        col: '${:,.0f}' if 'TOTAL' in col or 'Primas' in col or 'Siniestros' in col else '{:,.0f}%'
        for col in pivot_ramo.columns if col != 'Ramo'
    }),
    use_container_width=True
    )

    # --- Participación de Afiliados vs No Afiliados vs No reporta ---
    if 'AFILIADO' in df_normales.columns and 'Primas' in df_normales.columns:
        total_primas = df_normales['Primas'].sum()
        afiliados = df_normales[df_normales['AFILIADO'] == 'AFILIADO']['Primas'].sum()
        no_afiliados = df_normales[df_normales['AFILIADO'] == 'NO AFILIADO']['Primas'].sum()
        # Primas "No reporta" (de los registros no reporta)
        no_reporta_afiliados = df_no_reporta['USD'].sum() if 'USD' in df_no_reporta.columns else 0
        labels = ['Afiliados', 'No Afiliados', 'No reporta']
        values = [afiliados, no_afiliados, no_reporta_afiliados]
        fig_pie = px.pie(
            names=labels,
            values=values,
            title="Participación de Primas: Afiliados vs No Afiliados vs No reporta",
            color=labels,
            color_discrete_map={'Afiliados': '#004A8F', 'No Afiliados': '#B0B0B0', 'No reporta': '#FFB347'}
        )
        st.plotly_chart(fig_pie, use_container_width=True)
        st.info(f"**Afiliados:** {afiliados/total_primas:.1%} | **No Afiliados:** {no_afiliados/total_primas:.1%} | **No reporta:** {no_reporta_afiliados/(total_primas+no_reporta_afiliados):.1%}")
    else:
        st.warning("No se encontraron columnas 'AFILIADO' y 'Primas' para calcular la participación.")

    # --- Participación por Tipo de Afiliado (Miembro/Asociado) ---
    if 'Categoria' in df_filtrado.columns and 'Primas' in df_filtrado.columns:
        tipo_part = df_filtrado.groupby('Categoria')['Primas'].sum().reset_index()
        fig_tipo = px.pie(
            tipo_part,
            names='Categoria',
            values='Primas',
            title="Participación por Tipo de Afiliado (Miembro/Asociado)",
            color='Categoria'
        )
        st.plotly_chart(fig_tipo, use_container_width=True)

        # Ejemplo debajo de la gráfica de participación por tipo de afiliado
        contexto = tipo_part.head(10).to_string()
        prompt = "Analiza la gráfica de participación por tipo de afiliado."
        analisis = utils.analisis_ia_3_puntos(api_key, prompt, contexto)
        st.info(analisis)

# === TAB 3: GENERADOR PDF IA ===
with tab3:
    st.header("🧠 Inteligencia Artificial - Generador de Informes")
    st.info("Este módulo utiliza GPT-4 para redactar un análisis estratégico basado en los datos filtrados y genera un PDF oficial.")
    
    col_ia_1, col_ia_2 = st.columns([3, 1])
    with col_ia_1:
        foco_ia = st.text_area("🎯 Instrucción para la IA:", height=100, 
                             placeholder="Ej: Enfocarse en el crecimiento excepcional de México y proponer estrategias para reducir la siniestralidad en Autos...")
    with col_ia_2:
        st.write("") # Espacio
        st.write("")
        btn_generar = st.button("✨ GENERAR INFORME PDF", type="primary")

    if btn_generar:
        api_key = utils.get_api_key()
        if not api_key:
            st.error("⚠️ No se detectó la API KEY de OpenAI en secretos o variables de entorno.")
        else:
            with st.status("🤖 Procesando Estrategia...", expanded=True) as status:
                try:
                    # 1. Preparar Contexto
                    status.write("Analizando datos del dashboard...")
                    top_paises = pais_df.sort_values('Primas', ascending=False).head(3)['País'].tolist()
                    contexto = (
                        f"Resumen Ejecutivo ALSUM 2026. "
                        f"Primas Totales: USD {primas_tot:,.0f}. "
                        f"Siniestralidad Global: {ratio_global:.1f}%. "
                        f"Resultado Técnico: USD {res_tec:,.0f}. "
                        f"Top Mercados: {', '.join(top_paises)}. "
                        f"Instrucción del usuario: {foco_ia}"
                    )
                    
                    # 2. Llamar a OpenAI
                    status.write("Consultando a GPT-4...")
                    client = OpenAI(api_key=api_key)
                    response = client.chat.completions.create(
                        model="gpt-4o-mini", # Modelo rápido y eficiente
                        messages=[
                            {"role": "system", "content": "Eres un consultor experto en seguros y reaseguros."},
                            {"role": "user", "content": f"Escribe un informe estratégico ejecutivo de 3 párrafos y 3 bullet points de acción basado en: {contexto}"}
                        ]
                    )
                    texto_ia = response.choices[0].message.content
                    
                    # 3. Generar PDF
                    status.write("Maquetando PDF Profesional...")
                    pdf = utils.UltimatePDF()
                    pdf.cover_page("PLAN ESTRATÉGICO 2026", "INFORME DE INTELIGENCIA DE MERCADO")
                    pdf.add_page()
                    pdf.section_title("1. RESUMEN DE KPIs")
                    
                    # Cajas de métricas en el PDF (simuladas con texto)
                    pdf.chapter_body(f"PRIMAS: USD {primas_tot:,.0f} | RATIO: {ratio_global:.1f}%")
                    pdf.ln(10)
                    
                    pdf.section_title("2. DIAGNÓSTICO ESTRATÉGICO (IA)")
                    pdf.chapter_body(texto_ia)
                    
                    # Convertir a bytes
                    pdf_bytes = bytes(pdf.output(dest='S'))
                    
                    status.update(label="✅ ¡Informe Listo!", state="complete", expanded=False)
                    
                    # Botón de Descarga
                    st.download_button(
                        label="📥 DESCARGAR PDF FINAL",
                        data=pdf_bytes,
                        file_name="Estrategia_ALSUM_2026.pdf",
                        mime="application/pdf",
                        type="primary"
                    )
                    
                except Exception as e:
                    st.error(f"Error durante la generación: {str(e)}")

# === TAB 4: PROFUNDIZACIÓN ===
with tab4:
    st.subheader("🔬 Data Lab")
    
    # Análisis de Dispersión
    comp_scatter = df_filtrado.groupby('Compañía')[['Primas', 'Siniestros']].sum().reset_index()
    # Filtramos compañías muy pequeñas para limpiar el gráfico
    comp_scatter = comp_scatter[comp_scatter['Primas'] > comp_scatter['Primas'].mean() * 0.1]
    
    c_d1, c_d2 = st.columns([3, 1])
    with c_d1:
        fig_deep = px.scatter(
            comp_scatter, 
            x="Primas", 
            y="Siniestros", 
            hover_name="Compañía", 
            trendline="ols", # Línea de tendencia
            title="Correlación Primas vs Siniestros (Detección de Anomalías)",
            color_discrete_sequence=["#004A8F"]
        )
        st.plotly_chart(fig_deep, use_container_width=True)
        
    with c_d2:
        st.markdown("**🚨 Top Riesgos (Siniestros Altos)**")
        st.dataframe(
            comp_scatter.sort_values("Siniestros", ascending=False).head(10)
            .style.format({'Siniestros': '${:,.0f}', 'Primas': '${:,.0f}'}),
            hide_index=True,
            use_container_width=True
        )
        
        # Ejemplo debajo de la tabla de top riesgos
        contexto = comp_scatter.sort_values("Siniestros", ascending=False).head(10).to_string()
        prompt = "Analiza la tabla de top riesgos por siniestros altos."
        analisis = utils.analisis_ia_3_puntos(api_key, prompt, contexto)
        st.info(analisis)
        
    st.markdown("### Tabla de Datos Completa")
    st.dataframe(
    df_filtrado.style.format({
        col: '${:,.0f}' for col in df_filtrado.select_dtypes(include='number').columns
    }),
    use_container_width=True
)

st.markdown("### 🌎 Empresas por País")

empresas_por_pais = df_filtrado.groupby('País')['Compañía'].nunique().reset_index()
empresas_por_pais.columns = ['País', 'Empresas']

st.dataframe(
    empresas_por_pais.sort_values('Empresas', ascending=False)
    .style.format({'Empresas': '{:,.0f}'}),
    use_container_width=True,
    hide_index=True
)

contexto = empresas_por_pais.head(10).to_string()
prompt = "Analiza la tabla de empresas por país."
analisis = utils.analisis_ia_3_puntos(api_key, prompt, contexto)
st.info(analisis)
