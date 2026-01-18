import streamlit as st
import pandas as pd
import plotly.express as px
from fpdf import FPDF
import tempfile
import os
import datetime
from openai import OpenAI

# ==========================================
# 1. CONFIGURACIÓN DE PÁGINA Y ESTILO VIP
# ==========================================
st.set_page_config(
    page_title="ALSUM 2026 | Strategic Command", 
    layout="wide", 
    page_icon="🦅",
    initial_sidebar_state="expanded"
)

# --- ESTILOS CSS PREMIUM ---
st.markdown("""
<style>
    /* Tipografía y Espaciado */
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap');
    
    html, body, [class*="css"]  {
        font-family: 'Roboto', sans-serif;
    }
    .main .block-container { padding-top: 1rem; padding-bottom: 3rem; }
    
    /* Encabezados */
    h1, h2, h3 { color: #004A8F; font-weight: 700; }
    
    /* Tarjetas de Métricas (KPIs) */
    div[data-testid="metric-container"] {
        background-color: #FFFFFF;
        border: 1px solid #E0E0E0;
        padding: 15px;
        border-radius: 8px;
        border-left: 6px solid #004A8F;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        transition: transform 0.2s;
    }
    div[data-testid="metric-container"]:hover {
        transform: scale(1.02);
    }
    
    /* Botones Premium */
    div.stButton > button {
        background: linear-gradient(135deg, #004A8F 0%, #002a52 100%);
        color: white; 
        font-size: 16px; 
        font-weight: bold;
        padding: 12px 28px; 
        border-radius: 8px; 
        border: none; 
        width: 100%;
        box-shadow: 0 4px 10px rgba(0, 74, 143, 0.3);
        transition: all 0.3s ease;
    }
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 15px rgba(0, 74, 143, 0.4);
    }
    
    /* Tablas */
    .dataframe { font-size: 13px !important; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. CLASE PARA GENERACIÓN DE PDF "ULTIMATE"
# ==========================================
class UltimatePDF(FPDF):
    def header(self):
        if self.page_no() > 1: # Sin header en portada
            self.set_font('Arial', 'B', 9)
            self.set_text_color(120, 120, 120)
            self.cell(0, 10, 'MEMORANDO ESTRATÉGICO CONFIDENCIAL - PLAN 2026', 0, 0, 'L')
            self.cell(0, 10, f'{datetime.date.today().strftime("%d/%m/%Y")}', 0, 1, 'R')
            self.set_draw_color(0, 74, 143)
            self.set_line_width(0.5)
            self.line(10, 20, 200, 20)
            self.ln(15)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f'Autor: Lina Marcela Contreras | Página {self.page_no()}', 0, 0, 'C')

    def cover_page(self, title, subtitle):
        self.add_page()
        # Fondo Azul Elegante
        self.set_fill_color(0, 74, 143) 
        self.rect(0, 0, 210, 297, 'F') 
        
        # Elementos Portada
        self.set_text_color(255, 255, 255)
        self.set_font('Arial', 'B', 45)
        self.ln(60)
        self.cell(0, 20, "ALSUM", 0, 1, 'C')
        
        self.set_font('Arial', '', 14)
        self.cell(0, 10, "INTELIGENCIA & ESTRATEGIA DE NEGOCIOS", 0, 1, 'C')
        
        self.set_draw_color(255, 255, 255)
        self.set_line_width(1)
        self.line(50, 110, 160, 110)
        
        self.ln(40)
        self.set_font('Arial', 'B', 32)
        self.multi_cell(0, 15, title, 0, 'C')
        
        self.ln(5)
        self.set_font('Arial', 'I', 18)
        self.multi_cell(0, 10, subtitle, 0, 'C')
        
        self.set_y(-60)
        self.set_font('Arial', '', 12)
        self.cell(0, 10, "Preparado exclusivamente por:", 0, 1, 'C')
        self.set_font('Arial', 'B', 14)
        self.cell(0, 10, "Lina Marcela Contreras", 0, 1, 'C')
        self.set_font('Arial', '', 11)
        self.cell(0, 10, f"Fecha de Emisión: {datetime.date.today().strftime('%B %d, %Y')}", 0, 1, 'C')

    def section_title(self, label):
        self.set_font('Arial', 'B', 16)
        self.set_text_color(0, 74, 143)
        self.cell(0, 10, label.upper(), 0, 1, 'L')
        self.ln(2)
        self.set_draw_color(200, 200, 200)
        self.set_line_width(0.2)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(8)

    def chapter_body(self, text):
        self.set_font('Arial', '', 11)
        self.set_text_color(40, 40, 40)
        self.multi_cell(0, 6, text)
        self.ln()

    def add_metric_box(self, label, value, x, y, bg_color=(245, 247, 250)):
        self.set_xy(x, y)
        self.set_fill_color(*bg_color)
        self.rect(x, y, 45, 28, 'F')
        self.set_draw_color(0, 74, 143)
        self.set_line_width(0.5)
        self.line(x, y, x, y+28) # Línea vertical izquierda decorativa
        
        self.set_xy(x+2, y+6)
        self.set_font('Arial', 'B', 8)
        self.set_text_color(100, 100, 100)
        self.cell(40, 5, label, 0, 2)
        
        self.set_font('Arial', 'B', 11)
        self.set_text_color(0, 0, 0)
        self.cell(40, 8, value, 0, 0)

    def create_table(self, df):
        # Header
        self.set_font('Arial', 'B', 9)
        self.set_fill_color(0, 74, 143)
        self.set_text_color(255, 255, 255)
        col_width = 190 / len(df.columns)
        
        for col in df.columns:
            self.cell(col_width, 8, str(col), 1, 0, 'C', 1)
        self.ln()
        
        # Rows
        self.set_font('Arial', '', 8)
        self.set_text_color(0, 0, 0)
        fill = False
        for i, row in df.iterrows():
            self.set_fill_color(240, 245, 255) if fill else self.set_fill_color(255, 255, 255)
            for item in row:
                txt = str(item)[:28] # Cortar texto largo
                self.cell(col_width, 7, txt, 1, 0, 'C', fill)
            self.ln()
            fill = not fill

# ==========================================
# 3. CARGA DE DATOS ROBUSTA
# ==========================================
@st.cache_data(show_spinner=False)
def load_data_universal(file):
    try:
        if file.name.endswith('.csv'):
            file.seek(0)
            df = pd.read_csv(
                file,
                sep=';',
                engine='python',
                usecols=range(10),
                on_bad_lines='skip',
                encoding='utf-8',
                header=0
            )
            df.columns = [c.strip() for c in df.columns]
        else:
            df = pd.read_excel(file, engine='openpyxl', header=0, usecols="A:J")
            df.columns = [c.strip() for c in df.columns]

        # Limpieza básica
        df['Compañía'] = df['Compañía'].astype(str).str.strip()
        if 'Subramo' in df.columns:
            df['Subramo'] = df['Subramo'].fillna('General')
        if 'Ramo' in df.columns:
            df['Ramo'] = df['Ramo'].fillna('Otros')

        # La columna USD ya está convertida - solo parseamos formato
        def parse_numero_latino(val):
            if pd.isna(val):
                return 0.0
            texto = str(val).strip()
            # Si es número directo (sin separadores), convertir
            try:
                return float(texto)
            except:
                # Si tiene separadores latinos . = miles, , = decimal
                texto = texto.replace('.', '').replace(',', '.')
                try:
                    return float(texto)
                except:
                    return 0.0

        df['USD'] = df['USD'].apply(parse_numero_latino)

        # Crear tabla pivote
        pivot_df = df.pivot_table(
            index=['País', 'Año', 'Compañía', 'Ramo', 'Subramo', 'AFILIADO'],
            columns='Tipo', values='USD', aggfunc='sum', fill_value=0
        ).reset_index()

        pivot_df.columns.name = None
        if 'Primas' not in pivot_df.columns:
            pivot_df['Primas'] = 0.0
        if 'Siniestros' not in pivot_df.columns:
            pivot_df['Siniestros'] = 0.0

        pivot_df['Siniestros'] = pivot_df['Siniestros'].abs()
        pivot_df['Siniestralidad'] = (
            pivot_df['Siniestros'] / pivot_df['Primas']
        ).replace([float('inf'), -float('inf')], 0) * 100
        pivot_df['Resultado Técnico'] = pivot_df['Primas'] - pivot_df['Siniestros']

        return pivot_df, None
    except Exception as e:
        return None, f"Error: {e}"

# ==========================================
# 4. INTERFAZ Y LÓGICA
# ==========================================

with st.sidebar:
    st.image("https://www.alsum.co/wp-content/uploads/2022/08/LOGO-ALSUM-BLANCO-1-1024x282.png", use_container_width=True)
    st.header("Centro de Mando")
    st.info("📊 ALSUM Intelligence System")
    
    # Cargador de archivo Excel
    uploaded_file = st.file_uploader("📂 Cargar base de datos", type=['xlsx', 'csv'], help="Sube el archivo Excel o CSV")

# --- CARGA ---
if uploaded_file is None:
    st.warning("⚠️ Por favor, carga el archivo desde el panel lateral para continuar.")
    st.stop()

try:
    with st.spinner('Inicializando protocolos de análisis...'):
        df_final, error = load_data_universal(uploaded_file)
except Exception as e:
    st.error(f"❌ Error al cargar archivo: {e}")
    st.stop()

if error:
    st.error(f"❌ {error}")
    st.stop()
elif df_final is not None:
    
    # --- KPIs ---
    escala = 1e9  # Miles de millones (Billions)
    primas_tot = df_final['Primas'].sum()
    siniestros_tot = df_final['Siniestros'].sum()
    ratio_global = (siniestros_tot / primas_tot) * 100 if primas_tot > 0 else 0
    res_tec = primas_tot - siniestros_tot

    st.title(f"🚀 Plan Estratégico & Comercial {datetime.date.today().year}")
    
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Volumen Primas", f"${primas_tot/escala:,.2f}B")
    k2.metric("Siniestros Totales", f"${siniestros_tot/escala:,.2f}B")
    k3.metric("Siniestralidad", f"{ratio_global:.1f}%", delta=f"{65-ratio_global:.1f}% vs Meta", delta_color="normal" if ratio_global < 65 else "inverse")
    k4.metric("Resultado Técnico", f"${res_tec/escala:,.2f}B")
    st.markdown("---")

    # --- PESTAÑAS CON LAZY LOADING ---
    tab1, tab2, tab3, tab4 = st.tabs(["🌎 Mapa de Analisis", "📦 Productos", "🧠 GENERADOR INFORME (PDF)", "🎯 Profundización"])

    # --- TAB 1: GEOGRÁFICO ---
    with tab1:
        @st.cache_data(show_spinner=False)
        def get_geo_data(df):
            anios_default = sorted(df['Año'].unique())
            return anios_default
        
        st.subheader("Análisis de Territorio")
        c1, c2 = st.columns([2, 1])
        
        anios_disponibles = get_geo_data(df_final)
        anios = st.multiselect("Filtrar Año", anios_disponibles, default=anios_disponibles)
        df_geo = df_final[df_final['Año'].isin(anios)]
        
        pais_df = df_geo.groupby('País')[['Primas', 'Siniestros']].sum().reset_index()
        pais_df['Siniestralidad'] = (pais_df['Siniestros']/pais_df['Primas'])*100
        pais_df['Primas_M'] = pais_df['Primas']/1e6

        with c1:
            fig = px.scatter(pais_df, x='Primas_M', y='Siniestralidad', 
                           size='Primas_M', color='País',
                           title="Matriz de Desempeño (Todos los Países)", size_max=60)
            fig.add_hline(y=65, line_dash="dash", line_color="red", annotation_text="Límite Riesgo")
            st.plotly_chart(fig, use_container_width=True)
            
        with c2:
            st.markdown("**Top Mercados**")
            st.dataframe(pais_df.sort_values('Primas', ascending=False)[['País','Primas','Siniestralidad']]
                         .style.format({'Primas':'${:,.0f}','Siniestralidad':'{:.1f}%'}), 
                         hide_index=True, use_container_width=True)
            
        st.subheader("Detalle por Compañía (Completo)")
        detail = df_geo.groupby(['País','Compañía'])['Primas'].sum().reset_index().sort_values(['País','Primas'], ascending=[True,False])
        st.dataframe(detail, use_container_width=True)

    # --- TAB 2: PRODUCTOS ---
    with tab2:
        @st.cache_data(show_spinner=False)
        def get_product_data(df):
            ramo_df = df.groupby('Ramo')[['Primas', 'Siniestros']].sum().reset_index()
            ramo_df['Ratio'] = (ramo_df['Siniestros']/ramo_df['Primas'])*100
            afi = df.groupby('AFILIADO')['Primas'].sum().reset_index()
            return ramo_df, afi
        
        st.subheader("Rentabilidad por Producto")
        ramo_df, afi = get_product_data(df_final)
        
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            fig_bar = px.bar(ramo_df.sort_values('Primas', ascending=False), 
                           x='Ramo', y='Primas', color='Ratio',
                           color_continuous_scale='RdYlGn_r', 
                           title="Todos los Ramos: Primas y Siniestralidad")
            st.plotly_chart(fig_bar, use_container_width=True)
        with col_p2:
            fig_pie = px.pie(afi, values='Primas', names='AFILIADO', 
                           title="Distribución Afiliados", 
                           color_discrete_map={'AFILIADO':'#004A8F', 'NO AFILIADO':'#BDBDBD'})
            st.plotly_chart(fig_pie, use_container_width=True)

    # --- TAB 3: GENERADOR PDF IA ---
    with tab3:
        st.header("🧠 Generador de Informe de Conquista 2026")
        st.info("Este módulo utiliza IA para redactar un Plan de Acción estratégico.")
        
        c_ai1, c_ai2 = st.columns([2, 1])
        with c_ai1:
            foco = st.text_area("🎯 Instrucción Especial (Opcional)", 
                              placeholder="Ej: Enfocarme en México y reducir siniestralidad...")
        
        with c_ai2:
            st.write("")
            st.write("")
            btn_gen = st.button("🔥 GENERAR INFORME MAESTRO", type="primary")

        if btn_gen:
            if "OPENAI_API_KEY" not in st.secrets:
                st.error("⚠️ Configura tu API KEY en Secrets.")
            else:
                with st.status("🛠️ Fabricando tu Plan de Conquista...", expanded=True) as status:
                    try:
                        status.write("🔍 Extrayendo inteligencia de mercado...")
                        pais_analisis = df_final.groupby('País')[['Primas', 'Siniestros']].sum().reset_index()
                        pais_analisis['Siniestralidad'] = (pais_analisis['Siniestros']/pais_analisis['Primas'])*100
                        
                        top_paises = pais_analisis.sort_values('Primas', ascending=False).head(3)['País'].tolist()
                        top_risk = pais_analisis.sort_values('Siniestralidad', ascending=False).head(3)['País'].tolist()
                        opportunities = pais_analisis[(pais_analisis['Siniestralidad'] < 50) & 
                                                     (pais_analisis['Primas'] > 1000000)]['País'].tolist()
                        
                        prompt_system = (
                            "Eres Lina Marcela Contreras, Estratega Comercial Senior en ALSUM. "
                            "Escribes un análisis directo y propositivo para la Gerencia General. "
                            "Tu objetivo: Presentar un plan claro para dominar el mercado en 2026."
                        )
                        
                        prompt_user = (
                            f"Datos Clave: Primas ${primas_tot/1e9:.2f}B USD. Siniestralidad {ratio_global:.1f}%. "
                            f"Mercados Grandes: {', '.join(top_paises)}. Mercados Riesgosos: {', '.join(top_risk)}. "
                            f"Oportunidades: {', '.join(opportunities)}. Instrucción: {foco}. "
                            "\n\nEscribe en 3 secciones: "
                            "1. DIAGNÓSTICO REALISTA (riesgos actuales) "
                            "2. ESTRATEGIA DE ATAQUE 2026 (cómo traer más dinero) "
                            "3. COMPROMISO DE CIERRE (mensaje contundente)"
                        )

                        status.write("🧠 Redactando estrategia (OpenAI)...")
                        client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
                        resp = client.chat.completions.create(
                            model="gpt-4o-mini",
                            messages=[
                                {"role": "system", "content": prompt_system},
                                {"role": "user", "content": prompt_user}
                            ],
                            temperature=0.5
                        )
                        texto_estrategia = resp.choices[0].message.content

                        status.write("📄 Ensamblando PDF...")
                        pdf = UltimatePDF()
                        pdf.cover_page("PLAN DE DOMINACIÓN 2026", "ESTRATEGIA PARA LA EXPANSIÓN DE MERCADO")
                        
                        pdf.add_page()
                        pdf.section_title("1. TABLERO DE CONTROL (KPIs)")
                        y_kpi = pdf.get_y() + 5
                        pdf.add_metric_box("PRIMAS (B)", f"${primas_tot/1e9:.2f}", 15, y_kpi)
                        pdf.add_metric_box("SINIESTROS (B)", f"${siniestros_tot/1e9:.2f}", 65, y_kpi)
                        pdf.add_metric_box("SINIESTRALIDAD", f"{ratio_global:.1f}%", 115, y_kpi, 
                                         bg_color=(255, 230, 230) if ratio_global > 65 else (230, 255, 230))
                        pdf.ln(40)
                        
                        pdf.section_title("2. CARTA ESTRATÉGICA A LA GERENCIA")
                        pdf.chapter_body(texto_estrategia)

                        pdf.add_page()
                        pdf.section_title("3. LISTA DE OBJETIVOS (TOP PROSPECTS)")
                        pdf.chapter_body("Cuentas clave con alto volumen y rentabilidad saludable (prioridad Q1 2026):")
                        pdf.ln(5)
                        
                        prospects = df_final.groupby(['País', 'Compañía'])[['Primas','Siniestros']].sum().reset_index()
                        prospects['Ratio'] = (prospects['Siniestros']/prospects['Primas'])*100
                        targets = prospects[(prospects['Primas']>50000) & 
                                          (prospects['Ratio']<60)].sort_values('Primas', ascending=False)
                        
                        tabla_pdf = targets.copy()
                        tabla_pdf['Primas'] = tabla_pdf['Primas'].apply(lambda x: f"${x/1e6:.1f}M")
                        tabla_pdf['Ratio'] = tabla_pdf['Ratio'].apply(lambda x: f"{x:.1f}%")
                        pdf.create_table(tabla_pdf[['País', 'Compañía', 'Primas', 'Ratio']])
                        
                        pdf_bytes = bytes(pdf.output(dest='S'))
                        
                        status.update(label="✅ ¡MISIÓN CUMPLIDA! Tu informe está listo.", 
                                    state="complete", expanded=False)
                        st.balloons()
                        
                        st.download_button(
                            label="📥 DESCARGAR PLAN MAESTRO PDF",
                            data=pdf_bytes,
                            file_name="Plan_Estrategico_2026.pdf",
                            mime="application/pdf",
                            type="primary"
                        )

                    except Exception as e:
                        st.error(f"Error crítico: {e}")

    # --- TAB 4: PROFUNDIZACIÓN ---
    with tab4:
        @st.cache_data(show_spinner=False)
        def get_deep_analysis(df):
            comp = df.groupby(['Compañía'])[['Primas','Siniestros']].sum().reset_index()
            comp['Ratio'] = (comp['Siniestros']/comp['Primas'])*100
            
            heat = df.groupby(['Ramo','AFILIADO'])[['Primas','Siniestros']].sum().reset_index()
            heat['Ratio'] = (heat['Siniestros']/heat['Primas'])*100
            
            trend = df.groupby('Año')[['Primas','Siniestros']].sum().reset_index()
            
            return comp, heat, trend
        
        st.header("Análisis de Profundización Total")

        # Filtro de afiliación
        filtro_af = st.radio(
            "Filtrar por condición de afiliación",
            ["Todos", "Afiliados", "No afiliados"],
            horizontal=True
        )
        if filtro_af == "Afiliados":
            df_focus = df_final[df_final['AFILIADO'] == 'AFILIADO']
        elif filtro_af == "No afiliados":
            df_focus = df_final[df_final['AFILIADO'] == 'NO AFILIADO']
        else:
            df_focus = df_final

        if df_focus.empty:
            st.warning("No hay datos para el filtro seleccionado.")
            st.stop()

        comp, heat, trend = get_deep_analysis(df_focus)
        
        st.subheader("Todas las Compañías Globales")
        fig_comp = px.bar(comp.sort_values('Primas', ascending=False), 
                         x='Primas', y='Compañía', orientation='h', 
                         color='Ratio', color_continuous_scale='RdYlGn_r')
        fig_comp.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_comp, use_container_width=True)
        st.markdown("- **Qué ves:** barras horizontales con el volumen de primas por compañía. El color indica la siniestralidad (%) — verde es menor siniestralidad, rojo es mayor.\n- **Para decidir rápido:** prioriza compañías con barras largas (más primas) y colores verdes/amarillos (mejor relación riesgo-retorno).")
        
        c_deep1, c_deep2 = st.columns(2)
        
        with c_deep1:
            st.subheader("Mapa de Calor (Riesgo)")
            fig_heat = px.density_heatmap(heat, x='AFILIADO', y='Ramo', z='Ratio', 
                                         color_continuous_scale='RdYlGn_r')
            st.plotly_chart(fig_heat, use_container_width=True)
            st.markdown("- **Qué ves:** calor por ramo y condición de afiliación. Colores rojos indican ramos con mayor siniestralidad.\n- **Para decidir rápido:** enfócate en ramos verdes/amarillos y mitiga o reprecifica los rojos.")
            
        with c_deep2:
            st.subheader("Estructura de Cartera (Completa)")
            fig_tree = px.treemap(df_focus[df_focus['Primas']>0], 
                                path=[px.Constant("Global"), 'País', 'Ramo'], 
                                values='Primas')
            st.plotly_chart(fig_tree, use_container_width=True)
            st.markdown("- **Qué ves:** tamaño relativo de la cartera por país y ramo (áreas más grandes = mayor volumen de primas).\n- **Para decidir rápido:** identifica países/ramos dominantes y detecta huecos de penetración (áreas pequeñas) para crecer.")
            
        st.subheader("Evolución Histórica")
        fig_line = px.line(trend, x='Año', y=['Primas','Siniestros'], markers=True)
        st.plotly_chart(fig_line, use_container_width=True)
        st.markdown("- **Qué ves:** tendencia anual de primas y siniestros.\n- **Para decidir rápido:** busca convergencia o cruces; si siniestros se acercan a primas, urge ajustar suscripción y precios.")
        
        st.markdown("---")
        st.subheader("🎯 Radar de Oportunidades")
        cf1, cf2, cf3 = st.columns(3)
        with cf1: 
            p_radar = st.multiselect("País Objetivo", sorted(df_focus['País'].unique()), key='rd_p')
        with cf2: 
            r_radar = st.slider("Máximo Riesgo (%)", 0, 100, 60, key='rd_r')
        with cf3: 
            v_radar = st.number_input("Mínimo Primas USD", value=10000, step=10000, key='rd_v')
        
        base_radar = df_focus.groupby(['País','Compañía','AFILIADO'])[['Primas','Siniestros']].sum().reset_index()
        base_radar['Ratio'] = (base_radar['Siniestros']/base_radar['Primas'])*100
        
        mask = (base_radar['Primas']>=v_radar) & (base_radar['Ratio']<=r_radar)
        if p_radar: 
            mask = mask & (base_radar['País'].isin(p_radar))
        
        final_radar = base_radar[mask].sort_values('Primas', ascending=False)
        st.dataframe(final_radar.style.format({'Primas':'${:,.0f}','Siniestros':'${:,.0f}','Ratio':'{:.1f}%'})
                     .background_gradient(subset=['Ratio'], cmap='RdYlGn_r'), 
                     use_container_width=True)
        st.markdown("- **Qué ves:** tabla filtrable de cuentas objetivo con buen volumen y siniestralidad bajo el umbral.\n- **Para decidir rápido:** ordena por primas y ejecuta sobre las primeras filas (mejor volumen con riesgo controlado).")