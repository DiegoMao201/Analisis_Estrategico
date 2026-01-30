import streamlit as st
import pandas as pd
import plotly.express as px
from fpdf import FPDF
import tempfile
import os
import datetime
from openai import OpenAI

# ==========================================
# 0. CONFIGURACIÓN INICIAL
# ==========================================
# Ruta de datos por defecto
DATA_REPO_PATH = os.path.join(os.path.dirname(__file__), "Plan de accion 2026.xlsx")

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
    /* Tipografía y General */
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap');
    
    html, body, [class*="css"]  { font-family: 'Roboto', sans-serif; }
    .main .block-container { padding-top: 1rem; padding-bottom: 3rem; }
    
    /* Encabezados */
    h1, h2, h3 { color: #004A8F; font-weight: 700; }
    
    /* Tarjetas de Métricas */
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
    
    /* CAJA DE ANÁLISIS IA (CSS) */
    .ai-box {
        background-color: #f0f7ff;
        border-left: 4px solid #004A8F;
        padding: 15px;
        border-radius: 5px;
        margin-top: 10px;
        margin-bottom: 25px;
        font-size: 0.95rem;
        color: #1e3a5f;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .ai-title {
        font-weight: bold;
        color: #004A8F;
        margin-bottom: 5px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    /* Botones */
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
# 2. MOTORES IA Y CLASE PDF
# ==========================================

def get_api_key():
    """Recupera la API Key de st.secrets o variables de entorno"""
    try:
        return st.secrets["OPENAI_API_KEY"]
    except:
        try:
            return os.environ["OPENAI_API_KEY"]
        except:
            return None

def generar_analisis_ia(contexto_datos, tipo_grafico):
    """
    Motor de Análisis IA: Toma datos y devuelve insights estratégicos.
    """
    api_key = get_api_key()
    
    if not api_key:
        return "⚠️ **IA Desactivada:** No se detectó la variable de entorno OPENAI_API_KEY."
    
    try:
        client = OpenAI(api_key=api_key)
        prompt_system = (
            "Eres un Consultor Estratégico Senior de Seguros (ALSUM). "
            "Analizas datos para la Junta Directiva. Sé breve, directo y perspicaz."
        )
        prompt_user = (
            f"Analiza estos datos de un {tipo_grafico}:\n"
            f"{contexto_datos}\n\n"
            "Responde con este formato Markdown exacto:\n"
            "**🔍 Qué muestra:** (1 frase describiendo la visualización)\n"
            "**📊 Interpretación:** (Cómo leer los datos, qué destaca)\n"
            "**🚀 Acción:** (1 recomendación de negocio imperativa)"
        )

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": prompt_system},
                {"role": "user", "content": prompt_user}
            ],
            temperature=0.3
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"❌ Error de conexión IA: {str(e)}"

class UltimatePDF(FPDF):
    def header(self):
        if self.page_no() > 1:
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
        self.cell(0, 10, f'Autor: ALSUM Intelligence System | Página {self.page_no()}', 0, 0, 'C')

    def cover_page(self, title, subtitle):
        self.add_page()
        self.set_fill_color(0, 74, 143) 
        self.rect(0, 0, 210, 297, 'F') 
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

    def section_title(self, label):
        self.set_font('Arial', 'B', 16)
        self.set_text_color(0, 74, 143)
        self.cell(0, 10, label.upper(), 0, 1, 'L')
        self.ln(2)
        self.set_draw_color(200, 200, 200)
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
        self.line(x, y, x, y+28)
        self.set_xy(x+2, y+6)
        self.set_font('Arial', 'B', 8)
        self.set_text_color(100, 100, 100)
        self.cell(40, 5, label, 0, 2)
        self.set_font('Arial', 'B', 11)
        self.set_text_color(0, 0, 0)
        self.cell(40, 8, value, 0, 0)

    def create_table(self, df):
        self.set_font('Arial', 'B', 9)
        self.set_fill_color(0, 74, 143)
        self.set_text_color(255, 255, 255)
        col_width = 190 / len(df.columns)
        for col in df.columns:
            self.cell(col_width, 8, str(col), 1, 0, 'C', 1)
        self.ln()
        self.set_font('Arial', '', 8)
        self.set_text_color(0, 0, 0)
        fill = False
        for i, row in df.iterrows():
            self.set_fill_color(240, 245, 255) if fill else self.set_fill_color(255, 255, 255)
            for item in row:
                txt = str(item)[:28]
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
            df = pd.read_csv(file, sep=';', engine='python', usecols=range(10), on_bad_lines='skip', encoding='utf-8', header=0)
            df.columns = [c.strip() for c in df.columns]
        else:
            df = pd.read_excel(file, engine='openpyxl', header=0, usecols="A:J")
            df.columns = [c.strip() for c in df.columns]

        # Limpieza
        df['Compañía'] = df['Compañía'].astype(str).str.strip()
        if 'Subramo' in df.columns: df['Subramo'] = df['Subramo'].fillna('General')
        if 'Ramo' in df.columns: df['Ramo'] = df['Ramo'].fillna('Otros')
        if 'AFILIADO' in df.columns:
            df['AFILIADO'] = df['AFILIADO'].fillna('NO AFILIADO').astype(str).str.strip().str.upper().replace({'NO AFILIADOS':'NO AFILIADO', 'AFILIADOS':'AFILIADO'})

        # Conversión Numérica (USD ya viene en dólares; no usamos tasa de cambio)
        def parse_numero_latino(val):
            if pd.isna(val): return 0.0
            texto = str(val).strip()
            try: return float(texto)
            except:
                texto = texto.replace('.', '').replace(',', '.')
                try: return float(texto)
                except: return 0.0

        df['USD'] = df['USD'].apply(parse_numero_latino)

        # Excluir ramos no deseados
        df = df[~df['Ramo'].str.upper().isin(['RIESGOS PORTUARIOS', 'RIESGOS PETROLEROS'])]

        # Pivoteo
        pivot_df = df.pivot_table(
            index=['País', 'Año', 'Compañía', 'Ramo', 'Subramo', 'AFILIADO'],
            columns='Tipo', values='USD', aggfunc='sum', fill_value=0
        ).reset_index()

        pivot_df.columns.name = None
        if 'Primas' not in pivot_df.columns: pivot_df['Primas'] = 0.0
        if 'Siniestros' not in pivot_df.columns: pivot_df['Siniestros'] = 0.0

        pivot_df['Siniestros'] = pivot_df['Siniestros'].abs()
        pivot_df['Siniestralidad'] = (pivot_df['Siniestros'] / pivot_df['Primas']).replace([float('inf'), -float('inf')], 0) * 100
        pivot_df['Resultado Técnico'] = pivot_df['Primas'] - pivot_df['Siniestros']

        return pivot_df, None
    except Exception as e:
        return None, f"Error: {e}"

# ==========================================
# 4. INTERFAZ Y LÓGICA PRINCIPAL
# ==========================================

# --- FUNCION AUXILIAR PARA PIVOT DE AÑOS ---
def crear_vista_pivot_anos(df_input, indice, valor='Primas'):
    """
    Crea una tabla con los años como columnas y una columna final de Total.
    """
    try:
        pivot = df_input.pivot_table(
            index=indice, 
            columns='Año', 
            values=valor, 
            aggfunc='sum', 
            fill_value=0
        )
        pivot['TOTAL CONSOLIDADO'] = pivot.sum(axis=1)
        pivot = pivot.sort_values('TOTAL CONSOLIDADO', ascending=False)
        pivot.columns = [str(c) for c in pivot.columns]
        return pivot.reset_index()
    except Exception as e:
        return pd.DataFrame()

with st.sidebar:
    st.image("https://www.alsum.co/wp-content/uploads/2022/08/LOGO-ALSUM-BLANCO-1-1024x282.png", use_container_width=True)
    st.header("Centro de Mando")
    st.info("📊 ALSUM Intelligence System")
    st.caption(f"Archivo base: {os.path.basename(DATA_REPO_PATH)}")

# --- CARGA INICIAL ---
if not os.path.exists(DATA_REPO_PATH):
    st.error(f"❌ No se encontró el archivo de datos: {DATA_REPO_PATH}")
    st.stop()

try:
    with st.spinner('Inicializando protocolos de análisis...'):
        with open(DATA_REPO_PATH, "rb") as f:
            df_final, error = load_data_universal(f)
except Exception as e:
    st.error(f"❌ Error crítico al cargar archivo: {e}")
    st.stop()

if error:
    st.error(f"❌ {error}")
    st.stop()

elif df_final is not None:
    # ---------------------------------------------------------
    # 1. FILTROS GLOBALES (SIDEBAR - MACRO)
    # ---------------------------------------------------------
    anios_disp = sorted(df_final['Año'].unique())
    ramos_disp = sorted(df_final['Ramo'].unique())
    paises_disp = sorted(df_final['País'].unique())

    st.sidebar.markdown("### 🌍 Filtros Macro")
    filtro_anios = st.sidebar.multiselect("Año", anios_disp, default=anios_disp)
    filtro_paises = st.sidebar.multiselect("País", paises_disp, default=paises_disp)
    filtro_afiliado = st.sidebar.radio("Afiliación", ["Todos", "Afiliados", "No afiliados"], horizontal=False)
    filtro_ramos = st.sidebar.multiselect("Ramo", ramos_disp, default=ramos_disp)

    # Aplicar Filtros Sidebar
    df_filtrado = df_final.copy()
    if filtro_anios: df_filtrado = df_filtrado[df_filtrado['Año'].isin(filtro_anios)]
    if filtro_paises: df_filtrado = df_filtrado[df_filtrado['País'].isin(filtro_paises)]
    if filtro_afiliado == "Afiliados": df_filtrado = df_filtrado[df_filtrado['AFILIADO'] == 'AFILIADO']
    elif filtro_afiliado == "No afiliados": df_filtrado = df_filtrado[df_filtrado['AFILIADO'] == 'NO AFILIADO']
    if filtro_ramos: df_filtrado = df_filtrado[df_filtrado['Ramo'].isin(filtro_ramos)]

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
        st.error("❌ No hay datos para los filtros seleccionados.")
        st.stop()

    # ---------------------------------------------------------
    # 3. KPIs GLOBALES (Afectados por Sidebar + Expander)
    # ---------------------------------------------------------
    escala = 1e9
    primas_tot = df_filtrado['Primas'].sum()
    siniestros_tot = df_filtrado['Siniestros'].sum()
    ratio_global = (siniestros_tot / primas_tot) * 100 if primas_tot > 0 else 0
    res_tec = primas_tot - siniestros_tot

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Volumen Primas", f"${primas_tot/escala:,.2f}B")
    k2.metric("Siniestros Totales", f"${siniestros_tot/escala:,.2f}B")
    k3.metric("Siniestralidad", f"{ratio_global:.1f}%", delta=f"{65-ratio_global:.1f}% vs Meta")
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
            st.markdown("##### 📅 Vista Desglosada por Años (Consolidado al final)")
            df_pivot_pais = crear_vista_pivot_anos(df_filtrado, 'País', metrica_view)
            format_dict = {col: '${:,.0f}' for col in df_pivot_pais.columns if col not in ['País']}
            st.dataframe(
                df_pivot_pais.style.format(format_dict).background_gradient(cmap="Blues", subset=['TOTAL CONSOLIDADO']),
                use_container_width=True, hide_index=True
            )
            pais_df_chart = df_filtrado.groupby('País')[['Primas', 'Siniestros']].sum().reset_index()
            pais_df_chart['Siniestralidad'] = (pais_df_chart['Siniestros']/pais_df_chart['Primas'])*100
        else:
            pais_df_chart = df_filtrado.groupby('País')[['Primas', 'Siniestros']].sum().reset_index()
            pais_df_chart['Siniestralidad'] = (pais_df_chart['Siniestros']/pais_df_chart['Primas'])*100
            pais_df_chart['Primas_M'] = pais_df_chart['Primas']/1e6
            c1, c2 = st.columns([2, 1])
            with c1:
                fig_map = px.scatter(pais_df_chart, x='Primas', y='Siniestralidad', 
                                   size='Primas', color='País',
                                   title="Matriz de Desempeño (Riesgo vs Volumen)")
                fig_map.add_hline(y=65, line_dash="dash", line_color="red")
                st.plotly_chart(fig_map, use_container_width=True)
            with c2:
                st.dataframe(pais_df_chart[['País', 'Primas', 'Siniestralidad']].sort_values('Primas', ascending=False)
                             .style.format({'Primas':'${:,.0f}', 'Siniestralidad':'{:.1f}%'}), 
                             hide_index=True, use_container_width=True)

        st.markdown("### Ranking de Compañías")
        if ver_desglose_anos and len(filtro_anios) > 1:
            st.info(f"Mostrando desglose anual de **{metrica_view}** por Compañía")
            df_pivot_comp = crear_vista_pivot_anos(df_filtrado, 'Compañía', metrica_view)
            st.dataframe(
                df_pivot_comp.head(50).style
                .format({col: '${:,.0f}' for col in df_pivot_comp.columns if col != 'Compañía'})
                .bar(subset=['TOTAL CONSOLIDADO'], color='#004A8F'),
                use_container_width=True, hide_index=True
            )
        else:
            comp_geo = df_filtrado.groupby('Compañía')[['Primas','Siniestros']].sum().reset_index()
            comp_geo['Siniestralidad'] = (comp_geo['Siniestros']/comp_geo['Primas'])*100
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
            df_pivot_ramo = crear_vista_pivot_anos(df_filtrado, 'Ramo', metrica_view)
            st.dataframe(
                df_pivot_ramo.style.format({col: '${:,.0f}' for col in df_pivot_ramo.columns if col != 'Ramo'})
                .background_gradient(subset=['TOTAL CONSOLIDADO'], cmap='Greens'),
                use_container_width=True, hide_index=True
            )
        else:
            ramo_df = df_filtrado.groupby('Ramo')[['Primas', 'Siniestros']].sum().reset_index()
            ramo_df['Ratio'] = (ramo_df['Siniestros']/ramo_df['Primas'])*100
            fig_bar = px.bar(ramo_df.sort_values('Primas', ascending=False), 
                           x='Ramo', y='Primas', color='Ratio',
                           color_continuous_scale='RdYlGn_r', 
                           title="Ramos: Volumen y Siniestralidad")
            st.plotly_chart(fig_bar, use_container_width=True)

    # === TAB 3: PDF (Igual que antes) ===
    with tab3:
        st.header("🧠 Generador de Informe de Conquista 2026")
        st.markdown("Este módulo utiliza **GPT-4** (configurado en el entorno) para redactar el plan estratégico.")
        c_ai1, c_ai2 = st.columns([2, 1])
        with c_ai1:
            foco = st.text_area("🎯 Instrucción Especial (Opcional)", placeholder="Ej: Enfocarme en México y reducir siniestralidad...")
        with c_ai2:
            st.write("")
            st.write("")
            btn_gen = st.button("🔥 GENERAR INFORME PDF", type="primary")
        if btn_gen:
            api_key = get_api_key()
            if not api_key:
                st.error("⚠️ Error: No se encontró la API KEY en las variables de entorno.")
            else:
                with st.status("🛠️ Fabricando tu Plan...", expanded=True) as status:
                    pais_analisis = df_filtrado.groupby('País')[['Primas', 'Siniestros']].sum().reset_index()
                    pais_analisis['Siniestralidad'] = (pais_analisis['Siniestros']/pais_analisis['Primas'])*100
                    top_paises = pais_analisis.sort_values('Primas', ascending=False).head(3)['País'].tolist()
                    prompt_user = (
                        f"Datos Clave: Primas ${primas_tot/1e9:.2f}B USD. Siniestralidad {ratio_global:.1f}%. "
                        f"Top Mercados: {', '.join(top_paises)}. Instrucción Usuario: {foco}. "
                        "Escribe un diagnóstico ejecutivo y 3 estrategias puntuales para crecer en 2026."
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
                        pdf = UltimatePDF()
                        pdf.cover_page("PLAN DE DOMINACIÓN 2026", "ESTRATEGIA PARA LA EXPANSIÓN")
                        pdf.add_page()
                        pdf.section_title("1. TABLERO DE CONTROL (KPIs)")
                        pdf.add_metric_box("PRIMAS (B)", f"${primas_tot/1e9:.2f}", 15, pdf.get_y()+5)
                        pdf.add_metric_box("SINIESTRALIDAD", f"{ratio_global:.1f}%", 115, pdf.get_y()+5)
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
        c_d1, c_d2 = st.columns([3,1])
        with c_d1:
            fig_scat_deep = px.scatter(comp_deep, x="Primas", y="Siniestros", hover_name="Compañía",
                                     title="Correlación Primas vs Siniestros por Empresa", trendline="ols")
            st.plotly_chart(fig_scat_deep, use_container_width=True)
        with c_d2:
            st.markdown("**Top Riesgos (Siniestros Altos)**")
            st.dataframe(comp_deep.sort_values("Siniestros", ascending=False).head(10)[['Compañía', 'Siniestros']], hide_index=True)