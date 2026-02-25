import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import datetime
import unicodedata
from openai import OpenAI
import os
import tempfile

def save_plotly_figure(fig, filename):
    fig.write_image(filename, format="png", width=1000, height=600)

# ==========================================
# 0. CONFIGURACIÓN INICIAL (DEBE IR PRIMERO)
# ==========================================
st.set_page_config(
    page_title="ALSUM 2026 | Strategic Command", 
    layout="wide", 
    page_icon="🦅",
    initial_sidebar_state="expanded"
)

# ==========================================
# 1. IMPORTACIÓN SEGURA DE UTILS
# ==========================================
# Intentamos importar utils, si falla definimos funciones dummy para que no rompa
try:
    import utils
    # Intentamos obtener la API key y el path, manejando errores si la función no existe
    try:
        api_key = utils.get_api_key()
    except AttributeError:
        api_key = None
        
    try:
        FULL_PATH = utils.get_file_path("plan_2026.xlsx")
    except AttributeError:
        FULL_PATH = "plan_2026.xlsx"
        
except (ImportError, AttributeError):
    st.warning("⚠️ Archivo `utils.py` no encontrado o incompleto. Usando modo seguro.")
    api_key = None
    FULL_PATH = "plan_2026.xlsx" # Fallback local
    
    # Definimos clases dummy por si utils no existe para que el PDF no falle
    class UltimatePDF:
        def cover_page(self, t, s): pass
        def add_page(self): pass
        def section_title(self, t): pass
        def chapter_body(self, t): pass
        def output(self, dest): return b''
        def ln(self, h): pass

# ==========================================
# 2. ESTILOS CSS PROFESIONALES
# ==========================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"]  { font-family: 'Inter', sans-serif; }
    
    /* Títulos */
    h1, h2, h3 { color: #0f172a; font-weight: 700; letter-spacing: -0.5px; }
    h4, h5 { color: #334155; font-weight: 600; }
    
    /* Métricas (KPI Cards) */
    div[data-testid="metric-container"] {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        padding: 20px;
        border-radius: 12px;
        border-left: 5px solid #004A8F;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        transition: transform 0.2s;
    }
    div[data-testid="metric-container"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    }
    
    /* Tablas */
    .dataframe { font-size: 13px !important; font-family: 'Inter', sans-serif !important; }
    
    /* Botones */
    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s;
        background: linear-gradient(135deg, #004A8F 0%, #002a52 100%);
        color: white;
        border: none;
    }
    .stButton > button:hover {
        box-shadow: 0 4px 12px rgba(0, 74, 143, 0.3);
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 3. FUNCIONES DE IA
# ==========================================
def generar_insight_ia(contexto_data, tipo_analisis):
    """Genera 3 bullet points de análisis estratégico."""
    if not api_key:
        return "⚠️ API Key no configurada. No se puede generar el análisis."
    
    try:
        client = OpenAI(api_key=api_key)
        prompt_sys = "Eres un analista senior de seguros (Reaseguros/Latam). Sé conciso, directo y estratégico."
        prompt_user = (
            f"Analiza los siguientes datos de {tipo_analisis}: \n{contexto_data}\n\n"
            "Genera ESTRICTAMENTE 3 conclusiones breves (bullet points) enfocadas en:"
            "1. Oportunidad de crecimiento o riesgo detectado."
            "2. Anomalía o dato destacado (outlier)."
            "3. Una recomendación de acción estratégica."
        )
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": prompt_sys},
                {"role": "user", "content": prompt_user}
            ],
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error IA: {str(e)}"

def generar_seccion_ia(api_key, contexto, tipo):
    prompts = {
        "resumen": (
            "Eres un analista senior de seguros. Resume en máximo 5 líneas, "
            "en lenguaje ejecutivo, los puntos más relevantes del contexto. "
            "Enfócate en visión estratégica, oportunidades y riesgos clave."
        ),
        "hallazgos": (
            "Actúa como consultor estratégico. Enumera 3 hallazgos clave en bullets, "
            "cada uno con enfoque en: 1) oportunidad, 2) riesgo o anomalía, 3) recomendación ejecutiva."
        ),
        "analisis": (
            "Haz un análisis detallado de los datos. Explica tendencias, riesgos, oportunidades, "
            "y destaca cualquier outlier relevante. Usa lenguaje profesional y preciso."
        ),
        "recomendaciones": (
            "Sugiere 3 recomendaciones estratégicas, claras y accionables, priorizadas según impacto. "
            "Sé concreto y profesional."
        ),
        "anexos": (
            "Describe en 3 líneas la metodología utilizada, las fuentes de datos y cualquier limitación relevante. "
            "Sé breve y formal."
        )
    }
    prompt = prompts.get(tipo, "Resume los datos de forma ejecutiva.")
    if not api_key:
        return "⚠️ API Key no configurada. No se puede generar el análisis."
    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Eres un analista senior de seguros, preciso y ejecutivo."},
                {"role": "user", "content": f"{prompt}\n\nContexto:\n{contexto}"}
            ],
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error IA: {str(e)}"

# ==========================================
# 4. CARGA Y LIMPIEZA DE DATOS (CORREGIDO)
# ==========================================

def clean_col_name(col):
    """Normaliza nombres de columnas de forma agresiva."""
    # 1. Convertir a string y minúsculas y quitar acentos para comparar
    c = str(col).strip().lower()
    c = unicodedata.normalize('NFKD', c).encode('ascii', 'ignore').decode('utf-8')
    
    # 2. Mapeo directo de variaciones comunes a Nombres Estándar (Title Case)
    if 'anio' in c or 'ano' in c or 'year' in c: return 'Año'
    if 'pais' in c or 'country' in c: return 'País'
    if 'compania' in c or 'empresa' in c: return 'Compañía'
    if 'ramo' in c or 'producto' in c: return 'Ramo'
    if 'tipo' in c and 'operacion' in c: return 'Tipo'
    if 'tipo' in c and len(c) < 6: return 'Tipo'
    if 'usd' in c or 'valor' in c or 'monto' in c: return 'USD'
    if 'afiliado' in c: return 'AFILIADO'
    
    # Si no coincide con las claves, devolvemos formato Título
    return col.strip().title()

@st.cache_data(show_spinner=False)
def load_data():
    try:
        # Intentamos usar utils, sino pandas directo
        if 'utils' in globals() and hasattr(utils, 'load_plan_accion_procesado'):
            df, error = utils.load_plan_accion_procesado(FULL_PATH, sheet_name="plan_2026")
        else:
            df = pd.read_excel(FULL_PATH)
            error = None
    except Exception as e:
        return None, f"Error leyendo archivo: {str(e)}"

    if error: return None, error
    if df is None or df.empty: return None, "Archivo vacío."

    # 1. Aplicar limpieza de nombres de columnas
    df.columns = [clean_col_name(c) for c in df.columns]

    # === CORRECCIÓN CRÍTICA: ELIMINAR DUPLICADOS DE COLUMNAS ===
    # Si 'pais' y 'País' se normalizan ambos a 'País', Pandas crea dos columnas con el mismo nombre.
    # Melt falla al recibir columnas duplicadas. Esta línea mantiene solo la primera aparición.
    df = df.loc[:, ~df.columns.duplicated()]

    # 2. Transformación Inteligente (Melt/Unpivot)
    # Buscamos si existen las columnas de valores por separado (Formato Ancho)
    cols_posibles_valor = ['Primas', 'Siniestros', 'No Reporta', 'Resultado Tecnico', 'No_Reporta']
    cols_encontradas = [c for c in df.columns if c in cols_posibles_valor or 'Prima' in c or 'Siniestro' in c]
    
    # Si encontramos columnas de valores y NO existe la columna 'Tipo', hacemos melt
    if len(cols_encontradas) > 0 and 'Tipo' not in df.columns:
        # Identificadores son todo lo que NO sea valor
        id_vars = [c for c in df.columns if c not in cols_encontradas]
        
        # Validamos que id_vars no esté vacío para evitar errores raros
        if not id_vars:
            # Si no hay variables ID, es un archivo raro, pero intentamos procesarlo
            df['id_temp'] = df.index
            id_vars = ['id_temp']
            
        try:
            df = df.melt(id_vars=id_vars, value_vars=cols_encontradas, var_name='Tipo', value_name='USD')
        except Exception as e:
            return None, f"Error transformando estructura de datos (Melt): {str(e)}"
    
    # 3. Limpieza final de valores en la columna 'Tipo' y 'AFILIADO'
    if 'Tipo' in df.columns:
        df['Tipo'] = df['Tipo'].astype(str).str.title().str.strip()
        # Normalizar variaciones de "No Reporta"
        df['Tipo'] = df['Tipo'].replace({'No_Reporta': 'No Reporta', 'Noreporta': 'No Reporta'})
    
    # Asegurar columna AFILIADO
    if 'AFILIADO' not in df.columns:
        df['AFILIADO'] = 'NO AFILIADO'
    
    df['AFILIADO'] = df['AFILIADO'].fillna('NO AFILIADO').astype(str).str.upper()
    # Si contiene 'NO', es NO AFILIADO, sino es AFILIADO
    df.loc[df['AFILIADO'].str.contains('NO', na=False), 'AFILIADO'] = 'NO AFILIADO'
    df.loc[~df['AFILIADO'].str.contains('NO', na=False), 'AFILIADO'] = 'AFILIADO'

    return df, None

# Carga de datos
with st.spinner("Cargando Motor de Datos..."):
    df_final, err = load_data()

if err:
    st.error(f"❌ Error Fatal: {err}")
    st.stop()

# ==========================================
# 5. BARRA LATERAL (FILTROS)
# ==========================================
st.sidebar.image("https://alsum.co/wp-content/uploads/2018/06/Logo-Alsum-Web.png", width=160)
st.sidebar.markdown("### ⚙️ Configuración Global")

df_filtrado = df_final.copy()

# --- FILTRO 1: AÑOS ---
if 'Año' in df_filtrado.columns:
    # Convertir a numérico para ordenar bien
    df_filtrado['Año'] = pd.to_numeric(df_filtrado['Año'], errors='coerce')
    df_filtrado = df_filtrado.dropna(subset=['Año']) # Eliminar filas sin año
    df_filtrado['Año'] = df_filtrado['Año'].astype(int)
    
    # SOLO mostrar años 2022, 2023, 2024
    anios_disponibles = [a for a in sorted(df_filtrado['Año'].unique()) if a in [2022, 2023, 2024]]
    
    if not anios_disponibles:
        st.error("La columna 'Año' existe pero no tiene datos válidos para 2022, 2023 o 2024.")
        st.stop()
        
    sel_anios = st.sidebar.multiselect("📅 Años Fiscales", anios_disponibles, default=anios_disponibles)
    
    if sel_anios:
        df_filtrado = df_filtrado[df_filtrado['Año'].isin(sel_anios)]
    else:
        st.warning("Selecciona al menos un año.")
        st.stop()
else:
    st.error(f"❌ No se encontró la columna 'Año'. Columnas detectadas: {df_filtrado.columns.tolist()}")
    st.stop()

# --- FILTRO 2: PAÍS ---
if 'País' in df_filtrado.columns:
    paises = sorted(df_filtrado['País'].astype(str).unique())
    sel_pais = st.sidebar.multiselect("🌎 Países", paises, default=paises)
    if sel_pais:
        df_filtrado = df_filtrado[df_filtrado['País'].isin(sel_pais)]

# --- FILTRO 3: RAMO ---
if 'Ramo' in df_filtrado.columns:
    # Excluye los ramos no deseados
    ramos = sorted(df_filtrado['Ramo'].astype(str).unique())
    ramos_filtrados = [r for r in ramos if r not in ["Riesgos petroleros", "Riesgos portuarios"]]
    sel_ramos = st.sidebar.multiselect("📦 Ramo / Producto", ramos_filtrados, default=ramos_filtrados)
    if sel_ramos:
        df_filtrado = df_filtrado[df_filtrado['Ramo'].isin(sel_ramos)]

# --- FILTRO 4: AFILIADO ---
if 'AFILIADO' in df_filtrado.columns:
    opciones_afiliado = ["Todos", "AFILIADO", "NO AFILIADO"]
    sel_afiliado = st.sidebar.selectbox("🔗 Estado de Afiliación", opciones_afiliado, index=0)
    if sel_afiliado != "Todos":
        df_filtrado = df_filtrado[df_filtrado['AFILIADO'] == sel_afiliado]

# --- FILTRO 5: COMPAÑÍA ---
if 'Compañía' in df_filtrado.columns:
    companias = sorted(df_filtrado['Compañía'].astype(str).unique())
    opciones_companias = ["Todas"] + companias
    # Si ya hay selección previa, manténla; si no, selecciona "Todas" por defecto
    sel_companias = st.sidebar.multiselect(
        "🏢 Compañía", 
        opciones_companias, 
        default=["Todas"] if len(companias) > 1 else companias
    )
    # Si "Todas" está seleccionada, usa todas las compañías
    if "Todas" in sel_companias or not sel_companias:
        sel_companias = companias
    else:
        # Si no, usa solo las seleccionadas (sin "Todas")
        sel_companias = [c for c in sel_companias if c in companias]
    # Aplica el filtro
    if sel_companias:
        df_filtrado = df_filtrado[df_filtrado['Compañía'].isin(sel_companias)]

st.sidebar.markdown("---")

# ==========================================
# 6. LÓGICA DE NEGOCIO (KPIS)
# ==========================================
# Aseguramos que la columna USD sea numérica
col_valor = 'USD'
if col_valor not in df_filtrado.columns:
    st.error("No se encontró columna de valor (USD).")
    st.stop()

df_filtrado[col_valor] = pd.to_numeric(df_filtrado[col_valor], errors='coerce').fillna(0)

# Separación de datos usando 'Tipo'
df_primas = df_filtrado[df_filtrado['Tipo'].str.contains('Prima', case=False, na=False)]
df_siniestros = df_filtrado[df_filtrado['Tipo'].str.contains('Siniestro', case=False, na=False)]
df_noreporta = df_filtrado[df_filtrado['Tipo'].str.contains('No Reporta', case=False, na=False)]

total_primas = df_primas[col_valor].sum()
total_siniestros = df_siniestros[col_valor].sum()
total_noreporta = df_noreporta[col_valor].sum()

# Cálculo robusto de "No Reporta"
if 'No Reporta' in df_filtrado.columns:
    total_noreporta = df_filtrado['No Reporta'].sum()
elif 'Tipo' in df_filtrado.columns and col_valor in df_filtrado.columns:
    total_noreporta = df_filtrado[df_filtrado['Tipo'].str.contains('No Reporta', case=False, na=False)][col_valor].sum()
else:
    total_noreporta = 0

# Cálculo de Siniestralidad Global
siniestralidad_global = (total_siniestros / total_primas * 100) if total_primas > 0 else 0

st.title(f"🚀 Dashboard Estratégico {max(sel_anios) if sel_anios else ''}")
st.markdown("Visión consolidada del desempeño del mercado asegurador.")

# KPIs Cards
col1, col2, col3, col4 = st.columns(4)
col1.metric("💰 Primas Confirmadas", f"${total_primas:,.0f}")
col2.metric("🔥 Siniestros Confirmados", f"${total_siniestros:,.0f}")
col3.metric("📉 % Siniestralidad", f"{siniestralidad_global:.1f}%", 
            delta="Saludable" if siniestralidad_global < 60 else "Atención", delta_color="inverse")
col4.metric("⚠️ Data No Reportada (Riesgo)", f"${total_noreporta:,.0f}", delta="Incertidumbre", delta_color="off")

st.divider()

# ==========================================
# 8. PESTAÑAS DETALLADAS
# ==========================================
tab2, tab1, tab3, tab4 = st.tabs([
    "📦 Mapa & Territorio",
    "🗺️ Productos (Ramos)", 
    "📝 Generador PDF IA", 
    "🔬 Data Lab"
])

# === TAB 2: GEOGRAFÍA ===
with tab2:
    st.subheader("Territorio y Rentabilidad")
    
    if 'País' in df_filtrado.columns:
        pais_stats = df_filtrado.groupby(['País', 'Tipo'])[col_valor].sum().unstack(fill_value=0).reset_index()
        
        # Lógica de columnas segura
        cols_p = pais_stats.columns.tolist()
        def get_val(k, r): return r[[c for c in cols_p if k.lower() in str(c).lower()]].sum() if any(k.lower() in str(c).lower() for c in cols_p) else 0

        pais_stats['Primas'] = pais_stats.apply(lambda x: get_val('Prima', x), axis=1)
        pais_stats['Siniestros'] = pais_stats.apply(lambda x: get_val('Siniestro', x), axis=1)
        
        pais_stats['Siniestralidad'] = (pais_stats['Siniestros'] / pais_stats['Primas'] * 100).fillna(0)
        
        fig_map = px.scatter(
            pais_stats,
            x='Primas', y='Siniestralidad',
            size='Primas', color='País',
            hover_name='País', text='País',
            title="Matriz de Rentabilidad País",
            labels={'Primas': 'Volumen (USD)', 'Siniestralidad': 'Siniestralidad (%)'}
        )
        fig_map.add_hline(y=60, line_dash="dash", line_color="red", annotation_text="Límite 60%")
        fig_map.update_traces(textposition='top center')
        st.plotly_chart(fig_map, use_container_width=True)
        
        if st.button("🤖 Analizar Geografía con IA"):
            st.markdown(generar_insight_ia(pais_stats.to_string(), "Desempeño por País"))

# === TAB 1: PRODUCTOS ===
with tab1:
    st.subheader("Análisis de Portafolio y Participación")
    col_prod_1, col_prod_2 = st.columns([2, 1])
    
    if 'Ramo' in df_filtrado.columns:
        # Pivotar por Ramo
        ramo_stats = df_filtrado.groupby(['Ramo', 'Tipo'])[col_valor].sum().unstack(fill_value=0).reset_index()
        
        # Normalización de nombres de columnas del pivot
        cols_pivot = ramo_stats.columns.tolist()
        def get_col_val(keyword, df_row):
            matches = [c for c in cols_pivot if keyword.lower() in str(c).lower()]
            return df_row[matches].sum() if matches else 0

        ramo_stats['Primas'] = ramo_stats.apply(lambda x: get_col_val('Prima', x), axis=1)
        ramo_stats['Siniestros'] = ramo_stats.apply(lambda x: get_col_val('Siniestro', x), axis=1)
        ramo_stats['No Reporta'] = ramo_stats.apply(lambda x: get_col_val('No Reporta', x), axis=1)
        
        ramo_stats['Siniestralidad'] = (ramo_stats['Siniestros'] / ramo_stats['Primas'] * 100).fillna(0)
        
        # Colores Semáforo
        def get_color(ratio):
            if ratio < 50: return '#10B981' # Verde
            elif ratio < 75: return '#F59E0B' # Amarillo
            else: return '#EF4444' # Rojo
            
        ramo_stats['Color'] = ramo_stats['Siniestralidad'].apply(get_color)
        ramo_stats = ramo_stats.sort_values('Primas', ascending=False)
        
        with col_prod_1:
            st.markdown("**🚥 Desempeño por Ramo (Color = Siniestralidad)**")
            fig_bar = go.Figure()
            
            fig_bar.add_trace(go.Bar(
                x=ramo_stats['Ramo'], y=ramo_stats['Primas'],
                name='Primas', marker_color=ramo_stats['Color'],
                text=ramo_stats['Siniestralidad'].apply(lambda x: f"{x:.1f}%"),
                textposition='auto'
            ))
            fig_bar.add_trace(go.Bar(
                x=ramo_stats['Ramo'], y=ramo_stats['No Reporta'],
                name='No Reporta', marker_color='#94A3B8'
            ))
            fig_bar.update_layout(barmode='stack', xaxis_tickangle=-45, legend=dict(orientation="h", y=1.1))
            st.plotly_chart(fig_bar, use_container_width=True)

        with col_prod_2:
            st.markdown("**📋 Detalle Numérico**")
            st.dataframe(
                ramo_stats[['Ramo', 'Primas', 'Siniestralidad']].style
                .format({'Primas': '${:,.0f}', 'Siniestralidad': '{:.1f}%'})
                .background_gradient(subset=['Siniestralidad'], cmap='RdYlGn_r'),
                use_container_width=True, height=400
            )
    
    st.divider()
    
    # Market Share
    if 'AFILIADO' in df_primas.columns:
        ms_col1, ms_col2 = st.columns(2)
        share_vol = df_primas.groupby('AFILIADO')[col_valor].sum().reset_index()
        share_count = df_primas.groupby('AFILIADO')['Compañía'].nunique().reset_index()
        share_count.columns = ['AFILIADO', 'Empresas']
        
        colores_share = {'AFILIADO': '#004A8F', 'NO AFILIADO': '#CBD5E1'}
        
        with ms_col1:
            st.markdown("**Participación por Volumen (USD)**")
            fig_vol = px.pie(share_vol, values=col_valor, names='AFILIADO', 
                             color='AFILIADO', color_discrete_map=colores_share, hole=0.6)
            st.plotly_chart(fig_vol, use_container_width=True)
            
        with ms_col2:
            st.markdown("**Participación por # de Empresas**")
            fig_count = px.pie(share_count, values='Empresas', names='AFILIADO', 
                               color='AFILIADO', color_discrete_map=colores_share, hole=0.6)
            st.plotly_chart(fig_count, use_container_width=True)

# --- A. GRÁFICA DE TORTA ---
c_chart, c_text = st.columns([2, 1])

with c_chart:
    st.subheader("📊 Distribución de Flujos")
    data_pie = pd.DataFrame({
        'Categoría': ['Primas', 'Siniestros', 'No Reporta'],
        'Valor': [total_primas, total_siniestros, total_noreporta]
    })
    
    fig_pie = px.pie(
        data_pie, values='Valor', names='Categoría',
        color='Categoría',
        color_discrete_map={'Primas': '#004A8F', 'Siniestros': '#DC2626', 'No Reporta': '#F59E0B'},
        hole=0.5
    )
    fig_pie.update_traces(textinfo='percent+label', textfont_size=14)
    st.plotly_chart(fig_pie, use_container_width=True)

with c_text:
    st.info("💡 **Análisis Rápido:** La proporción de 'No Reporta' indica el nivel de opacidad del mercado seleccionado.")
    if st.button("🤖 Analizar KPI Global con IA", key="btn_ia_global"):
        with st.spinner("Consultando estrategia..."):
            contexto = f"Primas: {total_primas}, Siniestros: {total_siniestros}, Siniestralidad: {siniestralidad_global}%, No Reportado: {total_noreporta}. {instruccion}"
            insight = generar_insight_ia(contexto, "KPIs Globales")
            st.success("Análisis Generado:")
            st.markdown(insight)

st.markdown("---")

# === TAB 3: GENERADOR PDF ===
with tab3:
    st.subheader("📝 Generación de Informe Ejecutivo (PDF)")
    col_pdf_1, col_pdf_2 = st.columns([3, 1])
    
    with col_pdf_1:
        instruccion = st.text_area("Instrucciones específicas:", "Enfocarse en la alta siniestralidad y riesgos ocultos.")
    
    with col_pdf_2:
        st.write("") 
        st.write("")
        btn_pdf = st.button("📄 GENERAR PDF", type="primary", use_container_width=True)
    
    if btn_pdf:
        with st.status("🛠️ Construyendo informe...", expanded=True):
            st.write("Analizando datos globales...")

            # Contextos para IA
            contexto_global = f"Primas: {total_primas}, Siniestros: {total_siniestros}, Siniestralidad: {siniestralidad_global}%, No Reportado: {total_noreporta}. {instruccion}"

            resumen = generar_seccion_ia(api_key, contexto_global, "resumen")
            hallazgos = generar_seccion_ia(api_key, contexto_global, "hallazgos").split('\n')
            analisis = generar_seccion_ia(api_key, contexto_global, "analisis")
            recomendaciones = generar_seccion_ia(api_key, contexto_global, "recomendaciones").split('\n')
            anexos = generar_seccion_ia(api_key, contexto_global, "anexos")

            # Guardar gráficos
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile1:
                save_plotly_figure(fig_bar, tmpfile1.name)
                bar_chart_path = tmpfile1.name
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile2:
                save_plotly_figure(fig_pie, tmpfile2.name)
                pie_chart_path = tmpfile2.name

            try:
                st.write("Maquetando documento...")
                pdf = utils.UltimatePDF()
                pdf.cover_page("INFORME ESTRATÉGICO 2026", "ALSUM INTELLIGENCE")
                pdf.executive_summary(resumen)
                pdf.key_findings([h for h in hallazgos if h.strip()])
                pdf.add_section("Análisis Detallado", analisis)
                pdf.section_title("Visualizaciones Clave")
                pdf.add_image_section("Desempeño por Ramo", bar_chart_path)
                pdf.add_image_section("Distribución de Flujos", pie_chart_path)
                pdf.recommendations([r for r in recomendaciones if r.strip()])
                pdf.annex(anexos)
                pdf_bytes = bytes(pdf.output(dest='S'))
                st.write("✅ ¡Hecho!")
                st.download_button("📥 Descargar PDF", data=pdf_bytes, file_name="Reporte_ALSUM.pdf", mime="application/pdf")
            except Exception as e:
                st.error(f"Error generando PDF: {e}")

# === TAB 4: DATA LAB ===
with tab4:
    st.subheader("🔬 Laboratorio de Datos")
    with st.expander("Ver Dataframe Completo", expanded=True):
        st.dataframe(df_filtrado, use_container_width=True)
    
    csv = df_filtrado.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Descargar CSV Filtrado", data=csv, file_name="data_filtrada.csv", mime="text/csv")

# ==========================================
# 7. VISUALIZACIÓN PRINCIPAL (HOME)
# ==========================================

# --- B. RANKING DE COMPAÑÍAS ---
st.subheader("🏆 Ranking de Compañías")
st.markdown("Desempeño consolidado ordenado por volumen de Primas.")

if 'Compañía' in df_filtrado.columns:
    # Agrupar datos para el ranking
    df_rank = df_filtrado.groupby(['Compañía', 'Tipo'])[col_valor].sum().unstack(fill_value=0).reset_index()
    
    # Asegurar columnas aunque falten datos
    for col in ['Primas', 'Siniestros', 'No Reporta']:
        # Buscamos columnas que contengan la palabra clave
        matches = [c for c in df_rank.columns if col.lower() in str(c).lower()]
        if matches:
            # Si hay multiple matches (ej No Reporta), sumamos, sino tomamos la que es
            if len(matches) > 1:
                df_rank[col] = df_rank[matches].sum(axis=1)
            else:
                df_rank[col] = df_rank[matches[0]]
        else:
            df_rank[col] = 0
            
    # Cálculos finales
    df_rank['Siniestralidad'] = (df_rank['Siniestros'] / df_rank['Primas'] * 100).fillna(0)
    df_rank['Resultado'] = df_rank['Primas'] - df_rank['Siniestros']
    
    # Ordenar y mostrar
    df_rank = df_rank.sort_values('Primas', ascending=False)
    
    st.dataframe(
        df_rank[['Compañía', 'Primas', 'Siniestros', 'Siniestralidad', 'No Reporta', 'Resultado']].style
        .format({
            'Primas': '${:,.0f}', 'Siniestros': '${:,.0f}', 
            'No Reporta': '${:,.0f}', 'Resultado': '${:,.0f}',
            'Siniestralidad': '{:.1f}%'
        })
        .background_gradient(subset=['Siniestralidad'], cmap='RdYlGn_r', vmin=0, vmax=100)
        .bar(subset=['Primas'], color='#e0f2fe'),
        use_container_width=True,
        height=400
    )
    
    with st.expander("🤖 Ver Análisis IA del Ranking"):
        if st.button("Generar Análisis Ranking"):
             top_3 = df_rank.head(3).to_string()
             worst_sin = df_rank.sort_values('Siniestralidad', ascending=False).head(3).to_string()
             res = generar_insight_ia(f"Top 3 Vol:\n{top_3}\nTop 3 Siniestralidad:\n{worst_sin}", "Ranking Empresas")
             st.markdown(res)

# --- ANÁLISIS POR EMPRESA Y PAÍS ---
if 'Compañía' in df_filtrado.columns and sel_companias:
    df_empresa = df_final[df_final['Compañía'].isin(sel_companias)]
    # Aplica los mismos filtros de año, país, ramo y afiliado al df_empresa
    if 'Año' in df_empresa.columns and sel_anios:
        df_empresa = df_empresa[df_empresa['Año'].isin(sel_anios)]
    if 'País' in df_empresa.columns and sel_pais:
        df_empresa = df_empresa[df_empresa['País'].isin(sel_pais)]
    if 'Ramo' in df_empresa.columns and sel_ramos:
        df_empresa = df_empresa[df_empresa['Ramo'].isin(sel_ramos)]
    if 'AFILIADO' in df_empresa.columns and sel_afiliado != "Todos":
        df_empresa = df_empresa[df_empresa['AFILIADO'] == sel_afiliado]

    resumen = (
        df_empresa.groupby(['Compañía', 'País', 'AFILIADO', 'Tipo'])['USD']
        .sum()
        .reset_index()
        .pivot_table(index=['Compañía', 'País', 'AFILIADO'], columns='Tipo', values='USD', fill_value=0)
        .reset_index()
    )
    # Asegura columnas aunque falten datos
    for col in ['Primas', 'Siniestros', 'No Reporta']:
        if col not in resumen.columns:
            resumen[col] = 0
    resumen['Siniestralidad %'] = (resumen['Siniestros'] / resumen['Primas'] * 100).fillna(0)
    st.dataframe(
        resumen[['Compañía', 'País', 'AFILIADO', 'Primas', 'Siniestros', 'No Reporta', 'Siniestralidad %']].style
        .format({'Primas': '${:,.0f}', 'Siniestros': '${:,.0f}', 'No Reporta': '${:,.0f}', 'Siniestralidad %': '{:.1f}%'}),
        use_container_width=True
    )

# --- TABLA: CUÁNTAS EMPRESAS HAY POR PAÍS (OBEDECE TODOS LOS FILTROS) ---
if 'Compañía' in df_filtrado.columns and 'País' in df_filtrado.columns:
    # Agrupa por país y cuenta empresas únicas
    empresas_por_pais = (
        df_filtrado.groupby('País')['Compañía']
        .nunique()
        .reset_index()
        .rename(columns={'Compañía': 'Empresas Únicas'})
        .sort_values('Empresas Únicas', ascending=False)
    )
    st.subheader("🌍 Empresas Únicas por País (Filtros Aplicados)")
    st.dataframe(
        empresas_por_pais.style
        .format({'Empresas Únicas': '{:,.0f}'})
        .background_gradient(subset=['Empresas Únicas'], cmap='Blues'),
        use_container_width=True
    )