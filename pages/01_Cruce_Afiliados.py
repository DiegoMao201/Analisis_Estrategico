import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import io
import utils  # Se asume que existe tu librería original 'utils.py'
from openai import OpenAI # Cliente oficial de OpenAI

# ==============================================================================
# 1. CONFIGURACIÓN ESTRATÉGICA Y ESTILOS
# ==============================================================================
st.set_page_config(
    page_title="ALSUM 360 | Enterprise Command Center",
    layout="wide",
    page_icon="💎",
    initial_sidebar_state="expanded"
)

# Estilos CSS Premium
st.markdown("""
<style>
    .stApp { background-color: #F4F6F9; }
    h1, h2, h3 { color: #003366; font-family: 'Segoe UI', sans-serif; }
    
    /* Tarjetas de Métricas Mejoradas */
    div[data-testid="stMetric"] {
        background-color: #ffffff;
        border-left: 5px solid #003366;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border-radius: 8px;
    }
    
    /* Tabs más profesionales */
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #ffffff;
        border-radius: 5px 5px 0 0;
        padding: 10px 20px;
        color: #003366;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #003366;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. FUNCIONES AUXILIARES DE ROBUSTEZ (NORMALIZACIÓN)
# ==============================================================================

def normalize_text(series):
    """Convierte una serie a string, minúsculas y quita espacios para comparaciones precisas."""
    return series.astype(str).str.strip().str.lower()

def find_col(df, keywords):
    """
    Busca una columna en el DataFrame que contenga alguna de las keywords.
    Es insensible a mayúsculas/minúsculas y acentos.
    Retorna el nombre real de la columna o None.
    """
    if df is None: return None
    # Normalizar nombres de columnas del DF
    cols_norm = {c.lower().strip().replace('á','a').replace('é','e').replace('í','i').replace('ó','o').replace('ú','u'): c for c in df.columns}
    
    for k in keywords:
        k_norm = k.lower().strip().replace('á','a').replace('é','e').replace('í','i').replace('ó','o').replace('ú','u')
        # Busqueda exacta primero
        if k_norm in cols_norm:
            return cols_norm[k_norm]
        # Busqueda parcial (contiene)
        for c_norm, c_real in cols_norm.items():
            if k_norm in c_norm:
                return c_real
    return None

def deduplicate_columns(df):
    df = df.copy()
    cols = pd.Series(df.columns)
    for dup in cols[cols.duplicated()].unique():
        cols[cols[dup].index] = [dup + '.' + str(i) if i != 0 else dup for i in range(sum(cols == dup))]
    df.columns = cols
    return df

def clean_column_names(df):
    # Elimina espacios extra en los nombres de columnas
    df.columns = [str(c).strip() for c in df.columns]
    return df

# ==============================================================================
# 3. CARGA Y PROCESAMIENTO DE DATOS (LÓGICA MEJORADA)
# ==============================================================================

@st.cache_data(ttl=3600, show_spinner=False)
def load_data_master():
    plan_path = utils.get_file_path("plan_2026.csv")
    dir_path = utils.get_file_path("Directorio_Afiliados_2025.xlsx")
    
    # 1. Directorio
    try:
        df_dir = utils.load_excel_sheet(dir_path, sheet_name="Directorio 2025")
        df_dir = deduplicate_columns(df_dir)
        df_dir = clean_column_names(df_dir)
    except Exception as e:
        return None, None, None, f"Error cargando Directorio: {str(e)}"
    
    # 2. Nuevos
    try:
        xl = pd.ExcelFile(dir_path)
        sheet_candidates = [s for s in xl.sheet_names if 'nuevos' in s.lower()]
        if sheet_candidates:
            df_nuevos = pd.read_excel(dir_path, sheet_name=sheet_candidates[0])
            df_nuevos = clean_column_names(df_nuevos)
            
            # Renombrado Estratégico (Normalización inicial)
            rename_map = {
                'Tipo_de_Compañía': 'Categoria',
                'Compañía': 'Empresa',
                'Tipo_de_Afiliado': 'Tipo_de_Afiliado',
                'Tipo_Afiliado': 'Tipo_de_Afiliado',
            }
            # Aplicar renombrado si las columnas existen (insensible a mayúsculas)
            cols_actuales = {c.lower(): c for c in df_nuevos.columns}
            map_final = {}
            for k, v in rename_map.items():
                col_found = find_col(df_nuevos, [k])
                if col_found:
                    map_final[col_found] = v
            
            df_nuevos = df_nuevos.rename(columns=map_final)
            df_nuevos = deduplicate_columns(df_nuevos)
            
            # --- LOGICA DE ENRIQUECIMIENTO (PAÍS) ROBUSTA ---
            # Buscamos columnas clave en Directorio
            col_pais_dir = find_col(df_dir, ['país', 'pais', 'location'])
            col_empresa_dir = find_col(df_dir, ['empresa', 'compañía', 'compañia', 'nombre'])
            
            col_empresa_nuevos = find_col(df_nuevos, ['empresa', 'compañía'])
            
            if col_pais_dir and col_empresa_dir and col_empresa_nuevos:
                # Crear diccionario normalizado (minusculas -> Pais)
                keys = df_dir[col_empresa_dir].astype(str).str.strip().str.lower()
                values = df_dir[col_pais_dir]
                mapa_pais = dict(zip(keys, values))
                
                # Mapear
                claves_nuevos = df_nuevos[col_empresa_nuevos].astype(str).str.strip().str.lower()
                df_nuevos['País_Detectado'] = claves_nuevos.map(mapa_pais).fillna('Sin Asignar')
            else:
                df_nuevos['País_Detectado'] = 'No Data (Faltan Columnas)'
        else:
            return None, None, None, "No se encontró pestaña 'Nuevos'."
    except Exception as e:
        return None, None, None, f"Error cargando Nuevos: {str(e)}"
    
    # 3. Plan 2026
    df_plan, err = utils.load_plan_accion_procesado(plan_path)
    if err: return None, None, None, f"Error Plan: {err}"
    
    return df_dir, df_nuevos, df_plan, None

def generate_excel_download(df_nuevos, extra_sheets=None):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        if df_nuevos is not None:
            df_nuevos.to_excel(writer, sheet_name='Nuevos_Detallado', index=False)
            if 'País_Detectado' in df_nuevos.columns:
                resumen_pais = df_nuevos.groupby('País_Detectado').size().reset_index(name='Conteo')
                resumen_pais.to_excel(writer, sheet_name='Resumen_Pais', index=False)
        
        # Hojas extra
        if extra_sheets:
            for name, df in extra_sheets.items():
                if not df.empty:
                    df.to_excel(writer, sheet_name=name[:30], index=False)
    return output.getvalue()

# --- FUNCIÓN IA ---
def consultar_gpt4(api_key, prompt, data_context):
    if not api_key:
        return "⚠️ No se detectó la API Key en utils."
    
    client = OpenAI(api_key=api_key)
    full_prompt = f"""
    Actúa como un Analista Senior de Estrategia y Seguros de ALSUM. 
    Analiza los siguientes datos:
    {data_context}
    
    PREGUNTA DEL USUARIO: {prompt}
    
    Responde con insights accionables, puntos clave y recomendaciones estratégicas. Sé directo y profesional.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": full_prompt}],
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error API OpenAI: {str(e)}"

# ==============================================================================
# 4. INTERFAZ DE USUARIO (MAIN)
# ==============================================================================

def main():
    # --- SIDEBAR GLOBAL ---
    with st.sidebar:
        st.header("⚙️ Configuración & IA")
        api_key = utils.get_api_key()
        if api_key:
            st.success("API Key Detectada")
        else:
            st.warning("API Key No Detectada")
            
        st.info("Sistema cargado y optimizado.")

    col_logo, col_title, col_download = st.columns([1, 4, 2])
    with col_title:
        st.title("ALSUM Analytics | Estrategia 360º")
        st.markdown("**Inteligencia de Mercado & Gestión de Afiliados**")
        
    with st.spinner("🔄 Procesando matrices de datos y normalizando textos..."):
        df_dir, df_nuevos, df_plan, error_msg = load_data_master()
        
    if error_msg:
        st.error(error_msg)
        st.stop()
        
    # --- TABS REORGANIZADOS ---
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🌍 Dashboard Nuevos",
        "📋 Directorio & Desglose",
        "📂 Data Warehouse",
        "📈 Comparativo País & KPIs",
        "🤖 Laboratorio IA"
    ])

    # ==========================================================================
    # TAB 1: DASHBOARD NUEVOS (LÓGICA ORIGINAL OPTIMIZADA)
    # ==========================================================================
    with tab1:
        st.subheader("Panel de Control: Nuevas Incorporaciones")
        
        # Detectar columnas clave con la función robusta
        c_pais_nuevos = find_col(df_nuevos, ['país', 'pais', 'detectado'])
        c_cat_nuevos = find_col(df_nuevos, ['categoria', 'tipo compañía', 'tipo compañia'])
        c_tipo_nuevos = find_col(df_nuevos, ['tipo de afiliado', 'tipo afiliado', 'tipo'])
        c_empresa_nuevos = find_col(df_nuevos, ['empresa', 'compañía'])

        # --- FILTROS ---
        with st.expander("🔎 Filtros de Datos (Nuevos)", expanded=True):
            col_f1, col_f2, col_f3 = st.columns(3)
            
            sel_paises = []
            if c_pais_nuevos:
                paises_disp = sorted(df_nuevos[c_pais_nuevos].dropna().astype(str).unique())
                sel_paises = col_f1.multiselect("Filtrar por País", paises_disp, default=paises_disp)
            
            sel_cat = []
            if c_cat_nuevos:
                cats_disp = sorted(df_nuevos[c_cat_nuevos].dropna().astype(str).unique())
                sel_cat = col_f2.multiselect("Filtrar por Categoría", cats_disp, default=cats_disp)
            
            sel_tipo = []
            if c_tipo_nuevos:
                tipos_disp = sorted(df_nuevos[c_tipo_nuevos].dropna().astype(str).unique())
                sel_tipo = col_f3.multiselect("Filtrar por Tipo Afiliado", tipos_disp, default=tipos_disp)

        # --- APLICACIÓN DE FILTROS ---
        df_view = df_nuevos.copy()
        if c_pais_nuevos and sel_paises:
            df_view = df_view[df_view[c_pais_nuevos].isin(sel_paises)]
        if c_cat_nuevos and sel_cat:
            df_view = df_view[df_view[c_cat_nuevos].isin(sel_cat)]
        if c_tipo_nuevos and sel_tipo:
            df_view = df_view[df_view[c_tipo_nuevos].isin(sel_tipo)]

        # --- CRUCE CON PLAN (PRIMAS/SINIESTROS) ---
        has_finance = False
        c_plan_cia = find_col(df_plan, ['compañía', 'empresa', 'nombre'])
        c_plan_tipo = find_col(df_plan, ['tipo']) # Primas/Siniestros
        c_plan_usd = find_col(df_plan, ['usd', 'valor', 'monto'])
        
        if c_plan_cia and c_plan_tipo and c_plan_usd:
            # Pivotear Plan
            plan_pivot = df_plan.pivot_table(
                index=c_plan_cia, columns=c_plan_tipo, values=c_plan_usd, aggfunc='sum', fill_value=0
            ).reset_index()
            # Renombrar columnas del pivot para facilitar merge
            plan_pivot.columns = [str(c).strip() for c in plan_pivot.columns]
            
            # Normalizar nombres para cruce
            plan_pivot['key_merge'] = normalize_text(plan_pivot[c_plan_cia])
            if c_empresa_nuevos:
                df_view['key_merge'] = normalize_text(df_view[c_empresa_nuevos])
                
                # Merge
                # Buscamos columnas de Primas/Siniestros en el pivot
                c_primas = find_col(plan_pivot, ['primas'])
                c_siniestros = find_col(plan_pivot, ['siniestros'])
                
                cols_to_merge = ['key_merge']
                if c_primas: cols_to_merge.append(c_primas)
                if c_siniestros: cols_to_merge.append(c_siniestros)
                
                if len(cols_to_merge) > 1:
                    df_view = pd.merge(df_view, plan_pivot[cols_to_merge], on='key_merge', how='left')
                    has_finance = True
                    # Renombrar para consistencia visual
                    rename_fin = {}
                    if c_primas: rename_fin[c_primas] = 'Primas'
                    if c_siniestros: rename_fin[c_siniestros] = 'Siniestros'
                    df_view = df_view.rename(columns=rename_fin)

        # --- KPIS DASHBOARD ---
        st.markdown("### 📊 Indicadores de Nuevas Incorporaciones")
        k1, k2, k3, k4, k5 = st.columns(5)
        
        total_nuevos = len(df_view)
        paises_activos = df_view[c_pais_nuevos].nunique() if c_pais_nuevos else 0
        
        miembros = 0
        asociados = 0
        if c_tipo_nuevos:
            miembros = len(df_view[normalize_text(df_view[c_tipo_nuevos]).str.contains('miembro', na=False)])
            asociados = len(df_view[normalize_text(df_view[c_tipo_nuevos]).str.contains('asociado', na=False)])
        
        total_primas = df_view['Primas'].sum() if has_finance and 'Primas' in df_view.columns else 0
        
        k1.metric("Nuevas Empresas", total_nuevos, delta="2025 Activo")
        k2.metric("Países Detectados", paises_activos)
        k3.metric("Miembros (Core)", miembros)
        k4.metric("Asociados", asociados)
        if has_finance:
            k5.metric("Primas Estimadas", f"${total_primas:,.0f}")
        else:
            k5.metric("Finanzas", "No Cruzado")

        st.markdown("---")
        
        # --- GRÁFICOS ---
        row2_1, row2_2 = st.columns(2)
        with row2_1:
            if c_pais_nuevos and c_tipo_nuevos:
                st.markdown("**Distribución por País y Tipo**")
                df_stack = df_view.groupby([c_pais_nuevos, c_tipo_nuevos]).size().reset_index(name='Conteo')
                fig_stack = px.bar(
                    df_stack, x=c_pais_nuevos, y='Conteo', color=c_tipo_nuevos,
                    barmode='stack', template="plotly_white"
                )
                st.plotly_chart(fig_stack, use_container_width=True)
                
        with row2_2:
            if c_pais_nuevos and c_cat_nuevos:
                st.markdown("**Jerarquía: País -> Categoría**")
                fig_sun = px.sunburst(
                    df_view, path=[c_pais_nuevos, c_cat_nuevos], 
                    title="Radiografía del Mercado", color=c_pais_nuevos
                )
                st.plotly_chart(fig_sun, use_container_width=True)

        with st.expander("📋 Ver Datos Filtrados"):
            st.dataframe(df_view.drop(columns=['key_merge'], errors='ignore'), use_container_width=True)

    # ==========================================================================
    # TAB 2: DIRECTORIO & DESGLOSE (MEJORADO FULL)
    # ==========================================================================
    with tab2:
        st.header("📋 Directorio de Afiliados: Inteligencia & KPIs")
        
        # 1. Identificar Columnas Clave en Directorio (Robustez)
        
        # A) País
        c_dir_pais_op = find_col(df_dir, ['país operación', 'pais operacion', 'país op', 'operación'])
        
        # B) Categoría ALSUM (Negocio: Aseguradora, Broker...)
        # Buscamos explícitamente "Categoría ALSUM" o "Tipo de Compañía"
        c_dir_cat_alsum = find_col(df_dir, ['categoría alsum', 'categoria alsum', 'tipo de compañía'])
        
        # C) Categoría (Afiliación: Miembro vs Asociado) - SOLICITUD ESPECÍFICA
        # Buscamos estrictamente "categoria" o "categoría" primero, para evitar conflicto con "Categoría ALSUM" si es posible,
        # find_col buscará exacto primero.
        c_dir_membresia = find_col(df_dir, ['categoria', 'categoría'])
        
        # Fallback: Si 'c_dir_membresia' terminó siendo el mismo que 'c_dir_cat_alsum', intentamos buscar algo más.
        if c_dir_membresia == c_dir_cat_alsum:
             # Si son iguales, es posible que el excel solo tenga una columna "Categoría".
             # Pero si existen dos, intentamos desambiguar buscando "Tipo de Afiliado" para la membresía.
             c_alternativo = find_col(df_dir, ['tipo de afiliado', 'tipo afiliado'])
             if c_alternativo:
                 c_dir_membresia = c_alternativo

        # 2. Configuración de Filtros (AHORA SON 3)
        with st.container():
            st.markdown("#### 🔍 Filtros Avanzados")
            col_filtro_1, col_filtro_2, col_filtro_3 = st.columns(3)
            
            # Filtro 1: País Operación
            opciones_pais = []
            sel_pais_op = []
            if c_dir_pais_op:
                opciones_pais = sorted(df_dir[c_dir_pais_op].dropna().astype(str).unique())
                sel_pais_op = col_filtro_1.multiselect("🏳️ País de Operación", opciones_pais, default=opciones_pais, key="f_dir_pais")
            else:
                col_filtro_1.warning("Falta col. País")

            # Filtro 2: Categoría ALSUM (Rubro de Negocio)
            opciones_cat = []
            sel_cat_alsum = []
            if c_dir_cat_alsum:
                opciones_cat = sorted(df_dir[c_dir_cat_alsum].dropna().astype(str).unique())
                sel_cat_alsum = col_filtro_2.multiselect("🏷️ Categoría ALSUM (Rubro)", opciones_cat, default=opciones_cat, key="f_dir_cat")
            else:
                col_filtro_2.warning("Falta col. Categoría ALSUM")
                
            # Filtro 3: Tipo Afiliado (Miembro vs Asociado) -> Usando col "Categoría"
            opciones_membresia = []
            sel_membresia = []
            if c_dir_membresia:
                opciones_membresia = sorted(df_dir[c_dir_membresia].dropna().astype(str).unique())
                sel_membresia = col_filtro_3.multiselect("🤝 Categoría (Miembro/Asociado)", opciones_membresia, default=opciones_membresia, key="f_dir_mem")
            else:
                col_filtro_3.info("Columna 'Categoría' (Miembro/Asociado) no encontrada")

        # 3. Filtrado de Datos
        df_dir_filt = df_dir.copy()
        
        if c_dir_pais_op and sel_pais_op:
            df_dir_filt = df_dir_filt[df_dir_filt[c_dir_pais_op].isin(sel_pais_op)]
            
        if c_dir_cat_alsum and sel_cat_alsum:
            df_dir_filt = df_dir_filt[df_dir_filt[c_dir_cat_alsum].isin(sel_cat_alsum)]
            
        if c_dir_membresia and sel_membresia:
            df_dir_filt = df_dir_filt[df_dir_filt[c_dir_membresia].isin(sel_membresia)]

        st.divider()

        # 4. Cálculo de KPIs Dinámicos
        kpi_d1, kpi_d2, kpi_d3, kpi_d4 = st.columns(4)
        
        count_total = len(df_dir_filt)
        
        # Conteo inteligente Miembros vs Asociados usando la columna detectada
        count_miembros = 0
        count_asociados = 0
        col_para_conteo = c_dir_membresia if c_dir_membresia else c_dir_cat_alsum
        
        if col_para_conteo:
            series_norm = normalize_text(df_dir_filt[col_para_conteo])
            count_miembros = len(df_dir_filt[series_norm.str.contains('miembro', na=False)])
            count_asociados = len(df_dir_filt[series_norm.str.contains('asociado', na=False)])
        
        paises_unicos_dir = df_dir_filt[c_dir_pais_op].nunique() if c_dir_pais_op else 0

        kpi_d1.metric("Total Empresas", count_total)
        kpi_d2.metric("Países Operación", paises_unicos_dir)
        kpi_d3.metric("Miembros", count_miembros)
        kpi_d4.metric("Asociados", count_asociados)

        st.divider()

        # 5. Visualizaciones de Desglose
        col_viz1, col_viz2 = st.columns(2)
        
        with col_viz1:
            # --- PARTICIPACIÓN POR PAÍS (SOLICITADO) ---
            if c_dir_pais_op:
                st.markdown("### 🌎 Participación por País")
                conteo_pais_pie = df_dir_filt[c_dir_pais_op].value_counts().reset_index()
                conteo_pais_pie.columns = ['País', 'Cantidad']
                
                # Donut Chart
                fig_pie_pais = px.pie(
                    conteo_pais_pie, 
                    values='Cantidad', 
                    names='País', 
                    hole=0.4,
                    color_discrete_sequence=px.colors.qualitative.Prism,
                    title="Market Share Geográfico (Selección Actual)"
                )
                fig_pie_pais.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig_pie_pais, use_container_width=True)
            else:
                st.warning("No hay datos de País para graficar.")
        
        with col_viz2:
            # --- CATEGORÍAS ALSUM (RUBRO) ---
            if c_dir_cat_alsum:
                st.markdown("### 🏢 Top Categorías de Negocio")
                conteo_cat = df_dir_filt[c_dir_cat_alsum].value_counts().nlargest(10).reset_index()
                conteo_cat.columns = ['Categoría', 'Cantidad']
                fig_bar_cat = px.bar(
                    conteo_cat, 
                    x='Cantidad', 
                    y='Categoría', 
                    orientation='h', 
                    text='Cantidad',
                    color='Cantidad',
                    color_continuous_scale="Blues"
                )
                fig_bar_cat.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig_bar_cat, use_container_width=True)
            else:
                st.warning("No hay datos de Categoría para graficar.")

        # 6. Tabla de Datos
        st.subheader("Detalle de Empresas Filtradas")
        st.dataframe(df_dir_filt, use_container_width=True)

    # ==========================================================================
    # TAB 3: DATA WAREHOUSE
    # ==========================================================================
    with tab3:
        st.subheader("Auditoría de Datos")
        v = st.radio("Ver Dataset:", ["Nuevos Procesados", "Directorio Completo", "Plan 2026"], horizontal=True)
        if v == "Nuevos Procesados": st.dataframe(df_nuevos, use_container_width=True)
        elif v == "Directorio Completo": st.dataframe(df_dir, use_container_width=True)
        else: st.dataframe(df_plan, use_container_width=True)

    # ==========================================================================
    # TAB 4: COMPARATIVO Y NO AFILIADOS
    # ==========================================================================
    with tab4:
        st.header("📊 Comparativo País & KPIs")
        c1, c2, c3 = st.columns(3)
        
        # Datos para comparar
        nuevos_por_pais = pd.DataFrame()
        if 'País_Detectado' in df_nuevos.columns:
            nuevos_por_pais = df_nuevos['País_Detectado'].value_counts().reset_index()
            nuevos_por_pais.columns = ['País', 'Nuevos Afiliados']

        no_afiliados = pd.DataFrame()
        no_afiliados_por_pais = pd.DataFrame()
        
        c_plan_afiliado = find_col(df_plan, ['afiliado', 'estado'])
        c_plan_pais = find_col(df_plan, ['país', 'pais'])
        
        if c_plan_afiliado and c_plan_pais:
            no_afiliados = df_plan[normalize_text(df_plan[c_plan_afiliado]) == 'no afiliado']
            no_afiliados_por_pais = no_afiliados[c_plan_pais].value_counts().reset_index()
            no_afiliados_por_pais.columns = ['País', 'No Afiliados']

        # Métricas
        total_primas_plan = 0
        c_plan_primas = find_col(df_plan, ['primas', 'usd'])
        if c_plan_primas:
            total_primas_plan = df_plan[c_plan_primas].sum()

        with c1:
            st.metric("Primas Totales Plan", f"${total_primas_plan:,.0f}")
        with c2:
            st.metric("Total Nuevos (Global)", len(df_nuevos))
            if not nuevos_por_pais.empty: st.dataframe(nuevos_por_pais, use_container_width=True, height=150)
        with c3:
            st.metric("Oportunidad (No Afiliados)", len(no_afiliados))
            if not no_afiliados_por_pais.empty: st.dataframe(no_afiliados_por_pais, use_container_width=True, height=150)

        st.divider()
        st.markdown("### 🌍 Mapa de Oportunidad (No Afiliados)")
        if not no_afiliados_por_pais.empty:
            fig_map = px.choropleth(
                no_afiliados_por_pais, locations='País', locationmode='country names',
                color='No Afiliados', color_continuous_scale="Reds",
                title="Concentración Geográfica de No Afiliados"
            )
            st.plotly_chart(fig_map, use_container_width=True)

        st.markdown("### 🔎 Listado de No Afiliados")
        st.dataframe(no_afiliados, use_container_width=True)

    # ==========================================================================
    # TAB 5: LABORATORIO IA
    # ==========================================================================
    with tab5:
        st.header("🤖 Laboratorio de Inteligencia Artificial")
        st.markdown("Analiza tus tablas de datos usando GPT-4o-mini.")
        
        c_ia1, c_ia2 = st.columns([1, 2])
        with c_ia1:
            st.info("Configuración del Análisis")
            dataset_opt = st.selectbox("Fuente de Datos", ["Nuevos Afiliados", "Directorio Filtrado", "No Afiliados"])
            user_prompt = st.text_area("Instrucción para la IA:", "Dame 3 estrategias clave basadas en estos datos.")
            btn_ia = st.button("✨ Generar Insights", type="primary")
        
        with c_ia2:
            if btn_ia:
                contexto = ""
                if dataset_opt == "Nuevos Afiliados":
                    contexto = df_nuevos.describe(include='all').to_string() + "\n\n" + df_nuevos.head(20).to_string()
                elif dataset_opt == "Directorio Filtrado":
                    # Usamos df_dir por simplicidad de scope local, idealmente usar el filtrado
                    contexto = df_dir.head(50).to_string()
                elif dataset_opt == "No Afiliados":
                    if 'no_afiliados' in locals() and not no_afiliados.empty:
                        contexto = no_afiliados.head(30).to_string()
                    else:
                        contexto = "No hay datos de No Afiliados disponibles."
                
                with st.spinner("Analizando..."):
                    resultado = consultar_gpt4(api_key, user_prompt, contexto)
                    st.markdown(resultado)

    # ==========================================================================
    # BOTÓN DE DESCARGA GLOBAL
    # ==========================================================================
    with col_download:
        # Preparamos hojas extra
        extra_sheets = {}
        if 'no_afiliados' in locals():
            extra_sheets['No_Afiliados'] = no_afiliados
        
        excel_data = generate_excel_download(df_nuevos, extra_sheets=extra_sheets)
        
        st.download_button(
            label="📥 Descargar Reporte Maestro (.xlsx)",
            data=excel_data,
            file_name="ALSUM_Reporte_Estrategico_2026.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            type="primary"
        )

if __name__ == "__main__":
    main()