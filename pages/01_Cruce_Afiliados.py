import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import io
import utils  # Se asume que tu librería utils.py existe y funciona
from openai import OpenAI

# ==============================================================================
# 1. CONFIGURACIÓN ESTRATÉGICA Y ESTILOS
# ==============================================================================
st.set_page_config(
    page_title="ALSUM 360 | Enterprise Command Center",
    layout="wide",
    page_icon="💎",
    initial_sidebar_state="expanded"
)

# Estilos CSS para dar sensación "Premium" y profesional
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
# 2. FUNCIONES CORE Y UTILIDADES DE NORMALIZACIÓN
# ==============================================================================

def normalize_text_cols(df):
    """
    Convierte todas las columnas de texto a mayúsculas y elimina espacios extra
    para asegurar coincidencias precisas (Case Insensitive).
    """
    for col in df.select_dtypes(include=['object']):
        try:
            df[col] = df[col].astype(str).str.strip().str.upper()
            # Convertir 'NAN' o 'NAT' string a None real para limpieza
            df[col] = df[col].replace(['NAN', 'NAT', 'NONE'], None)
        except:
            pass
    return df

def find_column_fuzzy(columns, keywords):
    """
    Busca una columna que contenga las palabras clave, ignorando mayúsculas/minúsculas.
    Retorna el nombre real de la columna o None.
    """
    cols_upper = [str(c).upper() for c in columns]
    keywords_upper = [k.upper() for k in keywords]
    
    for i, col in enumerate(cols_upper):
        # Si todas las palabras clave están en el nombre de la columna
        if all(k in col for k in keywords_upper):
            return columns[i]
    return None

def deduplicate_columns(df):
    df = df.copy()
    cols = pd.Series(df.columns)
    for dup in cols[cols.duplicated()].unique():
        cols[cols[dup].index] = [dup + '.' + str(i) if i != 0 else dup for i in range(sum(cols == dup))]
    df.columns = cols
    return df

def clean_column_names(df):
    # Elimina espacios y normaliza nombres de columnas
    df.columns = [str(c).strip().replace(' ', '_') for c in df.columns]
    return df

@st.cache_data(ttl=3600, show_spinner=False)
def load_data_master():
    plan_path = utils.get_file_path("plan_2026.csv")
    dir_path = utils.get_file_path("Directorio_Afiliados_2025.xlsx")
    
    # 1. Directorio
    try:
        df_dir = utils.load_excel_sheet(dir_path, sheet_name="Directorio 2025")
        df_dir = deduplicate_columns(df_dir)
        # Normalización para precisión
        df_dir = normalize_text_cols(df_dir)
    except Exception as e:
        return None, None, None, f"Error cargando Directorio: {str(e)}"
    
    # 2. Nuevos
    try:
        xl = pd.ExcelFile(dir_path)
        sheet_candidates = [s for s in xl.sheet_names if 'nuevos' in s.lower()]
        if sheet_candidates:
            df_nuevos = pd.read_excel(dir_path, sheet_name=sheet_candidates[0])
            df_nuevos = clean_column_names(df_nuevos)
            
            # Renombrado Estratégico
            rename_map = {
                'Tipo_de_Compañía': 'Categoria',
                'Compañía': 'Empresa',
                'Tipo_de_Afiliado': 'Tipo_de_Afiliado',
                'Tipo_Afiliado': 'Tipo_de_Afiliado',
            }
            df_nuevos = df_nuevos.rename(columns=rename_map)
            df_nuevos = deduplicate_columns(df_nuevos)
            
            # Normalización antes del cruce
            df_nuevos = normalize_text_cols(df_nuevos)
            
            # Enriquecer con país desde Directorio
            col_pais_dir = find_column_fuzzy(df_dir.columns, ['PAIS'])
            col_empresa_dir = find_column_fuzzy(df_dir.columns, ['EMPRESA']) or find_column_fuzzy(df_dir.columns, ['COMPAÑÍA'])
            
            if col_pais_dir and col_empresa_dir and 'Empresa' in df_nuevos.columns:
                mapa_pais = dict(zip(df_dir[col_empresa_dir], df_dir[col_pais_dir]))
                df_nuevos['País_Detectado'] = df_nuevos['Empresa'].map(mapa_pais).fillna('SIN ASIGNAR')
            else:
                df_nuevos['País_Detectado'] = 'NO DATA'
        else:
            return None, None, None, "No se encontró pestaña 'Nuevos'."
    except Exception as e:
        return None, None, None, f"Error cargando Nuevos: {str(e)}"
    
    # 3. Plan 2026
    df_plan, err = utils.load_plan_accion_procesado(plan_path)
    if err: return None, None, None, f"Error Plan: {err}"
    if df_plan is not None:
        df_plan = normalize_text_cols(df_plan)
    
    return df_dir, df_nuevos, df_plan, None

def generate_excel_download(df_nuevos, extra_sheets=None):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
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
        return "⚠️ No se detectó la API Key."
    
    client = OpenAI(api_key=api_key)
    full_prompt = f"""
    Actúa como un Analista Senior de Estrategia y Seguros. 
    Analiza los siguientes datos resumidos:
    {data_context}
    
    PREGUNTA DEL USUARIO: {prompt}
    
    Responde con insights accionables. Sé directo y profesional en Español.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": full_prompt}],
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error API: {str(e)}"

# ==============================================================================
# 3. INTERFAZ DE USUARIO (MAIN)
# ==============================================================================

def main():
    # --- SIDEBAR GLOBAL ---
    with st.sidebar:
        st.header("⚙️ Configuración & IA")
        st.info("Sistema cargado. Modo Precisión Activado (Ignora mayúsculas/minúsculas).")

    col_logo, col_title, col_download = st.columns([1, 4, 2])
    with col_title:
        st.title("ALSUM Analytics | Estrategia 360º")
        st.markdown("**Gestión de Directorio, Inteligencia de Mercado y KPIs**")
        
    with st.spinner("🔄 Procesando y normalizando matrices de datos..."):
        df_dir, df_nuevos, df_plan, error_msg = load_data_master()
        
    if error_msg:
        st.error(error_msg)
        st.stop()
        
    # --- DEFINICIÓN DE TABS (NUEVO ORDEN) ---
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🌍 Dashboard Nuevos",
        "📋 Directorio & Desglose",
        "📂 Data Warehouse",
        "📈 Comparativo País & KPIs",
        "🤖 Laboratorio IA"
    ])

    # ==========================================================================
    # TAB 1: DASHBOARD NUEVOS (ORIGINAL MEJORADO)
    # ==========================================================================
    with tab1:
        st.subheader("Panel de Control: Nuevas Incorporaciones")
        
        # Filtros
        with st.expander("🔎 Filtros de Nuevos Afiliados", expanded=True):
            col_f1, col_f2, col_f3 = st.columns(3)
            
            paises_disp = sorted(df_nuevos['País_Detectado'].unique()) if 'País_Detectado' in df_nuevos.columns else []
            sel_paises = col_f1.multiselect("Filtrar por País", paises_disp, default=paises_disp)
            
            col_cat_nuevos = find_column_fuzzy(df_nuevos.columns, ['CATEGORIA'])
            cats_disp = sorted(df_nuevos[col_cat_nuevos].unique()) if col_cat_nuevos else []
            sel_cat = col_f2.multiselect("Filtrar por Categoría", cats_disp, default=cats_disp)
            
            col_tipo_nuevos = find_column_fuzzy(df_nuevos.columns, ['TIPO', 'AFILIADO'])
            tipos_disp = sorted(df_nuevos[col_tipo_nuevos].unique()) if col_tipo_nuevos else []
            sel_tipo = col_f3.multiselect("Filtrar por Tipo", tipos_disp, default=tipos_disp)

        # Aplicación Filtros
        df_view = df_nuevos.copy()
        if sel_paises and 'País_Detectado' in df_view.columns:
            df_view = df_view[df_view['País_Detectado'].isin(sel_paises)]
        if sel_cat and col_cat_nuevos:
            df_view = df_view[df_view[col_cat_nuevos].isin(sel_cat)]
        if sel_tipo and col_tipo_nuevos:
            df_view = df_view[df_view[col_tipo_nuevos].isin(sel_tipo)]

        # Cruce Financiero (si existe en Plan)
        has_finance = False
        col_plan_comp = find_column_fuzzy(df_plan.columns, ['COMPAÑÍA']) or find_column_fuzzy(df_plan.columns, ['EMPRESA'])
        if col_plan_comp and 'USD' in df_plan.columns:
             # Agrupar plan
             plan_pivot = df_plan.pivot_table(
                index=[col_plan_comp], columns='Tipo', values='USD', aggfunc='sum', fill_value=0
             ).reset_index()
             
             if 'Empresa' in df_view.columns:
                 df_view = pd.merge(
                    df_view,
                    plan_pivot,
                    left_on='Empresa', right_on=col_plan_comp, how='left'
                 )
                 has_finance = True

        # KPIs Visuales
        st.markdown("### 📊 Indicadores Clave")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Nuevas Empresas", len(df_view))
        k2.metric("Países", df_view['País_Detectado'].nunique() if 'País_Detectado' in df_view.columns else 0)
        
        miembros_nuevos = 0
        if col_tipo_nuevos:
            miembros_nuevos = len(df_view[df_view[col_tipo_nuevos].astype(str).str.contains('MIEMBRO', na=False)])
        k3.metric("Miembros Nuevos", miembros_nuevos)
        
        if has_finance and 'Primas' in df_view.columns:
             total_primas = df_view['Primas'].sum()
             k4.metric("Primas Estimadas", f"${total_primas:,.0f}")
        else:
             k4.metric("Finanzas", "No Data")

        # Gráficos
        row2_1, row2_2 = st.columns(2)
        with row2_1:
            if 'País_Detectado' in df_view.columns:
                df_count = df_view['País_Detectado'].value_counts().reset_index()
                df_count.columns = ['País', 'Cantidad']
                fig_bar = px.bar(df_count, x='País', y='Cantidad', title="Nuevos por País", color='Cantidad')
                st.plotly_chart(fig_bar, use_container_width=True)
        
        with row2_2:
            if col_cat_nuevos:
                fig_sun = px.sunburst(df_view, path=['País_Detectado', col_cat_nuevos], title="Jerarquía Regional")
                st.plotly_chart(fig_sun, use_container_width=True)

    # ==========================================================================
    # TAB 2: DIRECTORIO & DESGLOSE (MEJORADO Y MOVIDO)
    # ==========================================================================
    with tab2:
        st.header("📋 Directorio & Desglose de Afiliados")
        st.markdown("Análisis detallado de la base actual de afiliados con filtros precisos.")

        # 1. Identificación Inteligente de Columnas (Case Insensitive)
        # Buscamos columnas que contengan 'PAIS' y ('OPERACION' o 'SEDE')
        col_pais_operacion = find_column_fuzzy(df_dir.columns, ['PAIS', 'OPERACION'])
        if not col_pais_operacion:
            col_pais_operacion = find_column_fuzzy(df_dir.columns, ['PAIS', 'SEDE']) 
        
        # Buscamos Categoría ALSUM
        col_cat_alsum = find_column_fuzzy(df_dir.columns, ['CATEGORIA', 'ALSUM'])
        if not col_cat_alsum: # Fallback si no dice ALSUM explícitamente pero es categoría
             col_cat_alsum = find_column_fuzzy(df_dir.columns, ['CATEGORIA'])
        
        # Buscamos Tipo (Miembro/Asociado) para los KPIs
        col_tipo_afiliado = find_column_fuzzy(df_dir.columns, ['TIPO', 'AFILIADO'])
        if not col_tipo_afiliado:
            col_tipo_afiliado = find_column_fuzzy(df_dir.columns, ['CLASE'])

        # 2. Configuración de Filtros
        c_filter1, c_filter2, c_filter3 = st.columns(3)
        
        # Filtro País
        opciones_pais = []
        if col_pais_operacion:
            opciones_pais = sorted(df_dir[col_pais_operacion].dropna().unique())
            sel_pais_dir = c_filter1.multiselect("🏳️ País Sede / Operación", opciones_pais, default=opciones_pais)
        else:
            st.warning("No se detectó columna de País Sede en el Directorio.")
            sel_pais_dir = []

        # Filtro Categoría ALSUM
        opciones_cat = []
        if col_cat_alsum:
            opciones_cat = sorted(df_dir[col_cat_alsum].dropna().unique())
            sel_cat_dir = c_filter2.multiselect("🏷️ Categoría ALSUM", opciones_cat, default=opciones_cat)
        else:
            st.warning("No se detectó columna Categoría ALSUM.")
            sel_cat_dir = []
            
        # Filtro Adicional (Tipo) si existe
        opciones_tipo = []
        sel_tipo_dir = []
        if col_tipo_afiliado:
            opciones_tipo = sorted(df_dir[col_tipo_afiliado].dropna().unique())
            sel_tipo_dir = c_filter3.multiselect("📌 Tipo Afiliado", opciones_tipo, default=opciones_tipo)

        # 3. Lógica de Filtrado del DataFrame
        df_dir_filt = df_dir.copy()
        
        if col_pais_operacion and sel_pais_dir:
            df_dir_filt = df_dir_filt[df_dir_filt[col_pais_operacion].isin(sel_pais_dir)]
            
        if col_cat_alsum and sel_cat_dir:
            df_dir_filt = df_dir_filt[df_dir_filt[col_cat_alsum].isin(sel_cat_dir)]
            
        if col_tipo_afiliado and sel_tipo_dir:
            df_dir_filt = df_dir_filt[df_dir_filt[col_tipo_afiliado].isin(sel_tipo_dir)]

        st.markdown("---")

        # 4. KPIs Dinámicos (Solicitud Específica)
        # Calculamos los conteos basados en la data ya filtrada
        total_empresas = len(df_dir_filt)
        
        total_miembros = 0
        total_asociados = 0
        
        if col_tipo_afiliado:
            # Buscamos strings que contengan 'MIEMBRO' o 'ASOCIADO' (ya está todo en mayúsculas por normalización)
            total_miembros = df_dir_filt[col_tipo_afiliado].str.contains('MIEMBRO', na=False).sum()
            total_asociados = df_dir_filt[col_tipo_afiliado].str.contains('ASOCIADO', na=False).sum()

        st.markdown("### 📊 Indicadores del Desglose")
        kp1, kp2, kp3, kp4 = st.columns(4)
        
        kp1.metric("Total Empresas Listadas", total_empresas)
        kp2.metric("Miembros (Core)", int(total_miembros))
        kp3.metric("Asociados", int(total_asociados))
        
        # % de Representación del filtro sobre el total global
        pct_global = (total_empresas / len(df_dir) * 100) if len(df_dir) > 0 else 0
        kp4.metric("% del Directorio Total", f"{pct_global:.1f}%")

        # 5. Visualización Gráfica y Tabla
        col_viz, col_data = st.columns([1, 2])
        
        with col_viz:
            st.markdown("#### Distribución Visual")
            if col_cat_alsum:
                df_pie = df_dir_filt[col_cat_alsum].value_counts().reset_index()
                df_pie.columns = ['Categoría', 'Cantidad']
                fig_pie = px.pie(df_pie, values='Cantidad', names='Categoría', hole=0.4, color_discrete_sequence=px.colors.sequential.RdBu)
                st.plotly_chart(fig_pie, use_container_width=True)
            elif col_pais_operacion:
                df_bar = df_dir_filt[col_pais_operacion].value_counts().reset_index().head(10)
                df_bar.columns = ['País', 'Cantidad']
                fig_bar = px.bar(df_bar, x='Cantidad', y='País', orientation='h')
                st.plotly_chart(fig_bar, use_container_width=True)
                
        with col_data:
            st.markdown("#### Detalle de Empresas")
            st.dataframe(df_dir_filt, use_container_width=True, height=400)

    # ==========================================================================
    # TAB 3: DATA WAREHOUSE
    # ==========================================================================
    with tab3:
        st.subheader("Auditoría de Datos")
        v = st.radio("Ver Dataset:", ["Nuevos Procesados", "Directorio Completo", "Plan Estratégico"], horizontal=True)
        if v == "Nuevos Procesados": st.dataframe(df_nuevos, use_container_width=True)
        elif v == "Directorio Completo": st.dataframe(df_dir, use_container_width=True)
        else: st.dataframe(df_plan, use_container_width=True)

    # ==========================================================================
    # TAB 4: COMPARATIVO PAÍS & KPIs (MANTENIDO)
    # ==========================================================================
    with tab4:
        st.header("📊 Comparativo País & KPIs")
        c1, c2, c3 = st.columns(3)
        
        # Calcular No Afiliados desde el Plan
        no_afiliados = pd.DataFrame()
        col_afiliado_plan = find_column_fuzzy(df_plan.columns, ['AFILIADO'])
        col_pais_plan = find_column_fuzzy(df_plan.columns, ['PAIS'])
        
        if col_afiliado_plan and col_pais_plan:
            no_afiliados = df_plan[df_plan[col_afiliado_plan].astype(str) == 'NO AFILIADO']
            no_afiliados_por_pais = no_afiliados[col_pais_plan].value_counts().reset_index()
            no_afiliados_por_pais.columns = ['País', 'No Afiliados']
        else:
            no_afiliados_por_pais = pd.DataFrame()

        with c1:
            total_primas_plan = df_plan['Primas'].sum() if 'Primas' in df_plan.columns else 0
            st.metric("Primas Totales Plan", f"${total_primas_plan:,.0f}")
        with c2:
            st.metric("Total Nuevos Detectados", len(df_nuevos))
        with c3:
            st.metric("Oportunidades (No Afiliados)", len(no_afiliados))

        st.markdown("### 🌍 Mapa de Oportunidad (No Afiliados)")
        if not no_afiliados_por_pais.empty:
            fig_map = px.choropleth(
                no_afiliados_por_pais, locations='País', locationmode='country names',
                color='No Afiliados', color_continuous_scale="Reds",
                title="Concentración de No Afiliados por País"
            )
            st.plotly_chart(fig_map, use_container_width=True)
            
            st.markdown("#### Listado de Oportunidades")
            st.dataframe(no_afiliados, use_container_width=True)

    # ==========================================================================
    # TAB 5: LABORATORIO IA
    # ==========================================================================
    with tab5:
        st.header("🤖 Laboratorio de Inteligencia Artificial")
        st.markdown("Analiza los datos cargados usando GPT-4o-mini.")
        
        c_ia1, c_ia2 = st.columns([1, 2])
        with c_ia1:
            dataset_opt = st.selectbox("Fuente de Datos", ["Directorio (Filtrado)", "Nuevos Afiliados", "No Afiliados"])
            user_prompt = st.text_area("Consulta:", "Dame un análisis FODA y sugerencias estratégicas.")
            api_key = utils.get_api_key() # Asegúrate que utils tenga esta función
            btn_ia = st.button("✨ Generar Análisis")
        
        with c_ia2:
            if btn_ia:
                if not api_key:
                    st.error("❌ Falta la API Key.")
                else:
                    contexto = ""
                    if dataset_opt == "Directorio (Filtrado)":
                        contexto = df_dir_filt.head(40).to_string()
                    elif dataset_opt == "Nuevos Afiliados":
                        contexto = df_nuevos.describe(include='all').to_string()
                    elif dataset_opt == "No Afiliados":
                        contexto = no_afiliados.head(40).to_string()
                    
                    with st.spinner("Analizando..."):
                        res = consultar_gpt4(api_key, user_prompt, contexto)
                        st.markdown(res)

    # BOTÓN DE DESCARGA GLOBAL
    with col_download:
        no_afiliados_dl = no_afiliados if not no_afiliados.empty else pd.DataFrame()
        excel_data = generate_excel_download(df_nuevos, extra_sheets={'No_Afiliados': no_afiliados_dl})
        st.download_button(
            label="📥 Descargar Reporte Completo (.xlsx)",
            data=excel_data,
            file_name="ALSUM_Master_Report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            type="primary"
        )

if __name__ == "__main__":
    main()