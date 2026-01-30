import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import utils  # Se asume que tu archivo utils.py contiene fuzzy_merge y funciones de carga

# ==============================================================================
# 1. CONFIGURACIÓN ESTRATÉGICA DE LA PÁGINA
# ==============================================================================
st.set_page_config(
    page_title="ALSUM Analytics | Estrategia 2025-2026",
    layout="wide",
    page_icon="💎",
    initial_sidebar_state="expanded"
)

# --- CSS PROFESIONAL (ESTILO EJECUTIVO) ---
st.markdown("""
<style>
    /* Fondo y fuentes corporativas */
    .stApp {
        background-color: #FAFAFA;
    }
    h1, h2, h3 {
        color: #003366; /* Azul Oscuro Corporativo */
        font-family: 'Segoe UI', sans-serif;
    }
    /* Tarjetas de Métricas */
    div[data-testid="stMetric"] {
        background-color: #ffffff;
        border-left: 5px solid #003366;
        padding: 15px;
        box-shadow: 2px 2px 8px rgba(0,0,0,0.08);
        border-radius: 8px;
    }
    /* Pestañas */
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 1.1rem;
        font-weight: 600;
    }
    /* Tablas */
    div[data-testid="stDataFrame"] {
        border: 1px solid #e0e0e0;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. FUNCIONES DE CARGA Y LIMPIEZA ROBUSTA (CORE)
# ==============================================================================

def deduplicate_columns(df):
    """
    SOLUCIÓN CRÍTICA AL ERROR 'ValueError: Country label is not unique'.
    Si Pandas encuentra dos columnas con el mismo nombre (ej. 'País'), 
    renombra la segunda a 'País.1' para evitar que la app colapse.
    """
    df = df.copy()
    cols = pd.Series(df.columns)
    for dup in cols[cols.duplicated()].unique():
        cols[cols[dup].index] = [dup + '.' + str(i) if i != 0 else dup for i in range(sum(cols == dup))]
    df.columns = cols
    return df

def clean_column_names(df):
    """Limpia nombres de columnas: quita espacios y caracteres extraños."""
    df.columns = [str(c).strip().replace(' ', '_') for c in df.columns]
    return df

@st.cache_data(ttl=3600, show_spinner=False)
def load_data_master():
    """
    Carga todos los datos necesarios, limpia espacios, elimina columnas vacías
    y normaliza los nombres para que los filtros funcionen.
    """
    # Definir Rutas (Usando utils)
    plan_path = utils.get_file_path("plan_2026.csv")
    dir_path = utils.get_file_path("Directorio_Afiliados_2025.xlsx")
    
    # --- 1. CARGAR DIRECTORIO GENERAL ---
    try:
        df_dir = utils.load_excel_sheet(dir_path, sheet_name="Directorio 2025")
        # Asegurar columnas únicas inmediatamente
        df_dir = deduplicate_columns(df_dir)
    except Exception as e:
        return None, None, None, f"Error cargando Directorio: {str(e)}"

    # --- 2. CARGAR NUEVOS 2025 (Búsqueda inteligente y Renombrado) ---
    try:
        # Intentar cargar la hoja buscando 'nuevos' en el nombre
        xl = pd.ExcelFile(dir_path)
        sheet_candidates = [s for s in xl.sheet_names if 'nuevos' in s.lower()]
        
        if sheet_candidates:
            df_nuevos = pd.read_excel(dir_path, sheet_name=sheet_candidates[0])
            
            # Limpieza básica de cabeceras
            df_nuevos = clean_column_names(df_nuevos)
            
            # --- FILTRO 1: ELIMINAR COLUMNAS BASURA (Unnamed) ---
            cols_validas = [c for c in df_nuevos.columns if 'unnamed' not in c.lower()]
            df_nuevos = df_nuevos[cols_validas]
            
            # --- FILTRO 2: RENOMBRAR TUS COLUMNAS ESPECÍFICAS ---
            # Esto conecta tus datos reales con la lógica del dashboard
            rename_map = {
                'Tipo_de_Compañía': 'Categoria',  # Estandarizamos a Categoria
                'Tipo_de_Compania': 'Categoria',
                'Compañía': 'Empresa',
                'Compania': 'Empresa',
                # 'Tipo_Afiliado' ya suele venir bien, pero por si acaso
                'Tipo_Afiliado': 'Tipo_Afiliado' 
            }
            df_nuevos = df_nuevos.rename(columns=rename_map)
            
            # Deduplicar por seguridad
            df_nuevos = deduplicate_columns(df_nuevos)
        else:
            return None, None, None, "No se encontró la pestaña de 'Nuevos' en el Excel."
        
    except Exception as e:
        return None, None, None, f"Error cargando Nuevos: {str(e)}"

    # --- 3. CARGAR PLAN 2026 ---
    df_plan, err = utils.load_plan_accion_procesado(plan_path)
    if err:
        return None, None, None, f"Error Plan 2026: {err}"
    
    df_plan = deduplicate_columns(df_plan)

    return df_dir, df_nuevos, df_plan, None

@st.cache_data(show_spinner=False)
def perform_safe_match(df_plan, df_dir, left_col, right_col, threshold):
    """
    Función de Cruce Seguro:
    Añade sufijos '_Plan' y '_Dir' a las columnas ANTES de cruzar.
    Esto evita el error de columnas duplicadas (como País) en el resultado.
    """
    # 1. Crear copias con sufijos en las columnas
    df_plan_safe = df_plan.copy().add_suffix('_Plan')
    df_dir_safe = df_dir.copy().add_suffix('_Dir')
    
    # 2. Ajustar los nombres de las columnas llave para el cruce
    left_col_safe = f"{left_col}_Plan"
    right_col_safe = f"{right_col}_Dir"

    # 3. Ejecutar fuzzy_merge (usando la función de tu utils.py)
    result = utils.fuzzy_merge(
        df_left=df_plan_safe,
        df_right=df_dir_safe,
        left_on=left_col_safe,
        right_on=right_col_safe,
        threshold=threshold
    )
    
    # 4. Limpieza final de columnas duplicadas en el resultado
    return deduplicate_columns(result)

# ==============================================================================
# 3. INTERFAZ DE USUARIO (MAIN)
# ==============================================================================

def main():
    # --- HEADER ---
    col_logo, col_title = st.columns([1, 5])
    with col_title:
        st.title("Centro de Inteligencia y Estrategia ALSUM")
        st.markdown("**Analítica de Nuevas Incorporaciones 2025 y Radar de Oportunidades 2026**")
    
    st.markdown("---")

    # --- CARGA DE DATOS ---
    with st.spinner("🔄 Sincronizando datos maestros..."):
        df_dir, df_nuevos, df_plan, error_msg = load_data_master()

    if error_msg:
        st.error(f"⚠️ Alerta Crítica: {error_msg}")
        st.stop()

    # --- PESTAÑAS ---
    tab1, tab2, tab3 = st.tabs(["📊 KPI's Nuevos 2025", "🚀 Cruce Plan 2026 (Radar)", "📂 Explorador de Datos"])

    # ==========================================================================
    # TAB 1: ANÁLISIS DE NUEVOS AFILIADOS (CON TUS FILTROS)
    # ==========================================================================
    with tab1:
        st.subheader("Desempeño de Nuevas Afiliaciones")
        
        # Validación de columnas necesarias tras el renombrado
        # Se buscan 'Categoria' y 'Tipo_Afiliado' que renombramos en load_data_master
        col_cat = 'Categoria' if 'Categoria' in df_nuevos.columns else df_nuevos.columns[1]
        col_tipo = 'Tipo_Afiliado' if 'Tipo_Afiliado' in df_nuevos.columns else df_nuevos.columns[2]

        # --- PANEL LATERAL DE FILTROS ---
        with st.expander("🔎 Filtros Avanzados", expanded=True):
            f1, f2 = st.columns(2)
            with f1:
                # Filtro Categoria (Basado en Tipo_de_Compañía)
                clean_cats = [x for x in df_nuevos[col_cat].astype(str).unique() if x != 'nan']
                options_cat = ["Todas"] + sorted(clean_cats)
                sel_cat = st.selectbox("Filtrar por Categoría:", options_cat)
            with f2:
                # Filtro Tipo Afiliado
                clean_tipos = [x for x in df_nuevos[col_tipo].astype(str).unique() if x != 'nan']
                options_tipo = ["Todos"] + sorted(clean_tipos)
                sel_tipo = st.selectbox("Filtrar por Tipo de Afiliado:", options_tipo)

        # --- APLICAR FILTROS AL DATAFRAME ---
        df_view = df_nuevos.copy()
        
        if sel_cat != "Todas":
            df_view = df_view[df_view[col_cat].astype(str) == sel_cat]
        
        if sel_tipo != "Todos":
            df_view = df_view[df_view[col_tipo].astype(str) == sel_tipo]

        # --- MÉTRICAS ---
        st.markdown("### 📈 Indicadores Clave")
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        
        total_view = len(df_view)
        
        # Conteo Miembros vs Asociados (Robusto a mayúsculas/minúsculas)
        count_miembros = len(df_view[df_view[col_tipo].astype(str).str.contains('Miembro', case=False, na=False)])
        count_asociados = len(df_view[df_view[col_tipo].astype(str).str.contains('Asociado', case=False, na=False)])
        
        # Sumar valor membresía si existe la columna
        val_membresia = 0
        if 'Valor_membresía_2025' in df_view.columns:
             val_membresia = pd.to_numeric(df_view['Valor_membresía_2025'], errors='coerce').sum()

        kpi1.metric("Total Registros", total_view)
        kpi2.metric("Miembros (Core)", count_miembros)
        kpi3.metric("Asociados", count_asociados)
        kpi4.metric("Valor Total (USD)", f"${val_membresia:,.0f}")

        # --- GRÁFICOS ---
        st.markdown("---")
        g1, g2 = st.columns([1, 1])
        
        with g1:
            st.markdown("#### 🍩 Distribución por Tipo")
            if not df_view.empty:
                agg_tipo = df_view[col_tipo].value_counts().reset_index()
                agg_tipo.columns = ['Tipo', 'Cantidad']
                fig_pie = px.pie(agg_tipo, names='Tipo', values='Cantidad', hole=0.4, 
                                 color_discrete_sequence=px.colors.qualitative.Set3)
                st.plotly_chart(fig_pie, use_container_width=True)
            else:
                st.info("Sin datos para graficar.")

        with g2:
            st.markdown("#### 📊 Distribución por Categoría")
            if not df_view.empty:
                agg_cat = df_view[col_cat].value_counts().reset_index()
                agg_cat.columns = ['Categoria', 'Cantidad']
                fig_bar = px.bar(agg_cat, x='Cantidad', y='Categoria', orientation='h', 
                                 text_auto=True, color='Cantidad', color_continuous_scale='Blues')
                fig_bar.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig_bar, use_container_width=True)
            else:
                st.info("Sin datos para graficar.")
        
        # Tabla detallada al final
        with st.expander("Ver Detalle de Datos Filtrados"):
            st.dataframe(df_view, use_container_width=True)


    # ==========================================================================
    # TAB 2: CRUCE INTELIGENTE (SAFE MATCH)
    # ==========================================================================
    with tab2:
        st.subheader("🎯 Radar de Oportunidades (Plan vs Directorio)")
        st.info("Este módulo usa Inteligencia Artificial (Fuzzy Matching) para cruzar las bases de datos.")
        
        # Selectores de Configuración
        c_conf1, c_conf2, c_conf3 = st.columns([1, 1, 1])
        
        # Autoselección de columnas si es posible
        idx_plan = next((i for i, c in enumerate(df_plan.columns) if c.lower() in ['nombre', 'empresa', 'company']), 0)
        idx_dir = next((i for i, c in enumerate(df_dir.columns) if c.lower() in ['nombre', 'empresa', 'empresa']), 0)
        
        with c_conf1:
            col_plan_sel = st.selectbox("Columna Nombre (Plan)", df_plan.columns, index=idx_plan)
        with c_conf2:
            col_dir_sel = st.selectbox("Columna Nombre (Directorio)", df_dir.columns, index=idx_dir)
        with c_conf3:
            threshold = st.slider("Sensibilidad de Coincidencia (%)", 60, 100, 88)

        # Botón de Acción
        if st.button("⚡ Ejecutar Cruce Inteligente", type="primary"):
            with st.spinner("Analizando matrices de datos y resolviendo conflictos..."):
                
                # EJECUCIÓN DEL CRUCE SEGURO (SAFE MATCH)
                # Esto añade _Plan y _Dir a las columnas para que 'País' no se duplique
                df_cruce = perform_safe_match(df_plan, df_dir, col_plan_sel, col_dir_sel, threshold)
                
                # Lógica: Si 'match_name' tiene valor, es cliente. Si es NaN, es oportunidad.
                ya_clientes = df_cruce[df_cruce['match_name'].notna()]
                oportunidades = df_cruce[df_cruce['match_name'].isna()]
                
                # --- KPIS DEL CRUCE ---
                m1, m2, m3 = st.columns(3)
                m1.metric("Empresas en Plan", len(df_plan))
                m2.metric("✅ Ya son Afiliados", len(ya_clientes))
                m3.metric("🎯 Oportunidades Netas", len(oportunidades), delta="Potencial Comercial", delta_color="normal")
                
                st.markdown("---")
                
                # --- ANÁLISIS DE OPORTUNIDADES ---
                col_res1, col_res2 = st.columns([2, 1])
                
                with col_res1:
                    st.markdown("#### Mapa Geográfico de Oportunidades")
                    
                    # Buscar columna País en el resultado (ahora tiene sufijo _Plan)
                    cols_pais_posibles = [c for c in oportunidades.columns if ('pais' in c.lower() or 'país' in c.lower()) and '_Plan' in c]
                    
                    if cols_pais_posibles:
                        col_pais_target = cols_pais_posibles[0]
                        st.caption(f"Usando columna geográfica: {col_pais_target}")
                        
                        # Agrupación
                        opp_geo = oportunidades[col_pais_target].value_counts().reset_index()
                        opp_geo.columns = ['País', 'Cantidad']
                        
                        # GRÁFICO TREE MAP
                        fig_tree = px.treemap(
                            opp_geo, 
                            path=['País'], 
                            values='Cantidad',
                            color='Cantidad',
                            color_continuous_scale='Reds'
                        )
                        st.plotly_chart(fig_tree, use_container_width=True)
                        
                        # --- TABLA DE DATOS PRIORIZADA ---
                        st.markdown("#### Listado Priorizado")
                        col_nombre_plan_safe = f"{col_plan_sel}_Plan"
                        
                        cols_final = [col_nombre_plan_safe, col_pais_target]
                        # Filtrar solo columnas que existen
                        cols_final = [c for c in cols_final if c in oportunidades.columns]
                        
                        st.dataframe(oportunidades[cols_final], use_container_width=True, hide_index=True)
                        
                    else:
                        st.warning("No se detectó columna de 'País' en el Plan para graficar. Mostrando tabla completa.")
                        st.dataframe(oportunidades, use_container_width=True)

                with col_res2:
                    st.success("💡 **Acción Comercial**")
                    st.write(f"Se han detectado **{len(oportunidades)} empresas** objetivo.")
                    st.write("Descarga el archivo y compártelo con el equipo.")
                    
                    # Botón Descarga CSV
                    csv = oportunidades.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "📥 Descargar CSV Oportunidades",
                        data=csv,
                        file_name="Oportunidades_ALSUM_2026.csv",
                        mime="text/csv",
                        key='download-csv'
                    )

    # ==========================================================================
    # TAB 3: EXPLORADOR DE DATOS (AUDITORÍA)
    # ==========================================================================
    with tab3:
        st.subheader("Auditoría de Datos Cargados")
        viz_sel = st.radio("Selecciona Dataset:", ["Nuevos 2025 (Procesado)", "Directorio Maestro", "Plan Estratégico"], horizontal=True)
        
        if viz_sel == "Nuevos 2025 (Procesado)":
            st.write(f"Registros: {len(df_nuevos)}")
            st.dataframe(df_nuevos, use_container_width=True)
        elif viz_sel == "Directorio Maestro":
            st.write(f"Registros: {len(df_dir)}")
            st.dataframe(df_dir, use_container_width=True)
        else:
            st.write(f"Registros: {len(df_plan)}")
            st.dataframe(df_plan, use_container_width=True)

if __name__ == "__main__":
    main()
