import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import io
import utils

# ==============================================================================
# 1. CONFIGURACIÓN ESTRATÉGICA DE LA PÁGINA
# ==============================================================================
st.set_page_config(
    page_title="ALSUM 360 | Command Center",
    layout="wide",
    page_icon="💎",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp { background-color: #F4F6F9; }
    h1, h2, h3 { color: #003366; font-family: 'Segoe UI', sans-serif; }
    div[data-testid="stMetric"] {
        background-color: #ffffff;
        border-left: 5px solid #003366;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border-radius: 8px;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #ffffff;
        border-radius: 5px 5px 0 0;
        padding: 10px 20px;
        color: #003366;
    }
    .stTabs [aria-selected="true"] {
        background-color: #003366;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. FUNCIONES DE CARGA Y ENRIQUECIMIENTO (CORE)
# ==============================================================================

def deduplicate_columns(df):
    df = df.copy()
    cols = pd.Series(df.columns)
    for dup in cols[cols.duplicated()].unique():
        cols[cols[dup].index] = [dup + '.' + str(i) if i != 0 else dup for i in range(sum(cols == dup))]
    df.columns = cols
    return df

def clean_column_names(df):
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
            # Enriquecer con país
            col_pais_dir = next((c for c in df_dir.columns if 'país' in c.lower() or 'pais' in c.lower()), None)
            col_empresa_dir = next((c for c in df_dir.columns if 'empresa' in c.lower() or 'compañía' in c.lower()), None)
            if col_pais_dir and col_empresa_dir and 'Empresa' in df_nuevos.columns:
                df_nuevos['Empresa'] = df_nuevos['Empresa'].astype(str).str.strip()
                df_dir[col_empresa_dir] = df_dir[col_empresa_dir].astype(str).str.strip()
                mapa_pais = dict(zip(df_dir[col_empresa_dir], df_dir[col_pais_dir]))
                df_nuevos['País_Detectado'] = df_nuevos['Empresa'].map(mapa_pais).fillna('Sin Asignar')
            else:
                df_nuevos['País_Detectado'] = 'No Data'
        else:
            return None, None, None, "No se encontró pestaña 'Nuevos'."
    except Exception as e:
        return None, None, None, f"Error cargando Nuevos: {str(e)}"
    # 3. Plan 2026
    df_plan, err = utils.load_plan_accion_procesado(plan_path)
    if err: return None, None, None, f"Error Plan: {err}"
    return df_dir, df_nuevos, df_plan, None

def generate_excel_download(df_nuevos, df_radar):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df_nuevos.to_excel(writer, sheet_name='Nuevos_Detallado', index=False)
        df_radar.to_excel(writer, sheet_name='Oportunidades_Radar', index=False)
        if 'País_Detectado' in df_nuevos.columns:
            resumen_pais = df_nuevos.groupby('País_Detectado').size().reset_index(name='Conteo')
            resumen_pais.to_excel(writer, sheet_name='Resumen_Pais', index=False)
    return output.getvalue()

# ==============================================================================
# 3. INTERFAZ DE USUARIO (MAIN)
# ==============================================================================

def main():
    col_logo, col_title, col_download = st.columns([1, 4, 2])
    with col_title:
        st.title("ALSUM Analytics | Estrategia 360º")
        st.markdown("**Inteligencia de Mercado, Nuevas Incorporaciones y Radar Comercial**")
    with st.spinner("🔄 Procesando matrices de datos..."):
        df_dir, df_nuevos, df_plan, error_msg = load_data_master()
    if error_msg:
        st.error(error_msg)
        st.stop()
    with col_download:
        st.write("")
        st.write("")
        radar_placeholder = pd.DataFrame({'Info': ['Ejecute el cruce en la pestaña Radar']})
        excel_data = generate_excel_download(df_nuevos, radar_placeholder)
        st.download_button(
            label="📥 Descargar Reporte Maestro (.xlsx)",
            data=excel_data,
            file_name="ALSUM_Reporte_Estrategico_Completo.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            type="primary"
        )

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🌍 Dashboard 360º (Nuevos)",
        "🎯 Radar de Oportunidades (IA)",
        "📂 Data Warehouse",
        "📊 Comparativo País & KPIs",
        "📋 Directorio & Desglose"
    ])

    # ==============================
    # TAB 1: DASHBOARD NUEVOS
    # ==============================
    with tab1:
        with st.expander("🔎 Panel de Control y Filtros", expanded=True):
            col_f1, col_f2, col_f3 = st.columns(3)
            paises_disp = sorted(df_nuevos['País_Detectado'].astype(str).unique()) if 'País_Detectado' in df_nuevos.columns else []
            sel_paises = col_f1.multiselect("Filtrar por País", paises_disp, default=paises_disp)
            cats_disp = sorted(df_nuevos['Categoria'].astype(str).unique()) if 'Categoria' in df_nuevos.columns else []
            sel_cat = col_f2.multiselect("Filtrar por Categoría", cats_disp, default=cats_disp)
            col_tipo = None
            for c in ['Tipo_de_Afiliado', 'Categoria']:
                if c in df_nuevos.columns:
                    col_tipo = c
                    break
            sel_tipo = []
            if col_tipo:
                tipos_disp = sorted(df_nuevos[col_tipo].astype(str).unique())
                label_tipo = "Filtrar por Tipo de Afiliado" if col_tipo == 'Tipo_de_Afiliado' else f"Filtrar por {col_tipo}"
                sel_tipo = col_f3.multiselect(label_tipo, tipos_disp, default=tipos_disp, key="multiselect_tipo_dinamico")
        # Aplicar filtros
        df_view = df_nuevos.copy()
        if sel_paises and 'País_Detectado' in df_view.columns:
            df_view = df_view[df_view['País_Detectado'].isin(sel_paises)]
        if sel_cat and 'Categoria' in df_view.columns:
            df_view = df_view[df_view['Categoria'].isin(sel_cat)]
        if sel_tipo and col_tipo:
            df_view = df_view[df_view[col_tipo].isin(sel_tipo)]

        # --- ENRIQUECER df_view CON PRIMAS Y SINIESTROS DEL PLAN ---
        # Pivotear plan para obtener Primas y Siniestros por Compañía
        if {'Compañía', 'Tipo', 'USD'}.issubset(df_plan.columns):
            plan_pivot = df_plan.pivot_table(
                index=['Compañía', 'País'],
                columns='Tipo',
                values='USD',
                aggfunc='sum',
                fill_value=0
            ).reset_index()
            # Merge con df_view
            if 'Empresa' in df_view.columns:
                df_view = pd.merge(
                    df_view,
                    plan_pivot[['Compañía', 'Primas', 'Siniestros']],
                    left_on='Empresa',
                    right_on='Compañía',
                    how='left'
                )
        # --- KPIs SUPERIORES ---
        st.markdown("### 📊 Indicadores de Alto Nivel")
        k1, k2, k3, k4, k5 = st.columns(5)
        total_nuevos = len(df_view)
        col_pais_kpi = 'País_Detectado' if 'País_Detectado' in df_view.columns else next((c for c in df_view.columns if 'pais' in c.lower()), None)
        paises_activos = df_view[col_pais_kpi].nunique() if col_pais_kpi else 0
        if col_tipo:
            miembros = len(df_view[df_view[col_tipo].astype(str).str.contains('Miembro', case=False, na=False)])
            asociados = len(df_view[df_view[col_tipo].astype(str).str.contains('Asociado', case=False, na=False)])
        else:
            miembros = asociados = 0
            st.warning("⚠️ No se encontró columna de 'Tipo_de_Afiliado' ni 'Categoria'. Revisa el Excel.")
        has_finance = 'Primas' in df_view.columns
        total_primas = df_view['Primas'].sum() if has_finance else 0
        k1.metric("Nuevas Empresas", total_nuevos, delta="2025")
        k2.metric("Países Cubiertos", paises_activos)
        k3.metric("Miembros (Core)", miembros)
        k4.metric("Asociados", asociados)
        if has_finance:
            k5.metric("Primas Totales (Est.)", f"${total_primas:,.0f}")
        else:
            k5.metric("Data Financiera", "No Disponible", help="Faltan columnas 'Primas' en el Excel")
        st.markdown("---")
        # --- KPIs de Nuevos por País ---
        if 'País_Detectado' in df_view.columns:
            nuevos_por_pais = df_view['País_Detectado'].value_counts().reset_index()
            nuevos_por_pais.columns = ['País', 'Nuevos Afiliados']
        else:
            nuevos_por_pais = pd.DataFrame()
        # --- KPIs de No Afiliados por País (Plan menos Nuevos) ---
        if 'AFILIADO' in df_plan.columns and 'País' in df_plan.columns:
            no_afiliados = df_plan[df_plan['AFILIADO'].astype(str).str.upper() == 'NO AFILIADO']
            no_afiliados_por_pais = no_afiliados['País'].value_counts().reset_index()
            no_afiliados_por_pais.columns = ['País', 'No Afiliados']
        else:
            no_afiliados = pd.DataFrame()
            no_afiliados_por_pais = pd.DataFrame()
        st.markdown("### 📈 Análisis Comparativo")
        row2_1, row2_2 = st.columns(2)
        with row2_1:
            st.markdown("**Composición por País y Tipo**")
            if 'País_Detectado' in df_view.columns and col_tipo:
                df_stack = df_view.groupby(['País_Detectado', col_tipo]).size().reset_index(name='Conteo')
                fig_stack = px.bar(
                    df_stack,
                    x='País_Detectado',
                    y='Conteo',
                    color=col_tipo,
                    title=f"Distribución por {col_tipo}",
                    barmode='stack'
                )
                st.plotly_chart(fig_stack, use_container_width=True)
            else:
                st.info("Datos insuficientes para el gráfico de barras.")
        with row2_2:
            if 'Primas' in df_view.columns and 'Siniestros' in df_view.columns:
                st.markdown("**Correlación: Primas vs Siniestros**")
                fig_scat = px.scatter(
                    df_view,
                    x='Primas',
                    y='Siniestros',
                    size='Primas',
                    color='País_Detectado' if 'País_Detectado' in df_view.columns else None,
                    hover_name='Empresa',
                    log_x=True, log_y=True,
                    title="Dispersión de Cartera (Escala Log)"
                )
                st.plotly_chart(fig_scat, use_container_width=True)
            else:
                st.info("⚠️ Agrega columnas 'Primas' y 'Siniestros' al Excel para ver el Scatter Plot de Rentabilidad.")
        with st.expander("📋 Ver Tabla Maestra de Datos Filtrados"):
            st.dataframe(
                df_view.style.background_gradient(subset=['Primas'], cmap='Greens') if has_finance else df_view,
                use_container_width=True
            )

    # ==============================
    # TAB 2: RADAR DE OPORTUNIDADES (FUZZY MATCH)
    # ==============================
    with tab2:
        st.subheader("🎯 Identificación de Gaps Comerciales")
        col_conf1, col_conf2, col_act = st.columns([2, 2, 1])
        idx_plan = next((i for i, c in enumerate(df_plan.columns) if c.lower() in ['nombre', 'empresa', 'compañía']), 0)
        idx_dir = next((i for i, c in enumerate(df_dir.columns) if c.lower() in ['nombre', 'empresa', 'compañía']), 0)
        with col_conf1:
            c_plan = st.selectbox("Columna Plan", df_plan.columns, index=idx_plan)
        with col_conf2:
            c_dir = st.selectbox("Columna Directorio", df_dir.columns, index=idx_dir)
        with col_act:
            st.write("")
            run_match = st.button("🚀 Ejecutar Cruce", type="primary")
        if run_match:
            p_safe = df_plan.add_suffix('_Plan')
            d_safe = df_dir.add_suffix('_Dir')
            res = utils.fuzzy_merge(
                p_safe, d_safe,
                f"{c_plan}_Plan", f"{c_dir}_Dir",
                threshold=88
            )
            oportunidades = res[res['match_name'].isna()]
            st.success(f"Se encontraron {len(oportunidades)} empresas objetivo que NO están en el directorio.")
            st.dataframe(oportunidades, use_container_width=True)
            st.session_state['last_radar'] = oportunidades

    # ==============================
    # TAB 3: DATA EXPLORER
    # ==============================
    with tab3:
        st.subheader("Auditoría de Datos")
        v = st.radio("Ver:", ["Nuevos Procesados", "Directorio", "Plan"], horizontal=True)
        if v == "Nuevos Procesados": st.dataframe(df_nuevos)
        elif v == "Directorio": st.dataframe(df_dir)
        else: st.dataframe(df_plan)

    # ==============================
    # TAB 4: Comparativo País & KPIs
    # ==============================
    with tab4:
        st.header("📊 Comparativo País: Nuevos, Plan y No Afiliados")
        c1, c2, c3 = st.columns(3)
        # KPIs Financieros Globales del Plan
        total_primas_plan = df_plan['Primas'].sum() if 'Primas' in df_plan.columns else 0
        total_siniestros_plan = df_plan['Siniestros'].sum() if 'Siniestros' in df_plan.columns else 0
        with c1:
            st.metric("Primas Totales Plan", f"${total_primas_plan:,.0f}")
            st.metric("Siniestros Totales Plan", f"${total_siniestros_plan:,.0f}")
        with c2:
            st.metric("Total Nuevos Afiliados", len(df_view))
            if not nuevos_por_pais.empty:
                st.dataframe(nuevos_por_pais, use_container_width=True)
        with c3:
            st.metric("Total No Afiliados Plan", len(no_afiliados))
            if not no_afiliados_por_pais.empty:
                st.dataframe(no_afiliados_por_pais, use_container_width=True)
        st.markdown("### 🔎 Tabla Detallada de No Afiliados")
        if 'País' in no_afiliados.columns:
            paises_noaf = sorted(no_afiliados['País'].dropna().unique())
            filtro_pais_noaf = st.multiselect("Filtrar No Afiliados por País", paises_noaf, default=paises_noaf, key="filtro_pais_noaf")
            no_afiliados_filt = no_afiliados[no_afiliados['País'].isin(filtro_pais_noaf)]
        else:
            no_afiliados_filt = no_afiliados
        if not no_afiliados_filt.empty:
            st.dataframe(no_afiliados_filt, use_container_width=True)
        else:
            st.info("No hay empresas no afiliadas detectadas en el plan.")

    # ==============================
    # TAB 5: DIRECTORIO & DESGLOSE
    # ==============================
    with tab5:
        st.header("📋 Directorio de Afiliados: Filtros y Desglose")

        # Detectar nombres de columnas
        col_pais_dir = next((c for c in df_dir.columns if 'pais' in c.lower() and 'operacion' in c.lower()), None)
        col_cat_alsum = next((c for c in df_dir.columns if 'categoria' in c.lower() and 'alsum' in c.lower()), None)
        col_cat = next((c for c in df_dir.columns if c.lower() == 'categoria'), None)

        # Filtros globales
        paises_dir = sorted(df_dir[col_pais_dir].dropna().unique()) if col_pais_dir else []
        cats_alsum = sorted(df_dir[col_cat_alsum].dropna().unique()) if col_cat_alsum else []
        cats = sorted(df_dir[col_cat].dropna().unique()) if col_cat else []

        filtro_pais_dir = st.multiselect("Filtrar por País de Operación", paises_dir, default=paises_dir, key="filtro_pais_dir")
        filtro_cat_alsum = st.multiselect("Filtrar por Categoría ALSUM", cats_alsum, default=cats_alsum, key="filtro_cat_alsum")
        filtro_cat = st.multiselect("Filtrar por Tipo (Miembro/Asociado)", cats, default=cats, key="filtro_cat")

        # Aplicar filtros
        df_dir_filt = df_dir.copy()
        if filtro_pais_dir and col_pais_dir:
            df_dir_filt = df_dir_filt[df_dir_filt[col_pais_dir].isin(filtro_pais_dir)]
        if filtro_cat_alsum and col_cat_alsum:
            df_dir_filt = df_dir_filt[df_dir_filt[col_cat_alsum].isin(filtro_cat_alsum)]
        if filtro_cat and col_cat:
            df_dir_filt = df_dir_filt[df_dir_filt[col_cat].isin(filtro_cat)]

        st.markdown("### 📑 Directorio Filtrado")
        st.dataframe(df_dir_filt, use_container_width=True)

        # Desglose por Categoría ALSUM
        if col_cat_alsum:
            st.markdown("#### Desglose por Categoría ALSUM")
            st.dataframe(
                df_dir_filt[col_cat_alsum].value_counts().reset_index().rename(
                    columns={'index': 'Categoría ALSUM', col_cat_alsum: 'Cantidad'}
                ),
                use_container_width=True
            )
        # Desglose por País
        if col_pais_dir:
            st.markdown("#### Desglose por País de Operación")
            st.dataframe(
                df_dir_filt[col_pais_dir].value_counts().reset_index().rename(
                    columns={'index': 'País de Operación', col_pais_dir: 'Cantidad'}
                ),
                use_container_width=True
            )

if __name__ == "__main__":
    main()