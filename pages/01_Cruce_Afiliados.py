import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import io
import utils  # Se mantiene tu librería original
from openai import OpenAI # Cliente oficial de OpenAI

# ==============================================================================
# 1. CONFIGURACIÓN ESTRATÉGICA Y ESTILOS (MEJORADO)
# ==============================================================================
st.set_page_config(
    page_title="ALSUM 360 | Enterprise Command Center",
    layout="wide",
    page_icon="💎",
    initial_sidebar_state="expanded"
)

# Estilos CSS para dar sensación "Premium" sin romper la funcionalidad
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
# 2. FUNCIONES CORE (TU LÓGICA ORIGINAL + OPTIMIZACIONES)
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
            # Renombrado Estratégico (TU LÓGICA)
            rename_map = {
                'Tipo_de_Compañía': 'Categoria',
                'Compañía': 'Empresa',
                'Tipo_de_Afiliado': 'Tipo_de_Afiliado',
                'Tipo_Afiliado': 'Tipo_de_Afiliado',
            }
            df_nuevos = df_nuevos.rename(columns=rename_map)
            df_nuevos = deduplicate_columns(df_nuevos)
            
            # Enriquecer con país (TU LÓGICA)
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

def generate_excel_download(df_nuevos, df_radar, extra_sheets=None):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df_nuevos.to_excel(writer, sheet_name='Nuevos_Detallado', index=False)
        df_radar.to_excel(writer, sheet_name='Oportunidades_Radar', index=False)
        if 'País_Detectado' in df_nuevos.columns:
            resumen_pais = df_nuevos.groupby('País_Detectado').size().reset_index(name='Conteo')
            resumen_pais.to_excel(writer, sheet_name='Resumen_Pais', index=False)
        # Hojas extra si las enviamos
        if extra_sheets:
            for name, df in extra_sheets.items():
                if not df.empty:
                    df.to_excel(writer, sheet_name=name[:30], index=False)
    return output.getvalue()

# --- FUNCIÓN IA (NUEVA) ---
def consultar_gpt4(prompt, data_context):
    api_key = utils.get_api_key()
    if not api_key:
        return "⚠️ No se detectó la API Key en el servidor."
    
    client = OpenAI(api_key=api_key)
    full_prompt = f"""
    Actúa como un Analista Senior de Estrategia y Seguros. 
    Analiza los siguientes datos resumidos:
    {data_context}
    
    PREGUNTA DEL USUARIO: {prompt}
    
    Responde con insights accionables, puntos clave y recomendaciones estratégicas. Sé directo.
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
# 3. INTERFAZ DE USUARIO (MAIN EXTENDIDO)
# ==============================================================================

def main():
    # --- SIDEBAR GLOBAL ---
    with st.sidebar:
        st.header("⚙️ Configuración & IA")
        st.info("Sistema cargado y listo.")

    col_logo, col_title, col_download = st.columns([1, 4, 2])
    with col_title:
        st.title("ALSUM Analytics | Estrategia 360º Ultimate")
        st.markdown("**Inteligencia de Mercado, Nuevas Incorporaciones y Radar Comercial IA**")
        
    with st.spinner("🔄 Procesando matrices de datos complejas..."):
        df_dir, df_nuevos, df_plan, error_msg = load_data_master()
        
    if error_msg:
        st.error(error_msg)
        st.stop()
        
    # Preparar descarga inicial
    radar_placeholder = pd.DataFrame({'Info': ['Ejecute el cruce en la pestaña Radar']})
    
    # --- TABS (SE MANTIENEN LOS TUYOS Y SE AGREGAN NUEVOS) ---
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "🌍 Dashboard 360º (Original)",
        "📊 Deep Analytics (Nuevo)",
        "🎯 Radar de Oportunidades",
        "📂 Data Warehouse",
        "📈 Comparativo País & KPIs",
        "📋 Directorio & Desglose",
        "🤖 Laboratorio IA"
    ])

    # ==========================================================================
    # TAB 1: DASHBOARD NUEVOS (MANTENIENDO TU LÓGICA + MEJORAS VISUALES)
    # ==========================================================================
    with tab1:
        st.subheader("Panel de Control: Nuevas Incorporaciones")
        
        # --- TU SISTEMA DE FILTROS ORIGINAL ---
        with st.expander("🔎 Filtros de Datos (Tu Configuración)", expanded=True):
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

        # --- APLICACIÓN DE TUS FILTROS ---
        df_view = df_nuevos.copy()
        if sel_paises and 'País_Detectado' in df_view.columns:
            df_view = df_view[df_view['País_Detectado'].isin(sel_paises)]
        if sel_cat and 'Categoria' in df_view.columns:
            df_view = df_view[df_view['Categoria'].isin(sel_cat)]
        if sel_tipo and col_tipo:
            df_view = df_view[df_view[col_tipo].isin(sel_tipo)]

        # --- TU LÓGICA DE CRUCE CON PLAN (PRIMAS/SINIESTROS) ---
        has_finance = False
        if {'Compañía', 'Tipo', 'USD'}.issubset(df_plan.columns):
            plan_pivot = df_plan.pivot_table(
                index=['Compañía', 'País'], columns='Tipo', values='USD', aggfunc='sum', fill_value=0
            ).reset_index()
            
            if 'Empresa' in df_view.columns:
                df_view = pd.merge(
                    df_view,
                    plan_pivot[['Compañía', 'Primas', 'Siniestros']],
                    left_on='Empresa', right_on='Compañía', how='left'
                )
                has_finance = True

        # --- TUS KPIS ORIGINALES (MEJORADOS VISUALMENTE) ---
        st.markdown("### 📊 Indicadores de Alto Nivel")
        k1, k2, k3, k4, k5 = st.columns(5)
        
        # Cálculos originales
        total_nuevos = len(df_view)
        col_pais_kpi = 'País_Detectado' if 'País_Detectado' in df_view.columns else next((c for c in df_view.columns if 'pais' in c.lower()), None)
        paises_activos = df_view[col_pais_kpi].nunique() if col_pais_kpi else 0
        
        miembros = 0
        asociados = 0
        if col_tipo:
            miembros = len(df_view[df_view[col_tipo].astype(str).str.contains('Miembro', case=False, na=False)])
            asociados = len(df_view[df_view[col_tipo].astype(str).str.contains('Asociado', case=False, na=False)])
        
        total_primas = df_view['Primas'].sum() if 'Primas' in df_view.columns else 0
        
        # Renderizado
        k1.metric("Nuevas Empresas", total_nuevos, delta="2025 Activo")
        k2.metric("Países Cubiertos", paises_activos)
        k3.metric("Miembros (Core)", miembros, delta=f"{round((miembros/total_nuevos)*100,1)}%" if total_nuevos>0 else None)
        k4.metric("Asociados", asociados)
        if has_finance:
            k5.metric("Primas Totales (Est.)", f"${total_primas:,.0f}", delta="Cruzado con Plan")
        else:
            k5.metric("Data Financiera", "No Disponible")

        st.markdown("---")

        # --- GRÁFICOS ORIGINALES + NUEVOS VISUALES AVANZADOS ---
        row2_1, row2_2 = st.columns(2)
        
        with row2_1:
            st.markdown("**Composición por País y Tipo (Tu Gráfico Original)**")
            if 'País_Detectado' in df_view.columns and col_tipo:
                df_stack = df_view.groupby(['País_Detectado', col_tipo]).size().reset_index(name='Conteo')
                fig_stack = px.bar(
                    df_stack, x='País_Detectado', y='Conteo', color=col_tipo,
                    title=f"Distribución por {col_tipo}", barmode='stack', template="plotly_white"
                )
                st.plotly_chart(fig_stack, use_container_width=True)
        
        with row2_2:
            # NUEVO: Sunburst Chart (Visualización Jerárquica)
            st.markdown("**Jerarquía: País -> Categoría -> Empresa (NUEVO)**")
            if 'País_Detectado' in df_view.columns and 'Categoria' in df_view.columns:
                fig_sun = px.sunburst(
                    df_view, path=['País_Detectado', 'Categoria'], 
                    title="Radiografía del Mercado",
                    color='País_Detectado'
                )
                st.plotly_chart(fig_sun, use_container_width=True)

        # --- SCATTER PLOT ORIGINAL (Si hay finanzas) ---
        if 'Primas' in df_view.columns and 'Siniestros' in df_view.columns:
            st.markdown("#### 💰 Análisis de Rentabilidad (Dispersión)")
            fig_scat = px.scatter(
                df_view, x='Primas', y='Siniestros', size='Primas',
                color='País_Detectado' if 'País_Detectado' in df_view.columns else None,
                hover_name='Empresa', log_x=True, log_y=True,
                title="Primas vs Siniestros (Escala Logarítmica)",
                marginal_x="box", marginal_y="violin" # MEJORA: Añadidos márgenes estadísticos
            )
            st.plotly_chart(fig_scat, use_container_width=True)

        with st.expander("📋 Ver Tabla Maestra de Datos Filtrados"):
            st.dataframe(df_view.style.background_gradient(subset=['Primas'], cmap='Greens') if has_finance else df_view, use_container_width=True)

    # ==========================================================================
    # TAB 2: DEEP ANALYTICS (NUEVA PESTAÑA POTENCIADA)
    # ==========================================================================
    with tab2:
        st.header("🔍 Deep Analytics & Market Intelligence")
        st.info("Aquí aplicamos análisis estadísticos avanzados sobre los datos cargados.")
        
        da1, da2 = st.columns(2)
        
        with da1:
            st.markdown("#### Ley de Pareto (80/20) - Concentración de Primas")
            if 'Primas' in df_plan.columns and 'País' in df_plan.columns:
                pareto_df = df_plan.groupby('País')['Primas'].sum().sort_values(ascending=False).reset_index()
                pareto_df['Acumulado'] = pareto_df['Primas'].cumsum()
                pareto_df['Porcentaje'] = 100 * pareto_df['Acumulado'] / pareto_df['Primas'].sum()
                
                fig_pareto = go.Figure()
                fig_pareto.add_trace(go.Bar(x=pareto_df['País'], y=pareto_df['Primas'], name='Primas'))
                fig_pareto.add_trace(go.Scatter(x=pareto_df['País'], y=pareto_df['Porcentaje'], name='% Acumulado', yaxis='y2', mode='lines+markers', line=dict(color='red')))
                fig_pareto.update_layout(yaxis2=dict(overlaying='y', side='right', range=[0, 110]), title="Pareto por País (Plan 2026)")
                st.plotly_chart(fig_pareto, use_container_width=True)
            else:
                st.warning("Se requiere columna 'Primas' en el Plan para Pareto.")
                
        with da2:
            st.markdown("#### Mapa de Calor (Heatmap) de Actividad")
            # Heatmap entre País y Tipo de Afiliado (Directorio)
            col_pais_h = next((c for c in df_dir.columns if 'pais' in c.lower()), None)
            col_cat_h = next((c for c in df_dir.columns if 'categoria' in c.lower()), None)
            
            if col_pais_h and col_cat_h:
                heat_data = df_dir.groupby([col_pais_h, col_cat_h]).size().reset_index(name='Cantidad')
                fig_heat = px.density_heatmap(
                    heat_data, x=col_pais_h, y=col_cat_h, z='Cantidad',
                    title="Densidad de Afiliados: País vs Categoría",
                    color_continuous_scale="Viridis"
                )
                st.plotly_chart(fig_heat, use_container_width=True)
            else:
                st.warning("Faltan columnas en Directorio para el Heatmap.")

    # ==========================================================================
    # TAB 3: RADAR (MANTENIENDO TU LÓGICA EXACTA)
    # ==========================================================================
    with tab3:
        st.subheader("🎯 Radar de Oportunidades (Tu Lógica Fuzzy)")
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
            st.session_state['oportunidades_radar'] = oportunidades # Guardar en sesión
            st.success(f"Se encontraron {len(oportunidades)} empresas objetivo que NO están en el directorio.")
            
            # MEJORA: Botón de análisis IA directo sobre el resultado
            if not oportunidades.empty and api_key:
                if st.button("🤖 Analizar Oportunidades con IA"):
                    resumen_txt = oportunidades.head(20).to_string()
                    analisis = consultar_gpt4(api_key, "Analiza estas empresas oportunidad y suguiere 3 acciones de abordaje.", resumen_txt)
                    st.info(analisis)
            
            st.dataframe(oportunidades, use_container_width=True)

    # ==========================================================================
    # TAB 4: DATA WAREHOUSE (ORIGINAL)
    # ==========================================================================
    with tab4:
        st.subheader("Auditoría de Datos")
        v = st.radio("Ver Dataset:", ["Nuevos Procesados", "Directorio", "Plan"], horizontal=True)
        if v == "Nuevos Procesados": st.dataframe(df_nuevos, use_container_width=True)
        elif v == "Directorio": st.dataframe(df_dir, use_container_width=True)
        else: st.dataframe(df_plan, use_container_width=True)

    # ==========================================================================
    # TAB 5: COMPARATIVO Y NO AFILIADOS (TU LÓGICA MANTENIDA)
    # ==========================================================================
    with tab5:
        st.header("📊 Comparativo País & KPIs")
        c1, c2, c3 = st.columns(3)
        
        # Recalcular KPIs por si acaso
        nuevos_por_pais = pd.DataFrame()
        if 'País_Detectado' in df_view.columns: # Usamos df_view para consistencia
            nuevos_por_pais = df_view['País_Detectado'].value_counts().reset_index()
            nuevos_por_pais.columns = ['País', 'Nuevos Afiliados']

        no_afiliados = pd.DataFrame()
        no_afiliados_por_pais = pd.DataFrame()
        if 'AFILIADO' in df_plan.columns and 'País' in df_plan.columns:
            no_afiliados = df_plan[df_plan['AFILIADO'].astype(str).str.upper() == 'NO AFILIADO']
            no_afiliados_por_pais = no_afiliados['País'].value_counts().reset_index()
            no_afiliados_por_pais.columns = ['País', 'No Afiliados']

        # Mostrar Métricas (Tu lógica)
        total_primas_plan = df_plan['Primas'].sum() if 'Primas' in df_plan.columns else 0
        with c1:
            st.metric("Primas Totales Plan", f"${total_primas_plan:,.0f}")
        with c2:
            st.metric("Total Nuevos (Filtrado)", len(df_view))
            if not nuevos_por_pais.empty: st.dataframe(nuevos_por_pais, use_container_width=True, height=200)
        with c3:
            st.metric("Total No Afiliados (Plan)", len(no_afiliados))
            if not no_afiliados_por_pais.empty: st.dataframe(no_afiliados_por_pais, use_container_width=True, height=200)

        # MEJORA: Mapa Choropleth de No Afiliados (Visualización Geo)
        st.markdown("### 🌍 Mapa de Oportunidad (No Afiliados)")
        if not no_afiliados_por_pais.empty:
            fig_map = px.choropleth(
                no_afiliados_por_pais, locations='País', locationmode='country names',
                color='No Afiliados', color_continuous_scale="Reds",
                title="Concentración de No Afiliados por País"
            )
            st.plotly_chart(fig_map, use_container_width=True)

        st.markdown("### 🔎 Tabla Detallada de No Afiliados")
        if 'País' in no_afiliados.columns:
            paises_noaf = sorted(no_afiliados['País'].dropna().unique())
            filtro_pais_noaf = st.multiselect("Filtrar No Afiliados por País", paises_noaf, default=paises_noaf, key="filtro_pais_noaf")
            no_afiliados_filt = no_afiliados[no_afiliados['País'].isin(filtro_pais_noaf)]
            st.dataframe(no_afiliados_filt, use_container_width=True)
        else:
            st.dataframe(no_afiliados, use_container_width=True)

    # ==========================================================================
    # TAB 6: DIRECTORIO (TU LÓGICA MANTENIDA)
    # ==========================================================================
    with tab6:
        st.header("📋 Directorio de Afiliados: Filtros y Desglose")
        
        # Detectar columnas (Igual que tu código)
        col_pais_dir = next((c for c in df_dir.columns if 'pais' in c.lower() and 'operacion' in c.lower()), None)
        col_cat_alsum = next((c for c in df_dir.columns if 'categoria' in c.lower() and 'alsum' in c.lower()), None)
        col_cat = next((c for c in df_dir.columns if c.lower() == 'categoria'), None)

        paises_dir = sorted(df_dir[col_pais_dir].dropna().unique()) if col_pais_dir else []
        cats_alsum = sorted(df_dir[col_cat_alsum].dropna().unique()) if col_cat_alsum else []
        cats = sorted(df_dir[col_cat].dropna().unique()) if col_cat else []

        c_d1, c_d2, c_d3 = st.columns(3)
        filtro_pais_dir = c_d1.multiselect("País Operación", paises_dir, default=paises_dir, key="filtro_pais_dir")
        filtro_cat_alsum = c_d2.multiselect("Categoría ALSUM", cats_alsum, default=cats_alsum, key="filtro_cat_alsum")
        filtro_cat = c_d3.multiselect("Tipo (Miembro/Asociado)", cats, default=cats, key="filtro_cat")

        # Filtrado
        df_dir_filt = df_dir.copy()
        if filtro_pais_dir and col_pais_dir: df_dir_filt = df_dir_filt[df_dir_filt[col_pais_dir].isin(filtro_pais_dir)]
        if filtro_cat_alsum and col_cat_alsum: df_dir_filt = df_dir_filt[df_dir_filt[col_cat_alsum].isin(filtro_cat_alsum)]
        if filtro_cat and col_cat: df_dir_filt = df_dir_filt[df_dir_filt[col_cat].isin(filtro_cat)]

        st.dataframe(df_dir_filt, use_container_width=True)
        
        # Desgloses (Tu lógica + Gráfico Pie Extra)
        col_pie1, col_pie2 = st.columns(2)
        if col_cat_alsum:
            with col_pie1:
                st.markdown("#### Por Categoría ALSUM")
                count_cat = df_dir_filt[col_cat_alsum].value_counts().reset_index()
                count_cat.columns = ['Categoría', 'Cantidad']
                fig_pie1 = px.pie(count_cat, values='Cantidad', names='Categoría', hole=0.4)
                st.plotly_chart(fig_pie1, use_container_width=True)

        if col_pais_dir:
            with col_pie2:
                st.markdown("#### Por País de Operación")
                count_pais = df_dir_filt[col_pais_dir].value_counts().reset_index()
                count_pais.columns = ['País', 'Cantidad']
                fig_bar_pais = px.bar(count_pais, x='Cantidad', y='País', orientation='h')
                st.plotly_chart(fig_bar_pais, use_container_width=True)

    # ==========================================================================
    # TAB 7: LABORATORIO IA (TOTALMENTE NUEVO)
    # ==========================================================================
    with tab7:
        st.header("🤖 Laboratorio de Inteligencia Artificial (GPT-4o-mini)")
        st.markdown("Utiliza la API de OpenAI para analizar los datos cargados en el sistema.")
        
        c_ia1, c_ia2 = st.columns([1, 2])
        with c_ia1:
            st.info("Selecciona qué datos quieres analizar:")
            dataset_opt = st.selectbox("Fuente de Datos", ["Nuevos Afiliados (Resumen)", "Radar de Oportunidades", "No Afiliados (Top 50)"])
            user_prompt = st.text_area("¿Qué quieres saber?", "Dame un análisis FODA de estos datos y suguiere estrategias de crecimiento.")
            btn_ia = st.button("✨ Generar Análisis IA")
        
        with c_ia2:
            if btn_ia:
                if not api_key:
                    st.error("❌ Falta la API Key en el menú lateral.")
                else:
                    # Preparar contexto
                    contexto = ""
                    if dataset_opt == "Nuevos Afiliados (Resumen)":
                        contexto = df_nuevos.describe(include='all').to_string() + "\n\n" + df_nuevos.head(20).to_string()
                    elif dataset_opt == "Radar de Oportunidades":
                        if 'oportunidades_radar' in st.session_state:
                            contexto = st.session_state['oportunidades_radar'].head(30).to_string()
                        else:
                            contexto = "El usuario aún no ha ejecutado el Radar."
                    elif dataset_opt == "No Afiliados (Top 50)":
                        contexto = no_afiliados.head(50).to_string()
                    
                    with st.spinner("Consultando a GPT-4o-mini..."):
                        resultado = consultar_gpt4(api_key, user_prompt, contexto)
                        st.markdown("### 💡 Insights Generados")
                        st.markdown(resultado)
                        st.download_button("📥 Descargar Reporte IA", resultado)

    # BOTÓN DE DESCARGA GLOBAL REUBICADO PARA ESTAR SIEMPRE VISIBLE AL FINAL
    with col_download:
        radar_df = st.session_state.get('oportunidades_radar', pd.DataFrame({'Info': ['Ejecute Radar primero']}))
        excel_data = generate_excel_download(df_nuevos, radar_df, extra_sheets={'No_Afiliados': no_afiliados})
        st.download_button(
            label="📥 Descargar Reporte Maestro Completo (.xlsx)",
            data=excel_data,
            file_name="ALSUM_Reporte_Estrategico_Ultimate.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            type="primary"
        )

if __name__ == "__main__":
    main()