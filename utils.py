import streamlit as st
import pandas as pd
import os
import datetime
from fpdf import FPDF
from openai import OpenAI
from fuzzywuzzy import process

# ==========================================
# 1. GESTIÓN DE RUTAS Y CONFIGURACIÓN
# ==========================================
def get_file_path(filename):
    """Retorna la ruta absoluta de un archivo en el directorio base."""
    # Busca en el directorio actual del script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(current_dir, filename)

def get_api_key():
    """Recupera la API Key de st.secrets o variables de entorno."""
    try:
        return st.secrets["OPENAI_API_KEY"]
    except:
        try:
            return os.environ["OPENAI_API_KEY"]
        except:
            return None

# ==========================================
# 2. CLASE PDF (UltimatePDF)
# ==========================================
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
# 3. LÓGICA DE DATOS UNIFICADA
# ==========================================

def parse_numero_latino(val):
    """Convierte formatos latinos (1.000,00) o mixtos a float puro."""
    if pd.isna(val): return 0.0
    texto = str(val).strip()
    try: 
        return float(texto)
    except:
        # Intento de limpieza estándar latino
        texto_limpio = texto.replace('.', '').replace(',', '.')
        try: 
            return float(texto_limpio)
        except: 
            return 0.0

@st.cache_data(show_spinner=False)
def load_plan_accion_procesado(filepath):
    """
    Carga y procesa el 'Plan de accion 2026.xlsx' con toda la lógica de limpieza
    del dashboard estratégico.
    """
    try:
        # Detectar extensión para usar el engine correcto
        if filepath.endswith('.csv'):
            df = pd.read_csv(filepath, sep=';', engine='python', on_bad_lines='skip', encoding='utf-8', header=0)
        else:
            df = pd.read_excel(filepath, engine='openpyxl', header=0)
        
        # Normalizar columnas
        df.columns = [c.strip() for c in df.columns]
        
        # Limpieza strings
        if 'Compañía' in df.columns: 
            df['Compañía'] = df['Compañía'].astype(str).str.strip()
        
        # Rellenos básicos
        if 'Subramo' in df.columns: df['Subramo'] = df['Subramo'].fillna('General')
        if 'Ramo' in df.columns: df['Ramo'] = df['Ramo'].fillna('Otros')
        
        # Lógica AFILIADO
        if 'AFILIADO' in df.columns:
            df['AFILIADO'] = df['AFILIADO'].fillna('NO AFILIADO').astype(str).str.strip().str.upper()
            df['AFILIADO'] = df['AFILIADO'].replace({'NO AFILIADOS':'NO AFILIADO', 'AFILIADOS':'AFILIADO'})

        # Conversión Numérica
        if 'USD' in df.columns:
            df['USD'] = df['USD'].apply(parse_numero_latino)

        # Filtros de negocio (excluir ramos específicos)
        if 'Ramo' in df.columns:
            df = df[~df['Ramo'].str.upper().isin(['RIESGOS PORTUARIOS', 'RIESGOS PETROLEROS'])]

        # Pivoteo para estructura final
        # Aseguramos que existan las columnas clave antes de pivotar
        cols_necesarias = ['País', 'Año', 'Compañía', 'Ramo', 'Subramo', 'AFILIADO', 'Tipo', 'USD']
        if all(col in df.columns for col in cols_necesarias):
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
            
            # Normalización de nombre para cruces
            pivot_df['nombre_norm'] = pivot_df['Compañía'].astype(str).str.lower().str.strip()
            
            return pivot_df, None
        else:
            return df, "Formato de archivo crudo (columnas faltantes para pivot automático)"

    except Exception as e:
        return None, str(e)

@st.cache_data(show_spinner=False)
def load_simple_excel(filepath):
    """Carga simple para otros archivos Excel."""
    try:
        df = pd.read_excel(filepath)
        # Crear columna normalizada para cruces si existe nombre/compañía
        for col in ['Compañía', 'Nombre', 'Empresa', 'Aseguradora']:
            if col in df.columns:
                df['nombre_norm'] = df[col].astype(str).str.lower().str.strip()
                break
        return df
    except Exception as e:
        return None

# ==========================================
# 4. FUNCIONES DE ANÁLISIS Y AUXILIARES
# ==========================================

def generar_analisis_ia(contexto_datos, tipo_grafico, api_key):
    """Motor de Análisis IA."""
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

def crear_vista_pivot_anos(df_input, indice, valor='Primas'):
    """Crea una tabla con los años como columnas y una columna final de Total."""
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

def fuzzy_merge(df_left, df_right, key_left, key_right, threshold=80, limit=1):
    """
    Realiza un cruce difuso entre dos DataFrames.
    Retorna df_left con una columna 'match_name' y 'score' basado en df_right.
    """
    s = df_right[key_right].tolist()
    
    # Función auxiliar para aplicar a cada fila
    def get_match(x):
        # Process.extractOne devuelve (match, score, index)
        match = process.extractOne(x, s)
        if match and match[1] >= threshold:
            return match[0], match[1]
        else:
            return None, 0
    
    # Aplicar la función
    matches = df_left[key_left].apply(get_match)
    
    # Desempaquetar resultados
    df_left['match_name'] = [m[0] if m else None for m in matches]
    df_left['match_score'] = [m[1] if m else 0 for m in matches]
    
    return df_left