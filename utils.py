import streamlit as st
import pandas as pd
import os
import datetime
from fpdf import FPDF
from openai import OpenAI

# ==========================================
# 1. GESTIÓN DE RUTAS Y CONFIGURACIÓN
# ==========================================
def get_file_path(filename):
    """Retorna la ruta absoluta de un archivo en la raíz del proyecto."""
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))
    return os.path.join(base_dir, filename)

def get_api_key():
    """Recupera la API Key de st.secrets o variables de entorno."""
    try:
        return st.secrets["OPENAI_API_KEY"]
    except:
        try:
            return os.environ.get("OPENAI_API_KEY")
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

# ==========================================
# 3. LÓGICA DE DATOS UNIFICADA
# ==========================================

def parse_numero_latino(val):
    """Convierte formatos latinos (1.000,00) o mixtos a float puro."""
    if pd.isna(val): return 0.0
    if isinstance(val, (int, float)): return float(val)
    texto = str(val).strip()
    try: 
        return float(texto)
    except:
        # Intento de limpieza estándar latino: eliminar puntos de mil, cambiar coma por punto
        texto_limpio = texto.replace('.', '').replace(',', '.')
        try: 
            return float(texto_limpio)
        except: 
            return 0.0

@st.cache_data(show_spinner=False)
def load_plan_accion_procesado(filepath, sheet_name=None):
    """
    Carga y procesa el archivo Excel. Usa cache para no recargar en cada interacción.
    Permite especificar la hoja.
    """
    try:
        # Carga optimizada
        df = pd.read_excel(filepath, engine='openpyxl', sheet_name=sheet_name)
        
        # 1. Limpieza de nombres de columnas
        df.columns = [c.strip() for c in df.columns]
        
        # 2. Limpieza básica de strings
        if 'Compañía' in df.columns: 
            df['Compañía'] = df['Compañía'].astype(str).str.strip()
        
        # 3. Rellenos de nulos
        if 'Subramo' in df.columns: df['Subramo'] = df['Subramo'].fillna('General')
        if 'Ramo' in df.columns: df['Ramo'] = df['Ramo'].fillna('Otros')
        
        # 4. Estandarización de AFILIADO
        if 'AFILIADO' in df.columns:
            df['AFILIADO'] = df['AFILIADO'].fillna('NO AFILIADO').astype(str).str.strip().str.upper()
            df['AFILIADO'] = df['AFILIADO'].replace({'NO AFILIADOS':'NO AFILIADO', 'AFILIADOS':'AFILIADO'})

        # 5. Conversión Numérica robusta
        if 'USD' in df.columns:
            df['USD'] = df['USD'].apply(parse_numero_latino)

        # 6. Pivoteo Fundamental
        # Transformamos la columna 'Tipo' (que debe contener "Primas" y "Siniestros" u otros) en columnas reales
        cols_clave = ['País', 'Año', 'Compañía', 'Ramo', 'Subramo', 'AFILIADO']
        
        # Verificamos que existan las columnas mínimas para pivotar
        if all(col in df.columns for col in cols_clave) and 'Tipo' in df.columns and 'USD' in df.columns:
            
            # Pivot table
            pivot_df = df.pivot_table(
                index=cols_clave,
                columns='Tipo',
                values='USD',
                aggfunc='sum',
                fill_value=0
            ).reset_index()
            
            # Limpiar nombre del índice de columnas
            pivot_df.columns.name = None
            
            # Asegurar que existan Primas y Siniestros aunque no vengan en el Excel
            if 'Primas' not in pivot_df.columns: pivot_df['Primas'] = 0.0
            if 'Siniestros' not in pivot_df.columns: pivot_df['Siniestros'] = 0.0
            
            # Cálculos derivados post-pivot
            pivot_df['Siniestros'] = pivot_df['Siniestros'].abs() # Asegurar positivo
            
            # Evitar división por cero
            pivot_df['Siniestralidad'] = pivot_df.apply(
                lambda x: (x['Siniestros'] / x['Primas'] * 100) if x['Primas'] > 0 else 0, axis=1
            )
            
            pivot_df['Resultado Técnico'] = pivot_df['Primas'] - pivot_df['Siniestros']
            
            return pivot_df, None
        else:
            return df, "Error: Faltan columnas clave (Tipo, USD, País, Año...) para realizar el pivoteo."

    except Exception as e:
        return None, f"Error cargando archivo: {str(e)}"

# ==========================================
# 4. FUNCIONES DE ANÁLISIS AUXILIARES
# ==========================================

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

def load_excel_sheet(filepath, sheet_name):
    """Carga un archivo Excel y una hoja específica, limpiando nombres de columnas."""
    df = pd.read_excel(filepath, sheet_name=sheet_name, engine="openpyxl")
    df.columns = [c.strip() for c in df.columns]
    return df