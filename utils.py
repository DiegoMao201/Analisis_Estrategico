import streamlit as st
import pandas as pd
import os
import datetime
import re
import gc
from fpdf import FPDF
from fuzzywuzzy import process, fuzz
import openai

# ==========================================
# 1. GESTIÓN DE SISTEMA Y RUTAS
# ==========================================
def get_file_path(filename):
    """Busca el archivo en la raíz o sube un nivel si está en pages."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    if base_dir.endswith("pages"):
        base_dir = os.path.dirname(base_dir)
    return os.path.join(base_dir, filename)

def get_api_key():
    try:
        return st.secrets["OPENAI_API_KEY"]
    except:
        return os.environ.get("OPENAI_API_KEY")

# ==========================================
# 2. MOTOR PDF
# ==========================================
class UltimatePDF(FPDF):
    def header(self):
        if self.page_no() > 1:
            self.set_font('Arial', 'B', 9)
            self.set_text_color(128, 128, 128)
            self.cell(0, 10, 'ALSUM - ESTRATEGIA 2026', 0, 0, 'L')
            self.cell(0, 10, f'{datetime.date.today().strftime("%d/%m/%Y")}', 0, 1, 'R')
            self.set_draw_color(0, 74, 143)
            self.set_line_width(0.5)
            self.line(10, 20, 200, 20)
            self.ln(15)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f'Página {self.page_no()}', 0, 0, 'C')

    def cover_page(self, title, subtitle):
        self.add_page()
        self.set_fill_color(0, 74, 143) 
        self.rect(0, 0, 210, 297, 'F') 
        self.set_text_color(255, 255, 255)
        self.set_font('Arial', 'B', 40)
        self.ln(60)
        self.cell(0, 20, "ALSUM", 0, 1, 'C')
        self.set_font('Arial', '', 14)
        self.cell(0, 10, "INTELIGENCIA DE NEGOCIOS", 0, 1, 'C')
        self.ln(40)
        self.set_font('Arial', 'B', 28)
        self.multi_cell(0, 15, title, 0, 'C')
        self.ln(10)
        self.set_font('Arial', 'I', 16)
        self.multi_cell(0, 10, subtitle, 0, 'C')

    def chapter_body(self, text):
        self.set_font('Arial', '', 11)
        self.set_text_color(40, 40, 40)
        self.multi_cell(0, 6, text)
        self.ln()

# ==========================================
# 3. CARGA DE DATOS (MÉTODO "SNIFFER" + LOW_MEMORY)
# ==========================================
def parse_numero_latino(val):
    if pd.isna(val) or val == '': return 0.0
    if isinstance(val, (int, float)): return float(val)
    texto = str(val).strip()
    texto = re.sub(r'[^\d.,-]', '', texto)
    try: return float(texto)
    except:
        try: return float(texto.replace('.', '').replace(',', '.'))
        except: return 0.0

@st.cache_data(ttl=3600, show_spinner=False)
def load_plan_accion_procesado(filepath, sheet_name=None):
    print(f"--- [UTILS] Iniciando carga de: {filepath}")
    
    if not os.path.exists(filepath):
        return None, f"Archivo no encontrado: {filepath}"

    try:
        gc.collect() 
        df = None

        # --- ESTRATEGIA "SNIFFER" (OLFATEAR EL ARCHIVO) ---
        if filepath.lower().endswith('.csv'):
            print("--- [UTILS] Detectando formato CSV...")
            # Leemos solo un pedacito (5KB) para detectar el separador
            sep = ','
            enc = 'utf-8'
            
            try:
                with open(filepath, 'rb') as f:
                    sample = f.read(10000) # Leemos más bytes para asegurar
                
                # Detectar Encoding
                for e in ['utf-8', 'latin-1', 'cp1252']:
                    try:
                        text_sample = sample.decode(e)
                        enc = e
                        break
                    except:
                        continue
                
                # Detectar Separador (Contamos cuál aparece más)
                if text_sample.count(';') > text_sample.count(','):
                    sep = ';'
                
                print(f"--- [UTILS] Formato detectado: Separador='{sep}' | Encoding='{enc}'")
                
                # Carga ÚNICA y DEFINITIVA con low_memory=False para evitar warnings
                df = pd.read_csv(filepath, sep=sep, encoding=enc, on_bad_lines='skip', low_memory=False)
                
            except Exception as e:
                print(f"--- [ERROR DETECCIÓN] {e}. Intentando fallback...")
                # Intento a fuerza bruta si falla el sniffer
                df = pd.read_csv(filepath, sep=';', encoding='latin-1', on_bad_lines='skip', low_memory=False)

        else:
            print("--- [UTILS] Leyendo Excel...")
            df = pd.read_excel(filepath, engine='openpyxl', sheet_name=sheet_name)

        # --- LIMPIEZA RÁPIDA ---
        df.columns = [str(c).strip() for c in df.columns]
        
        # Corrección Año
        if 'Año' not in df.columns:
            for col in df.columns:
                if 'A' in col and 'o' in col and len(col) <= 5:
                    df.rename(columns={col: 'Año'}, inplace=True)
                    break

        # Limpieza Textos
        if 'Compañía' in df.columns: df['Compañía'] = df['Compañía'].astype(str).str.strip()
        
        # Rellenos
        defaults = {'Subramo': 'General', 'Ramo': 'Otros', 'País': 'Desconocido', 'AFILIADO': 'NO AFILIADO'}
        df.fillna(defaults, inplace=True)
        
        if 'AFILIADO' in df.columns:
            df['AFILIADO'] = df['AFILIADO'].astype(str).str.upper()
            mask_no = df['AFILIADO'].str.contains('NO')
            df.loc[mask_no, 'AFILIADO'] = 'NO AFILIADO'
            df.loc[~mask_no, 'AFILIADO'] = 'AFILIADO'

        if 'USD' in df.columns: df['USD'] = df['USD'].apply(parse_numero_latino)

        # --- PIVOTEO SI ES NECESARIO ---
        if 'Primas' in df.columns and 'Siniestros' in df.columns:
            pivot_df = df
        else:
            cols_req = ['País', 'Año', 'Compañía', 'Ramo', 'Subramo', 'AFILIADO']
            if all(c in df.columns for c in cols_req) and 'Tipo' in df.columns and 'USD' in df.columns:
                print("--- [UTILS] Pivotando...")
                pivot_df = df.pivot_table(index=cols_req, columns='Tipo', values='USD', aggfunc='sum', fill_value=0).reset_index()
                pivot_df.columns.name = None
            else:
                pivot_df = df 

        # Métricas Finales
        for m in ['Primas', 'Siniestros']:
            if m not in pivot_df.columns: pivot_df[m] = 0.0
        
        pivot_df['Resultado Técnico'] = pivot_df['Primas'] - pivot_df['Siniestros']
        
        pivot_df['Siniestralidad'] = 0.0
        mask_pos = pivot_df['Primas'] > 0
        pivot_df.loc[mask_pos, 'Siniestralidad'] = (pivot_df.loc[mask_pos, 'Siniestros'] / pivot_df.loc[mask_pos, 'Primas']) * 100
        
        print(f"--- [UTILS] Carga completa. {len(pivot_df)} filas.")
        return pivot_df, None
            
    except Exception as e:
        print(f"--- [ERROR CRÍTICO] {e}")
        return None, f"Error: {str(e)}"

def load_excel_sheet(filepath, sheet_name=None):
    try:
        if not os.path.exists(filepath): return None
        if filepath.lower().endswith('.csv'):
             try: return pd.read_csv(filepath, sep=';', encoding='latin-1')
             except: return pd.read_csv(filepath, sep=',', encoding='utf-8')
        return pd.read_excel(filepath, sheet_name=sheet_name, engine="openpyxl")
    except: return None

def crear_vista_pivot_anos(df_input, indice, valor='Primas'):
    """Helper para crear vistas matriciales por Año."""
    try:
        if 'Año' not in df_input.columns: return pd.DataFrame()
        
        pivot = df_input.pivot_table(
            index=indice, columns='Año', values=valor, aggfunc='sum', fill_value=0
        )
        pivot['TOTAL CONSOLIDADO'] = pivot.sum(axis=1)
        pivot = pivot.sort_values('TOTAL CONSOLIDADO', ascending=False)
        # Convertir años a string para evitar problemas de formato
        pivot.columns = [str(c) for c in pivot.columns]
        return pivot.reset_index()
    except Exception:
        return pd.DataFrame()

# ==========================================
# 4. CRUCE INTELIGENTE (OPTIMIZADO)
# ==========================================
def normalize_text(text):
    if pd.isna(text): return ""
    return re.sub(r'[^\w\s]', '', str(text).upper().strip())

@st.cache_data(show_spinner=False)
def fuzzy_merge(df_left, df_right, left_on, right_on, threshold=85):
    print("--- [UTILS] Cruzando bases...")
    
    # 1. Normalizar
    s_left = df_left.copy()
    s_right = df_right.copy()
    s_left['key_norm'] = s_left[left_on].apply(normalize_text)
    s_right['key_norm'] = s_right[right_on].apply(normalize_text)
    
    # 2. Extraer Únicos (Velocidad x100)
    unique_left = s_left['key_norm'].unique()
    choices = s_right['key_norm'].unique().tolist()
    
    # 3. Calcular matches solo para únicos
    match_dict = {}
    for name in unique_left:
        if not name: continue
        match = process.extractOne(name, choices, scorer=fuzz.token_sort_ratio)
        if match and match[1] >= threshold:
            match_dict[name] = (match[0], match[1])
        else:
            match_dict[name] = (None, 0)
            
    # 4. Aplicar al dataframe completo
    s_left['match_info'] = s_left['key_norm'].map(match_dict)
    s_left['match_name'] = s_left['match_info'].apply(lambda x: x[0] if x else None)
    s_left['match_score'] = s_left['match_info'].apply(lambda x: x[1] if x else 0)
    
    # 5. Merge Final
    # Limpiamos duplicados del lado derecho antes de pegar
    right_clean = s_right.drop_duplicates(subset=['key_norm'])
    
    result = pd.merge(s_left, right_clean, left_on='match_name', right_on='key_norm', how='left', suffixes=('', '_right'))
    
    cols_drop = ['key_norm', 'match_info', 'key_norm_right']
    result.drop(columns=[c for c in cols_drop if c in result.columns], inplace=True)
    
    print("--- [UTILS] Cruce terminado.")
    return result

def analisis_ia_3_puntos(api_key, prompt, contexto):
    if not api_key:
        return "⚠️ No se detectó la API Key."
    client = OpenAI(api_key=api_key)
    full_prompt = (
        f"Actúa como analista de datos de seguros. Analiza la siguiente información:\n"
        f"{contexto}\n\n"
        f"{prompt}\n"
        f"Responde en 3 puntos clave: 1) ¿Qué muestra la gráfica/tabla? 2) ¿Qué es lo más importante? 3) ¿Cuál es el insight o prioridad principal? Sé breve y muy preciso."
    )
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": full_prompt}],
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error IA: {str(e)}"
