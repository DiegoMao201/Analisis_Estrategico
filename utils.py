import streamlit as st
import pandas as pd
import os
import datetime
import re
import gc
from fpdf import FPDF
from fuzzywuzzy import process, fuzz
import openai
from openai import OpenAI

# ✅ NUEVO: para leer tamaño de imágenes y escalarlas a la página (sin Pillow)
try:
    import matplotlib.image as mpimg
except Exception:
    mpimg = None

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
# 2. MOTOR PDF (BLINDADO Y NIVEL EJECUTIVO)
# ==========================================
class UltimatePDF(FPDF):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.set_margins(15, 20, 15)
        self.set_auto_page_break(auto=True, margin=15)

        # ✅ NUEVO: etiqueta de periodo (para header profesional, sin fechas “raras”)
        self.period_label = ""

        # === Fuentes Unicode (carpeta fonts/) ===
        try:
            font_regular = get_file_path("fonts/DejaVuSans.ttf")
            font_bold = get_file_path("fonts/DejaVuSans-Bold.ttf")
            font_italic = get_file_path("fonts/DejaVuSans-Oblique.ttf")

            if not os.path.exists(font_regular):
                raise FileNotFoundError(f"No existe: {font_regular}")

            self.add_font("DejaVu", "", font_regular, uni=True)

            if os.path.exists(font_bold):
                self.add_font("DejaVu", "B", font_bold, uni=True)
            else:
                self.add_font("DejaVu", "B", font_regular, uni=True)

            if os.path.exists(font_italic):
                self.add_font("DejaVu", "I", font_italic, uni=True)
            else:
                self.add_font("DejaVu", "I", font_regular, uni=True)

            self.font_family_base = "DejaVu"
        except Exception as e:
            print(f"⚠️ Advertencia: No se encontró fuente Unicode en /fonts. Usando Arial. ({e})")
            self.font_family_base = "Arial"

        self.set_font(self.font_family_base, "", 11)

    # ✅ NUEVO: control fino para evitar “saltos feos” y páginas vacías
    def _ensure_space(self, needed_mm: float) -> None:
        try:
            trigger = getattr(self, "page_break_trigger", None)
            if trigger is None:
                trigger = self.h - self.b_margin
            if self.get_y() + float(needed_mm) > float(trigger):
                self.add_page()
        except Exception:
            # fallback: no rompe
            pass

    def _clean_text(self, text):
        if not text:
            return ""
        text = str(text)
        text = text.replace("\xa0", " ").replace("\u200b", "").replace("\t", "    ")
        words = text.split(" ")
        safe_words = [w[:80] + "..." if len(w) > 80 else w for w in words]
        return " ".join(safe_words)

    def header(self):
        if self.page_no() > 1:
            self.set_font(self.font_family_base, "B", 9)
            self.set_text_color(100, 100, 100)

            self.cell(0, 10, "ALSUM - INTELIGENCIA DE NEGOCIOS", 0, 0, "L")

            # ✅ Antes mostraba la fecha del sistema (puede salir 2026). Ahora: PERIODO del análisis.
            right = f"Periodo: {self.period_label}" if self.period_label else ""
            self.cell(0, 10, right, 0, 1, "R")

            self.set_draw_color(0, 74, 143)
            self.set_line_width(0.5)
            self.line(15, 20, 195, 20)
            self.ln(8)

    def footer(self):
        self.set_y(-15)
        self.set_font(self.font_family_base, "I", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f"Página {self.page_no()}", 0, 0, "C")

    def cover_page(self, title, subtitle):
        self.add_page()
        self.set_fill_color(0, 74, 143)
        self.rect(0, 0, 210, 297, "F")
        self.set_text_color(255, 255, 255)
        self.set_font(self.font_family_base, "B", 40)
        self.ln(70)
        self.cell(0, 20, "ALSUM", 0, 1, "C")

        self.set_font(self.font_family_base, "", 14)
        self.set_text_color(200, 215, 230)
        self.cell(0, 10, "ESTRATEGIA & MERCADO", 0, 1, "C")

        self.ln(35)
        self.set_text_color(255, 255, 255)
        self.set_font(self.font_family_base, "B", 28)
        self.multi_cell(0, 15, self._clean_text(title), 0, "C")

        self.ln(12)
        self.set_font(self.font_family_base, "I", 16)
        self.set_text_color(220, 220, 220)
        self.multi_cell(0, 10, self._clean_text(subtitle), 0, "C")

    def chapter_body(self, text):
        self.set_font(self.font_family_base, "", 11)
        self.set_text_color(50, 50, 50)
        self.multi_cell(180, 6, self._clean_text(text))
        self.ln(3)

    # ✅ Ajuste: menos aire (más ejecutivo/compacto)
    def section_title(self, title, tight: bool = False):
        self._ensure_space(14 if tight else 18)
        self.set_font(self.font_family_base, "B", 15 if tight else 16)
        self.set_text_color(0, 74, 143)
        self.ln(4 if tight else 7)
        self.cell(0, 9 if tight else 10, self._clean_text(title), 0, 1, "L")
        self.set_draw_color(200, 200, 200)
        self.set_line_width(0.2)
        self.line(self.get_x(), self.get_y(), 195, self.get_y())
        self.ln(2 if tight else 4)

    def executive_summary(self, text):
        self.add_page()
        self.set_font(self.font_family_base, "B", 18)
        self.set_text_color(0, 74, 143)
        self.cell(0, 12, "Resumen Ejecutivo", 0, 1, "L")
        self.ln(2)

        self.set_font(self.font_family_base, "", 12)
        self.set_text_color(40, 40, 40)
        self.multi_cell(180, 7, self._clean_text(text))
        self.ln(6)

    def key_findings(self, findings):
        self.set_font(self.font_family_base, "B", 14)
        self.set_text_color(0, 74, 143)
        self.cell(0, 9, "Hallazgos Clave", 0, 1, "L")
        self.ln(1)

        self.set_font(self.font_family_base, "", 11)
        self.set_text_color(40, 40, 40)
        for point in findings:
            clean_point = self._clean_text(point)
            if clean_point.strip():
                self.cell(5, 6, "•", 0, 0, "L")
                self.multi_cell(175, 6, clean_point)
                self.ln(1)
        self.ln(3)

    def recommendations(self, recs):
        self.set_font(self.font_family_base, "B", 14)
        self.set_text_color(0, 74, 143)
        self.cell(0, 9, "Recomendaciones Estratégicas", 0, 1, "L")
        self.ln(1)

        self.set_font(self.font_family_base, "", 11)
        self.set_text_color(40, 40, 40)
        for rec in recs:
            clean_rec = self._clean_text(rec)
            if clean_rec.strip():
                self.cell(5, 6, "→", 0, 0, "L")
                self.multi_cell(175, 6, clean_rec)
                self.ln(1)
        self.ln(3)

    def add_section(self, title, content):
        self.section_title(title, tight=False)
        self.chapter_body(content)

    # ✅ Ajuste: tabla más compacta y evita cortes raros
    def add_table(self, data, col_widths=None, align="L", row_h: int = 7, after_space: int = 4):
        self.set_font(self.font_family_base, "", 9)
        if not data:
            self.cell(0, 8, "Sin datos disponibles.", 0, 1)
            return

        n_cols = len(data[0])
        if not col_widths:
            col_width = int(180 / n_cols)
            col_widths = [col_width] * n_cols

        # espacio aproximado mínimo para header + 2 filas
        self._ensure_space(max(18, row_h * 3))

        for row_idx, row in enumerate(data):
            if row_idx == 0:
                self.set_fill_color(240, 240, 240)
                self.set_font(self.font_family_base, "B", 9)
            else:
                self.set_fill_color(255, 255, 255)
                self.set_font(self.font_family_base, "", 9)

            for i, datum in enumerate(row):
                text = self._clean_text(str(datum))
                max_chars = max(3, int(col_widths[i] * 0.45))
                if len(text) > max_chars:
                    text = text[: max_chars - 3] + "..."
                self.cell(col_widths[i], row_h, text, border=1, align=align, fill=True)
            self.ln(row_h)

        self.ln(after_space)

    # ✅ NUEVO: calcula relación de aspecto (alto/ancho) para escalar y que nunca “se vaya” abajo
    def _img_ratio(self, image_path: str) -> float:
        if not mpimg or not os.path.exists(image_path):
            return 0.6  # fallback razonable
        try:
            arr = mpimg.imread(image_path)
            h_px = float(arr.shape[0])
            w_px = float(arr.shape[1])
            if w_px <= 0:
                return 0.6
            return h_px / w_px
        except Exception:
            return 0.6

    # ✅ Ajuste crítico: imagen SIEMPRE cabe; si no, agrega página o reduce tamaño
    def add_image_section(self, title, image_path, w=170, h=0, tight: bool = True):
        # reservamos espacio para título + un mínimo de imagen
        self._ensure_space(35)

        self.section_title(title, tight=tight)
        if w > 180:
            w = 180

        if not os.path.exists(image_path):
            self.set_font(self.font_family_base, "I", 10)
            self.set_text_color(255, 0, 0)
            self.cell(0, 8, "Error: Imagen de gráfico no encontrada o no pudo ser generada.", 0, 1)
            self.ln(4)
            return

        # calcular alto según ratio y ajustar a espacio disponible
        ratio = self._img_ratio(image_path)
        h_calc = (w * ratio) if not h else float(h)

        avail = (self.h - self.b_margin) - self.get_y() - 2
        if avail < 35:
            self.add_page()
            self.section_title(title, tight=tight)
            avail = (self.h - self.b_margin) - self.get_y() - 2

        if h_calc > avail and avail > 10:
            # Escala proporcional para que quepa
            scale = avail / h_calc
            w = max(120, w * scale)
            h_calc = avail

        x_pos = 15 + ((180 - w) / 2)
        self.image(image_path, x=x_pos, w=w, h=h_calc)
        self.ln(6)

    def annex(self, text):
        self.add_page()
        self.set_font(self.font_family_base, "B", 16)
        self.set_text_color(100, 100, 100)
        self.cell(0, 10, "Anexos & Metodología", 0, 1, "L")
        self.ln(3)

        self.set_font(self.font_family_base, "", 10)
        self.set_text_color(80, 80, 80)
        self.multi_cell(180, 6, self._clean_text(text))
        self.ln(4)

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
            sep = ','
            enc = 'utf-8'
            
            try:
                with open(filepath, 'rb') as f:
                    sample = f.read(10000) 
                
                # Detectar Encoding
                for e in ['utf-8', 'latin-1', 'cp1252']:
                    try:
                        text_sample = sample.decode(e)
                        enc = e
                        break
                    except:
                        continue
                
                # Detectar Separador
                if text_sample.count(';') > text_sample.count(','):
                    sep = ';'
                
                print(f"--- [UTILS] Formato detectado: Separador='{sep}' | Encoding='{enc}'")
                df = pd.read_csv(filepath, sep=sep, encoding=enc, on_bad_lines='skip', low_memory=False)
                
            except Exception as e:
                print(f"--- [ERROR DETECCIÓN] {e}. Intentando fallback...")
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
        for col, val in defaults.items():
            if col in df.columns:
                df[col] = df[col].fillna(val)
        
        if 'AFILIADO' in df.columns:
            df['AFILIADO'] = df['AFILIADO'].astype(str).str.upper()
            mask_no = df['AFILIADO'].str.contains('NO')
            df.loc[mask_no, 'AFILIADO'] = 'NO AFILIADO'
            df.loc[~mask_no, 'AFILIADO'] = 'AFILIADO'

        if 'USD' in df.columns: 
            df['USD'] = df['USD'].apply(parse_numero_latino)

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
    right_clean = s_right.drop_duplicates(subset=['key_norm'])
    result = pd.merge(s_left, right_clean, left_on='match_name', right_on='key_norm', how='left', suffixes=('', '_right'))
    
    cols_drop = ['key_norm', 'match_info', 'key_norm_right']
    result.drop(columns=[c for c in cols_drop if c in result.columns], inplace=True)
    
    print("--- [UTILS] Cruce terminado.")
    return result

# ==========================================
# 5. FUNCIONES DE IA
# ==========================================
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

# ✅ NUEVO: Export Excel ejecutivo (XLSX) con formato profesional
def export_excel_executive(
    df: pd.DataFrame,
    *,
    sheet_name: str = "Data",
    title: str = "ALSUM | Export",
    subtitle: str = "",
    period_label: str = "",
) -> bytes:
    """
    Exporta un Excel (.xlsx) con formato ejecutivo:
    - Header (título/subtítulo/periodo/fecha)
    - Tabla con estilo, autofiltro, freeze panes
    - Formatos: moneda/porcentaje/enteros
    - Ajuste de anchos + configuración de impresión
    """
    import io

    if df is None:
        df = pd.DataFrame()

    dfx = df.copy()

    # Evitar problemas con zonas horarias / tipos raros
    for c in dfx.columns:
        if pd.api.types.is_datetime64tz_dtype(dfx[c]):
            dfx[c] = dfx[c].dt.tz_convert(None)

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        wb = writer.book

        # Paleta ALSUM
        color_primary = "#004A8F"
        color_text = "#0f172a"
        color_muted = "#64748b"
        color_header_bg = "#EEF2F7"
        color_border = "#CBD5E1"

        # Formats
        fmt_title = wb.add_format(
            {"bold": True, "font_size": 18, "font_color": "white", "bg_color": color_primary, "align": "left", "valign": "vcenter"}
        )
        fmt_subtitle = wb.add_format({"bold": True, "font_size": 11, "font_color": color_text, "align": "left", "valign": "vcenter"})
        fmt_meta = wb.add_format({"font_size": 9, "font_color": color_muted, "align": "left", "valign": "vcenter"})

        fmt_hdr = wb.add_format(
            {"bold": True, "font_size": 10, "font_color": color_text, "bg_color": color_header_bg, "border": 1, "border_color": color_border, "align": "center", "valign": "vcenter"}
        )
        fmt_txt = wb.add_format({"font_size": 10, "font_color": color_text, "border": 1, "border_color": color_border})
        fmt_int = wb.add_format({"font_size": 10, "font_color": color_text, "num_format": "#,##0", "border": 1, "border_color": color_border})
        fmt_usd = wb.add_format({"font_size": 10, "font_color": color_text, "num_format": "$#,##0", "border": 1, "border_color": color_border})
        fmt_pct = wb.add_format({"font_size": 10, "font_color": color_text, "num_format": "0.0%", "border": 1, "border_color": color_border})

        # Sheet
        sheet = sheet_name[:31] if sheet_name else "Data"
        ws = wb.add_worksheet(sheet)
        writer.sheets[sheet] = ws

        # Layout de cabecera (3 filas)
        ws.set_row(0, 28)
        ws.merge_range(0, 0, 0, max(0, len(dfx.columns) - 1), title, fmt_title)

        meta_left = []
        if subtitle:
            meta_left.append(subtitle)
        if period_label:
            meta_left.append(f"Periodo: {period_label}")
        meta_left.append(f"Generado: {datetime.date.today().strftime('%d/%m/%Y')}")
        ws.write(1, 0, " | ".join([m for m in meta_left if m]), fmt_meta)

        # Si dataframe vacío
        if dfx.empty:
            ws.write(3, 0, "Sin datos para exportar con los filtros actuales.", fmt_subtitle)
            ws.set_column(0, 0, 70)
            # Config impresión
            ws.set_landscape()
            ws.fit_to_pages(1, 1)
            ws.set_margins(left=0.4, right=0.4, top=0.5, bottom=0.5)
            return output.getvalue()

        # Escribir DF a partir de fila 3 (0-index)
        start_row = 3
        start_col = 0

        # Convertir nombres a string
        cols = [str(c) for c in dfx.columns.tolist()]
        dfx.columns = cols

        # Write headers
        for j, c in enumerate(cols):
            ws.write(start_row, start_col + j, c, fmt_hdr)

        # Heurísticas de formato por columna
        def _col_kind(name: str) -> str:
            n = str(name).strip().lower()
            if n in {"año", "ano", "year"}:
                return "int"
            if "siniestralidad" in n or n.endswith("%") or "pct" in n or "porcentaje" in n:
                return "pct"
            if any(k in n for k in ["usd", "prima", "primas", "siniestro", "siniestros", "resultado", "monto", "valor", "total"]):
                return "usd"
            return "txt"

        col_kinds = [_col_kind(c) for c in cols]

        # Write data rows
        for i in range(len(dfx)):
            for j, c in enumerate(cols):
                val = dfx.iat[i, j]
                kind = col_kinds[j]

                # Normalización de NaN
                if pd.isna(val):
                    val = ""

                # Escribir según tipo
                if kind == "int":
                    try:
                        ws.write_number(start_row + 1 + i, start_col + j, int(val), fmt_int)
                    except Exception:
                        ws.write(start_row + 1 + i, start_col + j, str(val), fmt_txt)
                elif kind == "usd":
                    try:
                        ws.write_number(start_row + 1 + i, start_col + j, float(val), fmt_usd)
                    except Exception:
                        ws.write(start_row + 1 + i, start_col + j, str(val), fmt_txt)
                elif kind == "pct":
                    # Si viene como 50.2 (porcentaje), lo convertimos a 0.502 para formato %
                    try:
                        v = float(val)
                        v = (v / 100.0) if v > 1.5 else v
                        ws.write_number(start_row + 1 + i, start_col + j, v, fmt_pct)
                    except Exception:
                        ws.write(start_row + 1 + i, start_col + j, str(val), fmt_txt)
                else:
                    ws.write(start_row + 1 + i, start_col + j, str(val), fmt_txt)

        last_row = start_row + len(dfx)
        last_col = start_col + len(cols) - 1

        # Freeze panes debajo del header
        ws.freeze_panes(start_row + 1, 0)

        # Autofilter
        ws.autofilter(start_row, start_col, last_row, last_col)

        # Ajuste de anchos (capado)
        for j, c in enumerate(cols):
            sample = dfx[c].astype(str).head(200).tolist()
            max_len = max([len(str(c))] + [len(s) for s in sample]) if sample else len(str(c))
            width = min(max(10, max_len + 2), 45)
            ws.set_column(j, j, width)

        # Condicional ejecutivo para siniestralidad (si existe)
        for j, c in enumerate(cols):
            if "siniestralidad" in c.lower():
                # escala 0% a 100% (verde->amarillo->rojo)
                ws.conditional_format(start_row + 1, j, last_row, j, {"type": "3_color_scale"})
                break

        # Config de impresión
        ws.set_landscape()
        ws.fit_to_pages(1, 1)
        ws.set_margins(left=0.4, right=0.4, top=0.5, bottom=0.5)
        ws.repeat_rows(start_row, start_row)  # repetir header en impresión

    return output.getvalue()