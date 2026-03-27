import os
import re
import unicodedata
from decimal import Decimal, InvalidOperation
from html import escape
from urllib.parse import quote

import pandas as pd
import streamlit as st

import utils


st.set_page_config(
    page_title="ALSUM 360 | Centro de Envio",
    layout="wide",
    page_icon="📨",
    initial_sidebar_state="expanded",
)


PRIMARY = "#003366"
SECONDARY = "#0E5CAD"
SURFACE = "#F4F6F9"
MUTED = "#475569"


st.markdown(
    f"""
<style>
    .stApp {{
        background:
            radial-gradient(circle at top right, rgba(14, 92, 173, 0.12), transparent 24%),
            linear-gradient(180deg, #f8fbff 0%, {SURFACE} 42%, #eef3f8 100%);
    }}

    .block-container {{
        padding-top: 1.8rem;
        padding-bottom: 2rem;
    }}

    h1, h2, h3 {{
        color: {PRIMARY};
        font-family: 'Segoe UI', sans-serif;
    }}

    div[data-testid="stMetric"] {{
        background: rgba(255, 255, 255, 0.92);
        border: 1px solid #dbe4ee;
        border-left: 5px solid {PRIMARY};
        border-radius: 14px;
        padding: 16px;
        box-shadow: 0 12px 24px rgba(15, 23, 42, 0.05);
    }}

    .hero-card {{
        background: linear-gradient(135deg, rgba(0, 51, 102, 0.97) 0%, rgba(14, 92, 173, 0.94) 100%);
        border-radius: 22px;
        padding: 26px 28px;
        color: white;
        box-shadow: 0 20px 40px rgba(0, 51, 102, 0.16);
        margin-bottom: 1rem;
    }}

    .hero-kicker {{
        text-transform: uppercase;
        letter-spacing: 0.18em;
        font-size: 0.76rem;
        opacity: 0.8;
        margin-bottom: 0.6rem;
    }}

    .hero-title {{
        font-size: 2.15rem;
        font-weight: 700;
        margin-bottom: 0.55rem;
    }}

    .hero-copy {{
        font-size: 1rem;
        color: rgba(255, 255, 255, 0.86);
        max-width: 960px;
        line-height: 1.55;
    }}

    .section-card {{
        background: rgba(255, 255, 255, 0.94);
        border: 1px solid #dbe4ee;
        border-radius: 18px;
        padding: 18px 20px;
        box-shadow: 0 10px 24px rgba(15, 23, 42, 0.05);
    }}

    .mini-label {{
        color: {MUTED};
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-size: 0.74rem;
        font-weight: 700;
    }}

    .contact-card {{
        background: rgba(255, 255, 255, 0.96);
        border: 1px solid #dbe4ee;
        border-radius: 18px;
        padding: 16px 18px;
        box-shadow: 0 8px 22px rgba(15, 23, 42, 0.05);
    }}

    .contact-name {{
        color: #0f172a;
        font-size: 1rem;
        font-weight: 700;
        margin-bottom: 0.15rem;
    }}

    .contact-meta {{
        color: {MUTED};
        font-size: 0.88rem;
        line-height: 1.45;
    }}

    .status-ok, .status-review, .status-error {{
        display: inline-block;
        padding: 0.24rem 0.65rem;
        border-radius: 999px;
        font-size: 0.76rem;
        font-weight: 700;
        margin-top: 0.35rem;
    }}

    .status-ok {{
        color: #166534;
        background: rgba(22, 163, 74, 0.12);
    }}

    .status-review {{
        color: #92400e;
        background: rgba(245, 158, 11, 0.16);
    }}

    .status-error {{
        color: #991b1b;
        background: rgba(220, 38, 38, 0.12);
    }}

    .token-chip {{
        display: inline-block;
        margin: 0.2rem 0.35rem 0.2rem 0;
        padding: 0.28rem 0.62rem;
        border-radius: 999px;
        background: rgba(14, 92, 173, 0.09);
        color: {SECONDARY};
        font-size: 0.8rem;
        font-weight: 700;
    }}

    .future-note {{
        background: rgba(14, 92, 173, 0.06);
        border: 1px dashed rgba(14, 92, 173, 0.28);
        border-radius: 16px;
        padding: 16px 18px;
        color: {MUTED};
    }}

    .stButton > button, div[data-testid="stLinkButton"] a {{
        border-radius: 10px !important;
        font-weight: 700 !important;
    }}

    .stTextArea textarea, .stTextInput input, .stSelectbox div[data-baseweb="select"] > div, .stMultiSelect div[data-baseweb="select"] > div {{
        border-radius: 12px !important;
    }}
</style>
""",
    unsafe_allow_html=True,
)


def get_contacts_path() -> str | None:
    for filename in ("contactos_Alsum_4.xlsx", "contactos_Alsum_4.csv"):
        candidate = utils.get_file_path(filename)
        if os.path.exists(candidate):
            return candidate
    return None


def read_contacts_file(filepath: str) -> pd.DataFrame:
    if filepath.lower().endswith(".csv"):
        try:
            return pd.read_csv(filepath, sep=";", encoding="latin-1")
        except Exception:
            return pd.read_csv(filepath, encoding="utf-8")
    return pd.read_excel(filepath, engine="openpyxl")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    normalized.columns = [str(column).strip() for column in normalized.columns]
    return normalized


def slug_column_name(value: object) -> str:
    text = str(value or "").strip().lower()
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("utf-8")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def pick_column(df: pd.DataFrame, aliases: list[str]) -> str | None:
    alias_lookup = {slug_column_name(column): column for column in df.columns}
    for alias in aliases:
        match = alias_lookup.get(slug_column_name(alias))
        if match:
            return match
    return None


def standardize_contact_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], bool]:
    mapping = {
        "Cargo": ["Cargo", "Rol", "Puesto"],
        "Empresa": ["Empresa", "Organizacion", "Compania", "Compania aseguradora", "Company"],
        "Nombre": ["Nombre", "Contacto", "Nombre contacto", "Destinatario"],
        "Celular": ["Celular", "Telefono", "Telefono movil", "Movil", "Movil WhatsApp", "Telefono WhatsApp", "WhatsApp"],
        "Correo": ["Correo", "Correo electronico", "Email", "E-mail", "Mail"],
    }
    standardized = pd.DataFrame(index=df.index)
    required = ["Cargo", "Nombre", "Celular"]
    has_email_column = False

    for canonical_name, aliases in mapping.items():
        source_column = pick_column(df, aliases)
        if source_column:
            standardized[canonical_name] = df[source_column]
            if canonical_name == "Correo":
                has_email_column = True

    if "Empresa" not in standardized.columns:
        standardized["Empresa"] = ""
    if "Correo" not in standardized.columns:
        standardized["Correo"] = ""

    missing_columns = [column for column in required if column not in standardized.columns]
    return standardized, missing_columns, has_email_column


def clean_contact_name(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).replace("\xa0", " ").strip()
    text = re.sub(r"\s+", " ", text)
    return text.strip(" -")


def clean_company_name(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).replace("\xa0", " ").strip()
    return re.sub(r"\s+", " ", text)


def extract_short_name(person: str) -> str:
    connectors = {"de", "del", "la", "las", "los", "y", "e", "da", "das", "do", "dos"}
    for token in person.split():
        clean_token = re.sub(r"[^A-Za-z]", "", token)
        if clean_token and clean_token.lower() not in connectors:
            return clean_token.title()
    return "Colega"


def normalize_phone(value: object) -> str:
    if pd.isna(value):
        return ""

    if isinstance(value, int):
        return str(value)

    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return format(Decimal(str(value)), "f").replace(".", "")

    text = str(value).replace("\xa0", " ").replace("�", "").strip()
    scientific = text.replace(" ", "").replace(",", ".")
    if re.fullmatch(r"[+\-]?\d+(?:\.\d+)?[eE][+\-]?\d+", scientific):
        try:
            return format(Decimal(scientific), "f").replace(".", "")
        except InvalidOperation:
            pass

    digits = re.sub(r"\D", "", text)
    if digits.startswith("00"):
        digits = digits[2:]
    return digits


def strip_leading_symbols(value: object) -> str:
    text = str(value or "").replace("\xa0", " ").replace("�", "").strip()
    return re.sub(r"^[^\d+]+", "", text)


def has_explicit_international_prefix(value: object) -> bool:
    if pd.isna(value):
        return False
    text = strip_leading_symbols(value)
    return text.startswith("+") or text.startswith("00")


def detect_phone_source_format(value: object) -> str:
    if pd.isna(value):
        return "Sin dato"
    text = strip_leading_symbols(value)
    compact = text.replace(" ", "")
    if compact.startswith("+"):
        return "Internacional con +"
    if compact.startswith("00"):
        return "Internacional con 00"
    if re.fullmatch(r"[+\-]?\d+(?:[\.,]\d+)?[eE][+\-]?\d+", compact):
        return "Formato cientifico"
    return "Numerico estandar"


def classify_phone(phone: str) -> tuple[str, str]:
    if not phone:
        return "Sin celular", "error"
    if len(phone) < 9:
        return "Numero demasiado corto", "error"
    if len(phone) < 11:
        return "Revisar indicativo", "review"
    return "Listo para WhatsApp", "ok"


def apply_default_prefix(phone: str, prefix: str, raw_phone: object = "") -> str:
    clean_prefix = re.sub(r"\D", "", prefix or "")
    if not clean_prefix or not phone:
        return phone
    if has_explicit_international_prefix(raw_phone):
        return phone
    if len(phone) >= 11:
        return phone
    return f"{clean_prefix}{phone}"


def normalize_email(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip().replace("\xa0", " ")
    return re.sub(r"\s+", "", text).lower()


def has_valid_email(value: object) -> bool:
    text = normalize_email(value)
    return bool(re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", text))


def render_tokens(template: str, row: pd.Series, sender_name: str) -> str:
    replacements = {
        "{nombre}": row["short_name"],
        "{nombre_completo}": row["Nombre"],
        "{empresa}": row["Empresa"] or "su organizacion",
        "{cargo}": row["Cargo"] or "su cargo",
        "{correo}": row.get("Correo", ""),
        "{celular}": row.get("Celular WhatsApp", ""),
        "{remitente}": sender_name.strip() or "Lina Marcela Contreras",
    }
    rendered = template or ""
    for token, value in replacements.items():
        rendered = rendered.replace(token, str(value or ""))
    return rendered.strip()


def build_message(
    row: pd.Series,
    custom_message: str,
    campaign_type: str,
    sender_name: str,
    closing_line: str,
) -> str:
    openings = {
        "Invitacion ejecutiva": (
            f"Apreciado/a {row['Nombre']},\n\n"
            "Reciba un cordial saludo. Desde ALSUM | queremos compartirle una comunicacion prioritaria dentro de nuestra agenda institucional y de relacionamiento ejecutivo."
        ),
        "Convocatoria comercial": (
            f"Apreciado/a {row['Nombre']},\n\n"
            "Espero que se encuentre muy bien. Le escribimos desde ALSUM | acercarle una convocatoria con enfoque profesional y alto valor para nuestra red de aliados y decisores."
        ),
        "Seguimiento institucional": (
            f"Buen dia, {row['Nombre']}.\n\n"
            "Con un atento saludo, compartimos esta comunicacion desde ALSUM | como parte de nuestro seguimiento institucional y relacionamiento con actores clave del sector."
        ),
    }

    contextual_lines = []
    if row["Empresa"]:
        contextual_lines.append(f"Hemos incluido especialmente a {row['Empresa']} dentro de esta difusion por la relevancia de su participacion.")
    if row["Cargo"] and row["Cargo"].lower() != "sin cargo":
        contextual_lines.append(f"Esta comunicacion esta dirigida al cargo de {row['Cargo']} para facilitar una segmentacion precisa del mensaje.")

    body = render_tokens(custom_message.strip(), row, sender_name)
    closing = render_tokens(
        closing_line.strip() or "Sera un gusto contar con su atencion y participacion.",
        row,
        sender_name,
    )
    signature = sender_name.strip() or "Lina Marcela Contreras"

    parts = [openings[campaign_type]]
    if contextual_lines:
        parts.append("\n".join(contextual_lines))
    if body:
        parts.append(body)
    if closing:
        parts.append(closing)
    parts.append(f"Cordialmente,\n{signature}\nALSUM | Gestora Comercial")
    return "\n\n".join(part for part in parts if part.strip())


def build_custom_message(row: pd.Series, template: str, sender_name: str, add_signature: bool) -> str:
    body = render_tokens(template.strip(), row, sender_name)
    signature = sender_name.strip() or "Lina Marcela Contreras"
    if add_signature and signature and signature not in body:
        body = f"{body}\n\n{signature}".strip()
    return body


def build_email_subject(row: pd.Series, template: str, sender_name: str) -> str:
    return render_tokens(template.strip(), row, sender_name)


def build_email_body(row: pd.Series, template: str, sender_name: str) -> str:
    return render_tokens(template.strip(), row, sender_name)


def build_whatsapp_link(phone: str, message: str) -> str:
    return f"https://wa.me/{phone}?text={quote(message)}"


def apply_contact_filters(df: pd.DataFrame, selected_cargos: list[str], only_ready: bool, search_text: str) -> pd.DataFrame:
    filtered = df.copy()
    if selected_cargos:
        filtered = filtered[filtered["Cargo"].isin(selected_cargos)]
    if only_ready:
        filtered = filtered[filtered["status_kind_final"] == "ok"]
    if search_text.strip():
        mask = (
            filtered["Nombre"].str.contains(search_text, case=False, na=False)
            | filtered["Empresa"].str.contains(search_text, case=False, na=False)
            | filtered["Cargo"].str.contains(search_text, case=False, na=False)
            | filtered["Celular WhatsApp"].str.contains(search_text, case=False, na=False)
            | filtered["Correo"].str.contains(search_text, case=False, na=False)
        )
        filtered = filtered[mask]
    return filtered


def build_preview_label(row: pd.Series) -> str:
    company = row["Empresa"] or "Sin empresa"
    return f"{row['Nombre']} | {company} | {row['Cargo']}"


def prepare_export(df: pd.DataFrame, message_column: str, url_column: str) -> pd.DataFrame:
    export_df = df[
        [
            "Cargo",
            "Empresa",
            "Nombre",
            "Correo",
            "Formato celular",
            "Celular original",
            "Celular WhatsApp",
            "Estado final",
            url_column,
            message_column,
        ]
    ].copy()
    return export_df.rename(
        columns={
            "Formato celular": "Formato celular origen",
            "Celular WhatsApp": "Celular listo",
            url_column: "WhatsApp URL",
            message_column: "Mensaje preparado",
        }
    )


def render_contact_list(df: pd.DataFrame, message_column: str, url_column: str, empty_message: str) -> None:
    if df.empty:
        st.warning(empty_message)
        return

    for row in df.to_dict("records"):
        status_class = {
            "ok": "status-ok",
            "review": "status-review",
            "error": "status-error",
        }.get(row["status_kind_final"], "status-review")

        col_info, col_message, col_action = st.columns([1.25, 2.2, 0.95], gap="medium")

        with col_info:
            contact_name = escape(row["Nombre"])
            cargo = escape(row["Cargo"])
            company = escape(row["Empresa"] or "No identificada")
            source_format = escape(row["Formato celular"])
            original_phone = escape(str(row["Celular original"]))
            whatsapp_phone = escape(row["Celular WhatsApp"])
            email = escape(row.get("Correo", "") or "Pendiente")
            final_status = escape(row["Estado final"])
            st.markdown(
                f"""
                <div class="contact-card">
                    <div class="contact-name">{contact_name}</div>
                    <div class="contact-meta">
                        <strong>Cargo:</strong> {cargo}<br>
                        <strong>Empresa:</strong> {company}<br>
                        <strong>Correo:</strong> {email}<br>
                        <strong>Origen celular:</strong> {source_format}<br>
                        <strong>Celular original:</strong> {original_phone}<br>
                        <strong>Celular listo:</strong> {whatsapp_phone}<br>
                        <span class="{status_class}">{final_status}</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col_message:
            with st.expander(f"Ver mensaje para {row['short_name']}"):
                st.write(row[message_column])

        with col_action:
            if row["status_kind_final"] == "error":
                st.warning("Celular incompleto")
            else:
                st.link_button("Abrir WhatsApp", row[url_column], use_container_width=True)

        st.markdown("<div style='height: 0.55rem;'></div>", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def load_contacts() -> tuple[pd.DataFrame | None, str | None, str | None, bool]:
    contacts_path = get_contacts_path()
    if not contacts_path:
        return None, None, "No se encontro contactos_Alsum_4.xlsx ni contactos_Alsum_4.csv en la raiz del proyecto.", False

    try:
        df_contacts = normalize_columns(read_contacts_file(contacts_path))
    except Exception as exc:
        return None, contacts_path, f"No fue posible leer el archivo de contactos: {exc}", False

    standardized, missing_columns, has_email_column = standardize_contact_columns(df_contacts)
    if missing_columns:
        return None, contacts_path, f"Faltan columnas requeridas en la base de contactos: {', '.join(missing_columns)}", has_email_column

    base = standardized.copy()
    base["Cargo"] = base["Cargo"].fillna("Sin cargo").astype(str).str.strip().replace("", "Sin cargo")
    base["Empresa"] = base["Empresa"].map(clean_company_name)
    base["Nombre"] = base["Nombre"].map(clean_contact_name)
    base["Correo"] = base["Correo"].map(normalize_email)
    base["Celular original"] = base["Celular"]
    base["Formato celular"] = base["Celular"].map(detect_phone_source_format)
    base["Celular limpio"] = base["Celular"].map(normalize_phone)
    base["short_name"] = base["Nombre"].map(extract_short_name)
    base["Tiene correo valido"] = base["Correo"].map(has_valid_email)

    statuses = base["Celular limpio"].map(classify_phone)
    base[["Estado", "status_kind"]] = pd.DataFrame(statuses.tolist(), index=base.index)
    base["Nombre"] = base["Nombre"].replace("", "Contacto ALSUM")
    base["Orden"] = range(1, len(base) + 1)
    return base, contacts_path, None, has_email_column


st.markdown(
    """
    <div class="hero-card">
        <div class="hero-kicker">Outreach Layer</div>
        <div class="hero-title">Centro de Envio y Relacionamiento</div>
        <div class="hero-copy">
            Esta pagina ya opera como centro de contacto para ALSUM: limpia celulares con o sin +,
            segmenta por cargo, prepara mensajes filtrados para WhatsApp y queda lista para incorporar correo,
            SendGrid y futuras automatizaciones por API sobre la misma base de contactos.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.info(
    "Los celulares se normalizan siempre a formato internacional de solo digitos. "
    "Si el archivo trae +, 00, espacios, guiones o notacion cientifica, la pagina lo corrige antes de preparar el envio."
)


contacts_df, contacts_path, load_error, has_email_column = load_contacts()

if load_error:
    st.error(load_error)
    st.stop()


with st.sidebar:
    st.header("Configuracion del centro")
    if st.button("Recargar base de contactos", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    default_prefix = st.text_input(
        "Indicativo por defecto",
        value="",
        placeholder="Ejemplo: 57",
        help="Opcional. Solo se antepone a celulares cortos que no traen + ni codigo internacional.",
    )
    only_ready = st.toggle(
        "Mostrar solo listos para envio",
        value=False,
        help="Oculta contactos cuyo celular aun requiere revision manual.",
    )


working_df = contacts_df.copy()
working_df["Celular WhatsApp"] = working_df.apply(
    lambda row: apply_default_prefix(row["Celular limpio"], default_prefix, row["Celular original"]),
    axis=1,
)
working_df[["Estado final", "status_kind_final"]] = pd.DataFrame(
    working_df["Celular WhatsApp"].map(classify_phone).tolist(),
    index=working_df.index,
)
working_df["preview_label"] = working_df.apply(build_preview_label, axis=1)

cargo_options = sorted(option for option in working_df["Cargo"].dropna().astype(str).unique().tolist() if option)

filter_col, search_col = st.columns([1.15, 1.35], gap="large")
with filter_col:
    selected_cargos = st.multiselect(
        "Filtrar por cargo",
        options=cargo_options,
        default=cargo_options,
        help="Puedes dejar todos los cargos o enfocarte solo en un segmento antes de preparar los mensajes.",
    )
with search_col:
    search_text = st.text_input(
        "Buscar por nombre, empresa, cargo, celular o correo",
        value="",
        placeholder="Ejemplo: delegado, allianz, panama, nombre o correo",
    )

filtered_df = apply_contact_filters(working_df, selected_cargos, only_ready, search_text)

metric_1, metric_2, metric_3, metric_4, metric_5 = st.columns(5)
metric_1.metric("Base total", len(working_df))
metric_2.metric("Seleccion actual", len(filtered_df))
metric_3.metric("Listos WhatsApp", int((filtered_df["status_kind_final"] == "ok").sum()))
metric_4.metric("Revisar", int((filtered_df["status_kind_final"] == "review").sum()))
metric_5.metric("Con correo", int(filtered_df["Tiene correo valido"].sum()))

st.caption(f"Fuente activa: {os.path.basename(contacts_path)}")

tab_whatsapp, tab_personalizado, tab_email = st.tabs(
    ["WhatsApp institucional", "Mensaje personalizado", "Email y proximos canales"]
)


with tab_whatsapp:
    config_col, preview_col = st.columns([1.2, 0.8], gap="large")

    with config_col:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="mini-label">Plantilla principal</div>', unsafe_allow_html=True)
        st.subheader("Mensaje base para el filtro actual")
        campaign_type = st.selectbox(
            "Tipo de comunicacion",
            options=["Invitacion ejecutiva", "Convocatoria comercial", "Seguimiento institucional"],
            index=0,
            key="campaign_type",
        )
        custom_message = st.text_area(
            "Contenido principal",
            height=220,
            key="institutional_body",
            placeholder=(
                "Ejemplo:\n"
                "Nos complaceria contar con su presencia en el Congreso ALSUM 2026, un espacio de networking, analisis sectorial y conversaciones ejecutivas para la region.\n\n"
                "Puedes usar {nombre}, {nombre_completo}, {empresa}, {cargo}, {correo}, {celular} y {remitente}."
            ),
        )
        closing_line = st.text_input(
            "Cierre de mensaje",
            value="Sera un gusto contar con su participacion y confirmar su interes.",
            key="institutional_closing",
        )
        sender_name = st.text_input(
            "Firma remitente",
            value="Lina Marcela Contreras",
            key="institutional_sender",
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with preview_col:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="mini-label">Resumen operativo</div>', unsafe_allow_html=True)
        st.write(f"Se prepararan mensajes para {len(filtered_df)} contacto(s) segun el filtro activo.")
        st.write(f"Cargos seleccionados: {len(selected_cargos)} de {len(cargo_options)}")
        if not filtered_df.empty:
            preview_contact = st.selectbox(
                "Vista previa con",
                options=filtered_df["preview_label"].tolist(),
                index=0,
                key="preview_contact_institutional",
            )
        else:
            preview_contact = None
            st.warning("No hay contactos en el filtro actual.")
        st.markdown('</div>', unsafe_allow_html=True)

    if custom_message.strip() and not filtered_df.empty:
        institutional_df = filtered_df.copy()
        institutional_df["Mensaje institucional"] = institutional_df.apply(
            build_message,
            axis=1,
            custom_message=custom_message,
            campaign_type=campaign_type,
            sender_name=sender_name,
            closing_line=closing_line,
        )
        institutional_df["WhatsApp URL institucional"] = institutional_df.apply(
            lambda row: build_whatsapp_link(row["Celular WhatsApp"], row["Mensaje institucional"])
            if row["status_kind_final"] != "error"
            else "",
            axis=1,
        )

        preview_row = institutional_df.loc[institutional_df["preview_label"] == preview_contact].iloc[0]

        left_preview, right_preview = st.columns([1.05, 0.95], gap="large")
        with left_preview:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown('<div class="mini-label">Vista previa ejecutiva</div>', unsafe_allow_html=True)
            st.subheader(preview_row["Nombre"])
            st.caption(f"{preview_row['Cargo']} | {preview_row['Empresa'] or 'Sin empresa'}")
            st.text_area(
                "Mensaje final",
                value=preview_row["Mensaje institucional"],
                height=300,
                disabled=True,
                label_visibility="collapsed",
            )
            st.markdown('</div>', unsafe_allow_html=True)

        with right_preview:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown('<div class="mini-label">Base preparada</div>', unsafe_allow_html=True)
            prepared_df = prepare_export(institutional_df, "Mensaje institucional", "WhatsApp URL institucional")
            st.dataframe(
                prepared_df[["Cargo", "Empresa", "Nombre", "Celular listo", "Estado final"]],
                use_container_width=True,
                hide_index=True,
            )
            st.download_button(
                "Descargar base preparada de WhatsApp",
                data=prepared_df.to_csv(index=False).encode("utf-8-sig"),
                file_name="whatsapp_institucional_preparado.csv",
                mime="text/csv",
                use_container_width=True,
            )
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("### Lista operativa del filtro")
        st.caption("Cada boton abre WhatsApp Web con el mensaje institucional ya personalizado para el destinatario filtrado.")
        render_contact_list(
            institutional_df,
            "Mensaje institucional",
            "WhatsApp URL institucional",
            "No hay contactos para mostrar con los filtros actuales.",
        )
    elif filtered_df.empty:
        st.warning("No hay contactos para preparar mensajes con el filtro actual.")
    else:
        st.info("Escribe el contenido principal para generar el mensaje institucional sobre el filtro actual.")


with tab_personalizado:
    config_col, preview_col = st.columns([1.12, 0.88], gap="large")

    with config_col:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="mini-label">Plantilla libre</div>', unsafe_allow_html=True)
        st.subheader("Mensaje personalizado para todo el filtro")
        personalized_template = st.text_area(
            "Plantilla del mensaje",
            height=260,
            key="personalized_template",
            value=(
                "Hola {nombre},\n\n"
                "Te escribo de parte de ALSUM para compartirte una comunicacion dirigida a {cargo} de {empresa}.\n\n"
                "Aqui puedes redactar el mensaje exactamente como quieras, manteniendo placeholders para personalizarlo en masa.\n\n"
                "Quedo atenta a tus comentarios.\n\n"
                "{remitente}"
            ),
        )
        add_signature = st.checkbox(
            "Agregar firma automatica si no esta incluida",
            value=False,
            key="personalized_signature",
        )
        custom_sender_name = st.text_input(
            "Remitente para plantilla personalizada",
            value="Lina Marcela Contreras",
            key="custom_sender_name",
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with preview_col:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="mini-label">Tokens disponibles</div>', unsafe_allow_html=True)
        st.markdown('<span class="token-chip">{nombre}</span>', unsafe_allow_html=True)
        st.markdown('<span class="token-chip">{nombre_completo}</span>', unsafe_allow_html=True)
        st.markdown('<span class="token-chip">{empresa}</span>', unsafe_allow_html=True)
        st.markdown('<span class="token-chip">{cargo}</span>', unsafe_allow_html=True)
        st.markdown('<span class="token-chip">{correo}</span>', unsafe_allow_html=True)
        st.markdown('<span class="token-chip">{celular}</span>', unsafe_allow_html=True)
        st.markdown('<span class="token-chip">{remitente}</span>', unsafe_allow_html=True)
        if not filtered_df.empty:
            preview_contact_personalized = st.selectbox(
                "Vista previa con",
                options=filtered_df["preview_label"].tolist(),
                index=0,
                key="preview_contact_personalized",
            )
        else:
            preview_contact_personalized = None
            st.warning("No hay contactos en el filtro actual.")
        st.markdown('</div>', unsafe_allow_html=True)

    if personalized_template.strip() and not filtered_df.empty:
        custom_df = filtered_df.copy()
        custom_df["Mensaje personalizado"] = custom_df.apply(
            lambda row: build_custom_message(row, personalized_template, custom_sender_name, add_signature),
            axis=1,
        )
        custom_df["WhatsApp URL personalizado"] = custom_df.apply(
            lambda row: build_whatsapp_link(row["Celular WhatsApp"], row["Mensaje personalizado"])
            if row["status_kind_final"] != "error"
            else "",
            axis=1,
        )

        preview_row = custom_df.loc[custom_df["preview_label"] == preview_contact_personalized].iloc[0]

        left_preview, right_preview = st.columns([1.05, 0.95], gap="large")
        with left_preview:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown('<div class="mini-label">Vista previa personalizada</div>', unsafe_allow_html=True)
            st.subheader(preview_row["Nombre"])
            st.caption(f"{preview_row['Cargo']} | {preview_row['Empresa'] or 'Sin empresa'}")
            st.text_area(
                "Mensaje personalizado final",
                value=preview_row["Mensaje personalizado"],
                height=300,
                disabled=True,
                label_visibility="collapsed",
            )
            st.markdown('</div>', unsafe_allow_html=True)

        with right_preview:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown('<div class="mini-label">Lote listo para enviar</div>', unsafe_allow_html=True)
            prepared_df = prepare_export(custom_df, "Mensaje personalizado", "WhatsApp URL personalizado")
            st.dataframe(
                prepared_df[["Cargo", "Empresa", "Nombre", "Celular listo", "Estado final"]],
                use_container_width=True,
                hide_index=True,
            )
            st.download_button(
                "Descargar lote personalizado",
                data=prepared_df.to_csv(index=False).encode("utf-8-sig"),
                file_name="whatsapp_personalizado_preparado.csv",
                mime="text/csv",
                use_container_width=True,
            )
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("### Lista operativa personalizada")
        st.caption("Aqui puedes disparar el mensaje libre sobre todo el filtro activo o descargar el lote completo preparado.")
        render_contact_list(
            custom_df,
            "Mensaje personalizado",
            "WhatsApp URL personalizado",
            "No hay contactos para mostrar con los filtros actuales.",
        )
    elif filtered_df.empty:
        st.warning("No hay contactos para preparar mensajes personalizados con el filtro actual.")
    else:
        st.info("Escribe o ajusta la plantilla libre para preparar el mensaje personalizado sobre el filtro activo.")


with tab_email:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="mini-label">Base email-ready</div>', unsafe_allow_html=True)
    st.subheader("Correo masivo y proximos canales")

    if has_email_column:
        st.success("La pagina ya detecto una columna de correo en el archivo y puede preparar la base para campanas por email.")
    else:
        st.warning(
            "Aun no existe columna de correo en el archivo, pero la vista ya quedo lista. "
            "Cuando agregues una columna llamada Correo, Correo electronico o Email, se incorporara automaticamente."
        )

    subject_template = st.text_input(
        "Asunto sugerido",
        value="ALSUM | Informacion para {cargo} de {empresa}",
        key="email_subject",
    )
    email_sender_name = st.text_input(
        "Nombre remitente para email",
        value="Lina Marcela Contreras",
        key="email_sender_name",
    )
    email_template = st.text_area(
        "Cuerpo del correo",
        height=220,
        key="email_template",
        value=(
            "Apreciado/a {nombre_completo},\n\n"
            "Reciba un cordial saludo. Compartimos informacion dirigida a {cargo} de {empresa}.\n\n"
            "Este bloque queda listo para futuras campanas promocionales, informacion institucional y acciones masivas con SendGrid.\n\n"
            "Cordialmente,\n{remitente}"
        ),
    )

    if not filtered_df.empty:
        email_df = filtered_df.copy()
        email_df["Asunto sugerido"] = email_df.apply(
            lambda row: build_email_subject(row, subject_template, email_sender_name),
            axis=1,
        )
        email_df["Cuerpo correo"] = email_df.apply(
            lambda row: build_email_body(row, email_template, email_sender_name),
            axis=1,
        )
        email_preview_contact = st.selectbox(
            "Vista previa con",
            options=email_df["preview_label"].tolist(),
            index=0,
            key="preview_contact_email",
        )
        email_preview_row = email_df.loc[email_df["preview_label"] == email_preview_contact].iloc[0]

        preview_left, preview_right = st.columns([1.05, 0.95], gap="large")
        with preview_left:
            st.markdown("#### Vista previa de correo")
            st.write(f"**Asunto:** {email_preview_row['Asunto sugerido']}")
            st.text_area(
                "Cuerpo del correo listo",
                value=email_preview_row["Cuerpo correo"],
                height=260,
                disabled=True,
                label_visibility="collapsed",
            )

        with preview_right:
            email_export = email_df[
                ["Cargo", "Empresa", "Nombre", "Correo", "Tiene correo valido", "Asunto sugerido", "Cuerpo correo"]
            ].rename(columns={"Tiene correo valido": "Correo valido"})
            st.dataframe(
                email_export[["Cargo", "Empresa", "Nombre", "Correo", "Correo valido"]],
                use_container_width=True,
                hide_index=True,
            )
            st.download_button(
                "Descargar base email preparada",
                data=email_export.to_csv(index=False).encode("utf-8-sig"),
                file_name="email_outreach_preparado.csv",
                mime="text/csv",
                use_container_width=True,
            )
    else:
        st.info("No hay contactos seleccionados para preparar una base de correo.")

    st.markdown(
        """
        <div class="future-note">
            <strong>Siguiente fase lista:</strong><br>
            1. Cuando el archivo incluya correo, esta misma vista podra exportar la base lista para SendGrid.<br>
            2. El mismo filtro por cargo y busqueda ya sirve como capa de segmentacion para campanas futuras.<br>
            3. La pagina queda preparada conceptualmente para sumar WhatsApp API mas adelante usando la misma base procesada.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)