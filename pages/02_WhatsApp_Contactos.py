import os
import re
from decimal import Decimal, InvalidOperation
from html import escape
from urllib.parse import quote

import pandas as pd
import streamlit as st

import utils


st.set_page_config(
    page_title="ALSUM 360 | WhatsApp Command",
    layout="wide",
    page_icon="📨",
    initial_sidebar_state="expanded",
)


PRIMARY = "#003366"
SECONDARY = "#0E5CAD"
ACCENT = "#16A34A"
SURFACE = "#F4F6F9"
INK = "#0F172A"
MUTED = "#475569"
WARN = "#F59E0B"
ERROR = "#DC2626"


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
        max-width: 920px;
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
        color: {INK};
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

    .stButton > button, div[data-testid="stLinkButton"] a {{
        border-radius: 10px !important;
        font-weight: 700 !important;
    }}

    .stTextArea textarea, .stTextInput input, .stSelectbox div[data-baseweb="select"] > div {{
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


def clean_contact_name(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).replace("\xa0", " ").strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"(?<=[A-Z0-9\)])(?=[A-Z][a-z])", " ", text)
    return text.strip(" -")


def split_contact_name(full_name: str) -> tuple[str, str, str]:
    tokens = full_name.split()
    person_start = None

    for index, token in enumerate(tokens):
        if re.search(r"[a-z]", token):
            person_start = index
            break

    if person_start is None:
        person = full_name.strip()
        organization = ""
    else:
        person = " ".join(tokens[person_start:]).strip(" -")
        organization = " ".join(tokens[:person_start]).strip(" -")

    if not person:
        person = full_name.strip()

    short_name = extract_short_name(person)
    return organization, person, short_name


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

    text = str(value).strip().replace("\xa0", " ")
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


def has_explicit_international_prefix(value: object) -> bool:
    if pd.isna(value):
        return False
    text = str(value).strip().replace("\xa0", " ")
    return text.startswith("+") or text.startswith("00")


def detect_phone_source_format(value: object) -> str:
    if pd.isna(value):
        return "Sin dato"
    text = str(value).strip().replace("\xa0", " ")
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
        return "Sin telefono", "error"
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


def build_message(row: pd.Series, custom_message: str, campaign_type: str, sender_name: str, closing_line: str) -> str:
    recipient_name = row["contact_person"]
    organization = row["organization"]
    short_name = row["short_name"]

    openings = {
        "Invitacion ejecutiva": (
            f"Apreciado/a {recipient_name},\n\n"
            "Reciba un cordial saludo. Desde ALSUM Analytics | Estrategia 360 queremos compartirle una comunicacion prioritaria dentro de nuestra agenda institucional y de relacionamiento ejecutivo."
        ),
        "Convocatoria comercial": (
            f"Apreciado/a {recipient_name},\n\n"
            "Espero que se encuentre muy bien. Le escribimos desde ALSUM Analytics | Estrategia 360 para acercarle una convocatoria con enfoque profesional y alto valor para nuestra red de aliados y decisores."
        ),
        "Seguimiento institucional": (
            f"Buen dia, {recipient_name}.\n\n"
            "Con un atento saludo, compartimos esta comunicacion desde ALSUM Analytics | Estrategia 360 como parte de nuestro seguimiento institucional y relacionamiento con actores clave del sector."
        ),
    }

    organization_line = ""
    if organization:
        organization_line = f"\n\nHemos considerado especialmente a {organization} dentro de esta difusion, por la relevancia de su participacion en este espacio."

    body = custom_message.strip()
    closing = closing_line.strip() or "Sera un gusto contar con su atencion y participacion."
    signature = sender_name.strip() or "Equipo ALSUM Analytics"

    return (
        f"{openings[campaign_type]}"
        f"{organization_line}\n\n"
        f"{body}\n\n"
        f"{closing}\n\n"
        f"Cordialmente,\n{signature}\n"
        f"ALSUM Analytics | Strategic Command"
    ).replace("{short_name}", short_name)


def build_whatsapp_link(phone: str, message: str) -> str:
    return f"https://wa.me/{phone}?text={quote(message)}"


@st.cache_data(show_spinner=False)
def load_contacts() -> tuple[pd.DataFrame | None, str | None, str | None]:
    contacts_path = get_contacts_path()
    if not contacts_path:
        return None, None, "No se encontro contactos_Alsum_4.xlsx ni contactos_Alsum_4.csv en la raiz del proyecto."

    try:
        df_contacts = normalize_columns(read_contacts_file(contacts_path))
    except Exception as exc:
        return None, contacts_path, f"No fue posible leer el archivo de contactos: {exc}"

    required_columns = ["Cargo", "Nombre", "Telefono"]
    missing_columns = [column for column in required_columns if column not in df_contacts.columns]
    if missing_columns:
        return None, contacts_path, f"Faltan columnas requeridas en la base de contactos: {', '.join(missing_columns)}"

    base = df_contacts[required_columns].copy()
    base["Cargo"] = base["Cargo"].fillna("Sin cargo").astype(str).str.strip()
    base["Nombre"] = base["Nombre"].map(clean_contact_name)
    base["Telefono original"] = base["Telefono"].astype(str)
    base["Formato telefono"] = base["Telefono"].map(detect_phone_source_format)
    base["Telefono limpio"] = base["Telefono"].map(normalize_phone)

    split_values = base["Nombre"].map(split_contact_name)
    base[["organization", "contact_person", "short_name"]] = pd.DataFrame(split_values.tolist(), index=base.index)

    statuses = base["Telefono limpio"].map(classify_phone)
    base[["Estado", "status_kind"]] = pd.DataFrame(statuses.tolist(), index=base.index)
    base["contact_person"] = base["contact_person"].replace("", "Contacto ALSUM")
    base["Orden"] = range(1, len(base) + 1)
    return base, contacts_path, None


st.markdown(
    """
    <div class="hero-card">
        <div class="hero-kicker">Outreach Layer</div>
        <div class="hero-title">Centro de Mensajeria WhatsApp</div>
        <div class="hero-copy">
            Esta pagina concentra la difusion uno a uno sobre tu base de contactos, con un mismo mensaje base,
            personalizacion por destinatario y salida directa a WhatsApp Web. El objetivo es ejecutar invitaciones,
            convocatorias o comunicaciones institucionales sin alterar el flujo existente de ALSUM.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.info(
    "Los enlaces de WhatsApp se generan siempre con el numero normalizado en formato internacional de solo digitos. "
    "Si el dato original trae +, 00, espacios, guiones o notacion cientifica, la pagina lo limpia automaticamente antes del envio."
)


with st.sidebar:
    st.header("Configuracion de envio")
    if st.button("Recargar base de contactos", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    default_prefix = st.text_input(
        "Indicativo por defecto",
        value="",
        placeholder="Ejemplo: 57",
        help="Opcional. Se antepone solo a numeros cortos que no parecen incluir codigo internacional.",
    )
    only_ready = st.toggle(
        "Mostrar solo listos para envio",
        value=False,
        help="Oculta contactos cuyo telefono requiere revision manual.",
    )


contacts_df, contacts_path, load_error = load_contacts()

if load_error:
    st.error(load_error)
    st.stop()


campaign_col, preview_col = st.columns([1.2, 0.8], gap="large")

with campaign_col:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="mini-label">Plantilla central</div>', unsafe_allow_html=True)
    st.subheader("Mensaje base para toda la lista")

    campaign_type = st.selectbox(
        "Tipo de comunicacion",
        options=["Invitacion ejecutiva", "Convocatoria comercial", "Seguimiento institucional"],
        index=0,
    )
    custom_message = st.text_area(
        "Contenido principal",
        height=220,
        placeholder=(
            "Ejemplo:\n"
            "Nos complaceria contar con su presencia en el Congreso ALSUM 2026, un espacio de networking, analisis sectorial y conversaciones ejecutivas para la region."
        ),
    )
    closing_line = st.text_input(
        "Cierre de mensaje",
        value="Sera un gusto contar con su participacion y confirmar su interes.",
    )
    sender_name = st.text_input(
        "Firma remitente",
        value="Diego Mauricio Garcia Rengifo | Coordinacion ALSUM",
    )
    st.markdown('</div>', unsafe_allow_html=True)

with preview_col:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="mini-label">Resumen operativo</div>', unsafe_allow_html=True)
    ready_count = int((contacts_df["status_kind"] == "ok").sum())
    review_count = int((contacts_df["status_kind"] == "review").sum())
    error_count = int((contacts_df["status_kind"] == "error").sum())

    metric_1, metric_2 = st.columns(2)
    metric_3, metric_4 = st.columns(2)
    metric_1.metric("Contactos", len(contacts_df))
    metric_2.metric("Listos", ready_count)
    metric_3.metric("Revisar", review_count)
    metric_4.metric("Sin telefono", error_count)

    st.caption(f"Fuente activa: {os.path.basename(contacts_path)}")

    preview_options = contacts_df["contact_person"].tolist()
    preview_contact = st.selectbox("Vista previa con", options=preview_options, index=0)
    st.markdown('</div>', unsafe_allow_html=True)


if not custom_message.strip():
    st.info("Escribe el contenido principal para generar el mensaje profesional y habilitar los enlaces de envio.")
    st.stop()


working_df = contacts_df.copy()
working_df["Telefono WhatsApp"] = working_df.apply(
    lambda row: apply_default_prefix(row["Telefono limpio"], default_prefix, row["Telefono original"]),
    axis=1,
)
working_df[["Estado final", "status_kind_final"]] = pd.DataFrame(
    working_df["Telefono WhatsApp"].map(classify_phone).tolist(),
    index=working_df.index,
)
working_df["Mensaje"] = working_df.apply(
    build_message,
    axis=1,
    custom_message=custom_message,
    campaign_type=campaign_type,
    sender_name=sender_name,
    closing_line=closing_line,
)
working_df["WhatsApp URL"] = working_df.apply(
    lambda row: build_whatsapp_link(row["Telefono WhatsApp"], row["Mensaje"]) if row["status_kind_final"] != "error" else "",
    axis=1,
)

filtered_df = working_df.copy()
if only_ready:
    filtered_df = filtered_df[filtered_df["status_kind_final"] == "ok"]

search_text = st.text_input(
    "Buscar contacto o empresa",
    value="",
    placeholder="Filtra por nombre, organizacion o telefono",
)
if search_text.strip():
    mask = (
        filtered_df["Nombre"].str.contains(search_text, case=False, na=False)
        | filtered_df["contact_person"].str.contains(search_text, case=False, na=False)
        | filtered_df["organization"].str.contains(search_text, case=False, na=False)
        | filtered_df["Telefono WhatsApp"].str.contains(search_text, case=False, na=False)
    )
    filtered_df = filtered_df[mask]


preview_row = working_df.loc[working_df["contact_person"] == preview_contact].iloc[0]

left_preview, right_preview = st.columns([1.1, 0.9], gap="large")
with left_preview:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="mini-label">Vista previa ejecutiva</div>', unsafe_allow_html=True)
    st.subheader(preview_row["contact_person"])
    if preview_row["organization"]:
        st.caption(preview_row["organization"])
    st.text_area(
        "Mensaje final",
        value=preview_row["Mensaje"],
        height=280,
        disabled=True,
        label_visibility="collapsed",
    )
    st.markdown('</div>', unsafe_allow_html=True)

with right_preview:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="mini-label">Base procesada</div>', unsafe_allow_html=True)
    download_df = filtered_df[[
        "Cargo",
        "Nombre",
        "organization",
        "contact_person",
        "Formato telefono",
        "Telefono original",
        "Telefono WhatsApp",
        "Estado final",
        "WhatsApp URL",
        "Mensaje",
    ]].rename(
        columns={
            "organization": "Organizacion",
            "contact_person": "Contacto",
        }
    )
    st.dataframe(
        download_df[["Cargo", "Contacto", "Organizacion", "Telefono WhatsApp", "Estado final"]],
        use_container_width=True,
        hide_index=True,
    )
    st.download_button(
        "Descargar base preparada",
        data=download_df.to_csv(index=False).encode("utf-8-sig"),
        file_name="whatsapp_contactos_preparados.csv",
        mime="text/csv",
        use_container_width=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)


st.markdown("### Lista operativa de contactos")
st.caption("Cada boton abre WhatsApp Web con el mensaje ya personalizado para ese destinatario.")

if filtered_df.empty:
    st.warning("No hay contactos para mostrar con los filtros actuales.")
    st.stop()


for row in filtered_df.to_dict("records"):
    status_class = {
        "ok": "status-ok",
        "review": "status-review",
        "error": "status-error",
    }.get(row["status_kind_final"], "status-review")

    col_info, col_message, col_action = st.columns([1.25, 2.2, 0.95], gap="medium")

    with col_info:
        contact_person = escape(row["contact_person"])
        cargo = escape(row["Cargo"])
        organization = escape(row["organization"] or "No identificada")
        source_format = escape(row["Formato telefono"])
        original_phone = escape(row["Telefono original"])
        whatsapp_phone = escape(row["Telefono WhatsApp"])
        final_status = escape(row["Estado final"])
        st.markdown(
            f"""
            <div class="contact-card">
                <div class="contact-name">{contact_person}</div>
                <div class="contact-meta">
                    <strong>Cargo:</strong> {cargo}<br>
                    <strong>Organizacion:</strong> {organization}<br>
                    <strong>Origen:</strong> {source_format}<br>
                    <strong>Telefono original:</strong> {original_phone}<br>
                    <strong>Telefono listo:</strong> {whatsapp_phone}<br>
                    <span class="{status_class}">{final_status}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_message:
        with st.expander(f"Ver mensaje para {row['short_name']}"):
            st.write(row["Mensaje"])

    with col_action:
        if row["status_kind_final"] == "error":
            st.warning("Telefono incompleto")
        else:
            st.link_button(
                "Abrir WhatsApp",
                row["WhatsApp URL"],
                use_container_width=True,
            )

    st.markdown("<div style='height: 0.55rem;'></div>", unsafe_allow_html=True)