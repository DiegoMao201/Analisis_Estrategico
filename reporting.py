import os
import tempfile
from typing import Dict, List, Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from openai import OpenAI

import utils


MODEL = "gpt-4o-mini"


def _to_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0)


def _kw_sum_row(row: pd.Series, cols: List[str], keyword: str) -> float:
    matches = [c for c in cols if keyword.lower() in str(c).lower()]
    if not matches:
        return 0.0
    return float(row[matches].sum())


def _compute_totals_from_long(df: pd.DataFrame) -> Dict[str, float]:
    # Asume esquema de tu app: Tipo + USD
    if df.empty:
        return {"Primas": 0.0, "Siniestros": 0.0, "No Reporta": 0.0, "Siniestralidad": 0.0}

    d = df.copy()
    d["USD"] = _to_num(d.get("USD", 0))

    primas = d[d["Tipo"].astype(str).str.contains("Prima", case=False, na=False)]["USD"].sum()
    siniestros = d[d["Tipo"].astype(str).str.contains("Siniestro", case=False, na=False)]["USD"].sum()
    noreporta = d[d["Tipo"].astype(str).str.contains("No Reporta", case=False, na=False)]["USD"].sum()
    siniestralidad = (siniestros / primas * 100) if primas > 0 else 0.0

    return {
        "Primas": float(primas),
        "Siniestros": float(siniestros),
        "No Reporta": float(noreporta),
        "Siniestralidad": float(siniestralidad),
    }


def build_ramo_stats(df_pais: pd.DataFrame) -> pd.DataFrame:
    # Estadísticas por ramo usando tu lógica de pivot por Tipo
    if df_pais.empty:
        return pd.DataFrame(columns=["Ramo", "Primas", "Siniestros", "No Reporta", "Siniestralidad"])

    g = df_pais.groupby(["Ramo", "Tipo"])["USD"].sum().unstack(fill_value=0).reset_index()
    cols = g.columns.tolist()

    g["Primas"] = g.apply(lambda r: _kw_sum_row(r, cols, "Prima"), axis=1)
    g["Siniestros"] = g.apply(lambda r: _kw_sum_row(r, cols, "Siniestro"), axis=1)
    g["No Reporta"] = g.apply(lambda r: _kw_sum_row(r, cols, "No Reporta"), axis=1)
    g["Siniestralidad"] = (g["Siniestros"] / g["Primas"] * 100).replace([pd.NA, pd.NaT], 0).fillna(0)

    g = g[["Ramo", "Primas", "Siniestros", "No Reporta", "Siniestralidad"]]
    g = g.sort_values("Primas", ascending=False)
    return g


def build_top_empresas(df_pais: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    if df_pais.empty:
        return pd.DataFrame(columns=["Compañía", "Primas", "Siniestros", "Siniestralidad"])

    d = df_pais.copy()
    d["USD"] = _to_num(d["USD"])

    primas = (
        d[d["Tipo"].astype(str).str.contains("Prima", case=False, na=False)]
        .groupby("Compañía")["USD"]
        .sum()
        .reset_index()
        .rename(columns={"USD": "Primas"})
    )

    siniestros = (
        d[d["Tipo"].astype(str).str.contains("Siniestro", case=False, na=False)]
        .groupby("Compañía")["USD"]
        .sum()
        .reset_index()
        .rename(columns={"USD": "Siniestros"})
    )

    m = primas.merge(siniestros, on="Compañía", how="left").fillna({"Siniestros": 0.0})
    m["Siniestralidad"] = (m["Siniestros"] / m["Primas"] * 100).fillna(0)

    m = m.sort_values("Primas", ascending=False).head(top_n)
    return m


def _get_color_siniestralidad(ratio: float) -> str:
    # Semáforo similar a tu dashboard
    if ratio < 50:
        return "#10B981"
    if ratio < 75:
        return "#F59E0B"
    return "#EF4444"


def make_fig_primas_ramo(ramo_stats: pd.DataFrame, title: str) -> go.Figure:
    d = ramo_stats.copy()
    d["Color"] = d["Siniestralidad"].apply(_get_color_siniestralidad)

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=d["Ramo"],
            y=d["Primas"],
            name="Primas",
            marker_color=d["Color"],
            text=d["Siniestralidad"].apply(lambda x: f"{x:.1f}%"),
            textposition="auto",
        )
    )
    fig.add_trace(
        go.Bar(
            x=d["Ramo"],
            y=d["No Reporta"],
            name="No Reporta",
            marker_color="#94A3B8",
        )
    )
    fig.update_layout(
        title=title,
        barmode="stack",
        xaxis_tickangle=-35,
        template="plotly_white",
        legend=dict(orientation="h", y=1.12),
        margin=dict(l=10, r=10, t=60, b=10),
        height=520,
    )
    return fig


def make_fig_siniestralidad_ramo(ramo_stats: pd.DataFrame, title: str) -> go.Figure:
    d = ramo_stats.copy()
    d["Color"] = d["Siniestralidad"].apply(_get_color_siniestralidad)

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=d["Ramo"],
            y=d["Siniestralidad"],
            name="Siniestralidad (%)",
            marker_color=d["Color"],
            text=d["Siniestralidad"].apply(lambda x: f"{x:.1f}%"),
            textposition="auto",
        )
    )
    fig.update_layout(
        title=title,
        template="plotly_white",
        xaxis_tickangle=-35,
        yaxis_title="%",
        margin=dict(l=10, r=10, t=60, b=10),
        height=480,
    )
    return fig


def save_plotly_figure(fig: go.Figure, filename: str) -> None:
    # Kaleido
    fig.write_image(filename, format="png", width=1100, height=650)


def _df_to_context_table(df: pd.DataFrame, cols: List[str], max_rows: int) -> str:
    if df is None or df.empty:
        return "Sin datos."
    view = df[cols].head(max_rows).copy()
    return view.to_string(index=False)


def _call_ia_country(api_key: str, pais: str, years: List[int], totals: Dict[str, float], ramo_stats: pd.DataFrame, top_empresas: pd.DataFrame, instruccion: str) -> Dict[str, object]:
    """
    Retorna dict con:
      - resumen (str)
      - hallazgos (List[str])
      - recomendaciones (List[str])
    """
    if not api_key:
        return {
            "resumen": "⚠️ API Key no configurada. Se omite análisis IA por país.",
            "hallazgos": [],
            "recomendaciones": [],
        }

    ramo_ctx = _df_to_context_table(
        ramo_stats,
        cols=["Ramo", "Primas", "Siniestros", "Siniestralidad", "No Reporta"],
        max_rows=12,
    )
    emp_ctx = _df_to_context_table(
        top_empresas,
        cols=["Compañía", "Primas", "Siniestros", "Siniestralidad"],
        max_rows=10,
    )

    prompt = f"""
Actúa como Analista Senior de Seguros (LatAm) y consultor ejecutivo de ALSUM.
Necesito un informe CONSOLIDADO por país (no por empresa individual; solo top empresas para referencia).

PAÍS: {pais}
AÑOS INCLUIDOS: {", ".join(map(str, years)) if years else "No especificado"}

TOTALES PAÍS:
- Primas: {totals["Primas"]:,.0f}
- Siniestros: {totals["Siniestros"]:,.0f}
- No Reporta: {totals["No Reporta"]:,.0f}
- Siniestralidad: {totals["Siniestralidad"]:.1f}%

RESUMEN POR RAMO (TOP):
{ramo_ctx}

TOP EMPRESAS (referencia, NO extenderse):
{emp_ctx}

INSTRUCCIÓN DEL USUARIO:
{instruccion}

FORMATO DE RESPUESTA OBLIGATORIO (no agregues otras secciones):
RESUMEN:
- (máximo 5 líneas ejecutivas, sin tecnicismos innecesarios)

HALLAZGOS:
- (3 bullets: 1 oportunidad, 1 riesgo/anomalía, 1 tendencia)

RECOMENDACIONES:
- (3 bullets accionables, priorizadas por impacto)
""".strip()

    client = OpenAI(api_key=api_key)
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "Eres un analista senior de seguros, preciso, ejecutivo y accionable."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.6,
        )
        text = resp.choices[0].message.content or ""
    except Exception as e:
        return {"resumen": f"Error IA: {e}", "hallazgos": [], "recomendaciones": []}

    # Parse simple por encabezados
    def _extract_block(label: str, next_labels: List[str]) -> str:
        t = text
        idx = t.upper().find(label)
        if idx < 0:
            return ""
        start = idx + len(label)
        end = len(t)
        for nl in next_labels:
            j = t.upper().find(nl, start)
            if j >= 0:
                end = min(end, j)
        return t[start:end].strip()

    resumen_blk = _extract_block("RESUMEN:", ["HALLAZGOS:", "RECOMENDACIONES:"])
    hallazgos_blk = _extract_block("HALLAZGOS:", ["RECOMENDACIONES:"])
    recs_blk = _extract_block("RECOMENDACIONES:", [])

    def _bullets(block: str) -> List[str]:
        lines = [l.strip() for l in (block or "").splitlines()]
        out = []
        for l in lines:
            if not l:
                continue
            # Normaliza bullets comunes
            l = l.lstrip("-•* ").strip()
            if l:
                out.append(l)
        return out

    hallazgos = _bullets(hallazgos_blk)[:6]
    recs = _bullets(recs_blk)[:6]

    # resumen: devolver texto corrido pero limpio
    resumen = "\n".join([l.strip() for l in resumen_blk.splitlines() if l.strip()]) if resumen_blk else text.strip()

    return {"resumen": resumen, "hallazgos": hallazgos, "recomendaciones": recs}


def generate_pdf_consolidado_por_pais(
    df_filtrado: pd.DataFrame,
    api_key: str,
    instruccion: str,
    report_title: str = "INFORME CONSOLIDADO POR PAÍS 2026",
    subtitle: str = "ALSUM INTELLIGENCE",
    top_ramos: int = 12,
    top_empresas: int = 10,
) -> bytes:
    # Validaciones mínimas según tu esquema
    required = {"País", "Ramo", "Compañía", "Tipo", "USD", "Año"}
    missing = [c for c in required if c not in df_filtrado.columns]
    if missing:
        raise ValueError(f"Faltan columnas requeridas para informe por país: {missing}")

    dfx = df_filtrado.copy()
    dfx["USD"] = _to_num(dfx["USD"])

    years = sorted(pd.to_numeric(dfx["Año"], errors="coerce").dropna().astype(int).unique().tolist())
    paises = sorted(dfx["País"].astype(str).unique().tolist())

    # PDF base (mantiene tu formato profesional)
    pdf = utils.UltimatePDF()
    pdf.cover_page(report_title, subtitle)

    # Resumen global breve (sin cambiar tu estilo; usa tu capítulo)
    totals_global = _compute_totals_from_long(dfx)
    contexto_global = (
        f"Años: {years}. Países incluidos: {len(paises)}.\n"
        f"Primas: {totals_global['Primas']:,.0f} | "
        f"Siniestros: {totals_global['Siniestros']:,.0f} | "
        f"No Reporta: {totals_global['No Reporta']:,.0f} | "
        f"Siniestralidad: {totals_global['Siniestralidad']:.1f}%.\n"
        f"Instrucción: {instruccion}"
    )

    if api_key:
        client = OpenAI(api_key=api_key)
        prompt_global = (
            "Escribe un resumen ejecutivo global (máximo 8 líneas) del portafolio total, "
            "enfocado en: (1) concentración por países, (2) riesgo por siniestralidad, "
            "(3) opacidad por No Reporta, (4) 2 prioridades estratégicas."
        )
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": "Eres un analista senior de seguros, conciso y ejecutivo."},
                    {"role": "user", "content": f"{prompt_global}\n\nContexto:\n{contexto_global}"},
                ],
                temperature=0.6,
            )
            resumen_global = resp.choices[0].message.content or contexto_global
        except Exception:
            resumen_global = contexto_global
    else:
        resumen_global = contexto_global

    pdf.executive_summary(resumen_global)

    # Sección por país
    for pais in paises:
        df_pais = dfx[dfx["País"].astype(str) == str(pais)].copy()
        if df_pais.empty:
            continue

        totals = _compute_totals_from_long(df_pais)
        ramo_stats = build_ramo_stats(df_pais).head(top_ramos)
        top_emp = build_top_empresas(df_pais, top_n=top_empresas)

        # IA por país (resumen + hallazgos + recomendaciones)
        ia = _call_ia_country(api_key, pais, years, totals, ramo_stats, top_emp, instruccion)

        pdf.add_page()
        pdf.section_title(f"País: {pais}")

        # KPIs del país (tabla compacta)
        pdf.add_table(
            data=[
                ["KPI", "Valor"],
                ["Primas (USD)", f"{totals['Primas']:,.0f}"],
                ["Siniestros (USD)", f"{totals['Siniestros']:,.0f}"],
                ["No Reporta (USD)", f"{totals['No Reporta']:,.0f}"],
                ["Siniestralidad (%)", f"{totals['Siniestralidad']:.1f}%"],
            ],
            col_widths=[60, 120],
        )

        pdf.add_section("Resumen Ejecutivo del País", ia.get("resumen", ""))

        if ia.get("hallazgos"):
            pdf.key_findings(ia["hallazgos"])

        # Tablas resumen (top empresas + ramos)
        if not top_emp.empty:
            pdf.section_title(f"Top {min(top_empresas, len(top_emp))} Empresas (Referencia)")
            table_emp = [["Compañía", "Primas", "Siniestros", "Siniestr. %"]]
            for _, r in top_emp.iterrows():
                table_emp.append(
                    [
                        str(r["Compañía"]),
                        f"{float(r['Primas']):,.0f}",
                        f"{float(r['Siniestros']):,.0f}",
                        f"{float(r['Siniestralidad']):.1f}%",
                    ]
                )
            pdf.add_table(table_emp, col_widths=[90, 30, 30, 30])

        if not ramo_stats.empty:
            pdf.section_title(f"Ramos Principales (Top {min(top_ramos, len(ramo_stats))})")
            table_ramo = [["Ramo", "Primas", "Siniestr. %"]]
            for _, r in ramo_stats.iterrows():
                table_ramo.append([str(r["Ramo"]), f"{float(r['Primas']):,.0f}", f"{float(r['Siniestralidad']):.1f}%"])
            pdf.add_table(table_ramo, col_widths=[110, 40, 30])

        # Gráficas por país (guardadas a PNG temporal)
        tmp_files = []
        try:
            fig1 = make_fig_primas_ramo(ramo_stats, title=f"{pais} | Primas y No Reporta por Ramo (color=siniestralidad)")
            fig2 = make_fig_siniestralidad_ramo(ramo_stats, title=f"{pais} | Siniestralidad por Ramo")

            f1 = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            f2 = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            f1.close()
            f2.close()

            save_plotly_figure(fig1, f1.name)
            save_plotly_figure(fig2, f2.name)
            tmp_files.extend([f1.name, f2.name])

            pdf.section_title("Visualizaciones Clave del País")
            pdf.add_image_section("Primas/No Reporta por Ramo", f1.name, w=170)
            pdf.add_image_section("Siniestralidad por Ramo", f2.name, w=170)
        finally:
            for p in tmp_files:
                try:
                    os.unlink(p)
                except Exception:
                    pass

        if ia.get("recomendaciones"):
            pdf.recommendations(ia["recomendaciones"])

    # Cierre metodológico corto
    pdf.annex(
        "Metodología: consolidación por país a partir de Primas/Siniestros/No Reporta (USD) filtrados por Años, Ramos, Afiliación y Compañías. "
        "Las conclusiones de IA se generan sobre agregados (no micro-datos) y deben validarse contra contexto regulatorio/local."
    )

    return bytes(pdf.output(dest="S"))