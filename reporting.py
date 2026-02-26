import os
import tempfile
from typing import Dict, List
import re  # <-- NUEVO

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from openai import OpenAI

import utils

# ✅ NUEVO (render robusto sin Kaleido/Chromium)
import matplotlib
matplotlib.use("Agg")  # backend headless para servidores
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

MODEL = "gpt-4o-mini"


# -----------------------
# Helpers de formateo PNG
# -----------------------
def _fmt_usd(v: float) -> str:
    try:
        return f"{float(v):,.0f}"
    except Exception:
        return "0"


def _fmt_pct(v: float) -> str:
    try:
        return f"{float(v):.1f}%"
    except Exception:
        return "0.0%"


def _apply_exec_style(ax):
    ax.grid(True, axis="y", color="#e2e8f0", linewidth=1, alpha=1)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


def _annotate_bars(ax, bars, *, fmt="usd", y_pad_ratio=0.01, fontsize=10):
    # Etiquetas SIEMPRE visibles (valores sobre cada barra)
    ymax = 0.0
    for b in bars:
        try:
            ymax = max(ymax, float(b.get_height()))
        except Exception:
            pass
    pad = max(1.0, ymax * y_pad_ratio)

    for b in bars:
        h = float(b.get_height())
        label = _fmt_pct(h) if fmt == "pct" else _fmt_usd(h)
        ax.text(
            b.get_x() + b.get_width() / 2,
            h + pad,
            label,
            ha="center",
            va="bottom",
            fontsize=fontsize,
            color="#0f172a",
            clip_on=False,
        )


def _save_png_primas_vs_siniestros_ramo(ramo_stats: pd.DataFrame, filename: str, title: str) -> None:
    d = ramo_stats.copy()
    if d.empty:
        fig = plt.figure(figsize=(14, 7), dpi=200)
        plt.title(title)
        plt.text(0.5, 0.5, "Sin datos", ha="center", va="center")
        plt.axis("off")
        fig.savefig(filename, bbox_inches="tight")
        plt.close(fig)
        return

    d = d.sort_values("Primas", ascending=False).head(15)
    x = list(d["Ramo"].astype(str).values)

    fig, ax = plt.subplots(figsize=(15, 7.5), dpi=220)
    _apply_exec_style(ax)

    idx = range(len(x))
    width = 0.38

    bars1 = ax.bar([i - width / 2 for i in idx], d["Primas"].values, width=width, color="#004A8F", label="Primas (USD)")
    bars2 = ax.bar([i + width / 2 for i in idx], d["Siniestros"].values, width=width, color="#DC2626", label="Siniestros (USD)")

    ax.set_title(title, loc="left", fontsize=18, fontweight="bold", color="#0f172a")
    ax.set_xlabel("Ramo", fontsize=12, color="#0f172a")
    ax.set_ylabel("USD", fontsize=12, color="#0f172a")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))

    ax.set_xticks(list(idx))
    ax.set_xticklabels(x, rotation=35, ha="right", fontsize=10)

    _annotate_bars(ax, bars1, fmt="usd", y_pad_ratio=0.012, fontsize=10)
    _annotate_bars(ax, bars2, fmt="usd", y_pad_ratio=0.012, fontsize=10)

    # Deja espacio para etiquetas
    y_max = float(max(d["Primas"].max(), d["Siniestros"].max()))
    ax.set_ylim(0, (y_max * 1.35) if y_max > 0 else 1)

    ax.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#cbd5e1")
    fig.tight_layout()
    fig.savefig(filename, bbox_inches="tight")
    plt.close(fig)


def _save_png_siniestralidad_ramo(ramo_stats: pd.DataFrame, filename: str, title: str) -> None:
    d = ramo_stats.copy()
    if d.empty:
        fig = plt.figure(figsize=(14, 7), dpi=200)
        plt.title(title)
        plt.text(0.5, 0.5, "Sin datos", ha="center", va="center")
        plt.axis("off")
        fig.savefig(filename, bbox_inches="tight")
        plt.close(fig)
        return

    d = d.sort_values("Primas", ascending=False).head(15)
    x = list(d["Ramo"].astype(str).values)

    def color_ratio(r):
        if r < 50:
            return "#10B981"
        if r < 75:
            return "#F59E0B"
        return "#EF4444"

    colors = [color_ratio(float(v)) for v in d["Siniestralidad"].values]

    fig, ax = plt.subplots(figsize=(15, 7.2), dpi=220)
    _apply_exec_style(ax)

    bars = ax.bar(x, d["Siniestralidad"].values, color=colors, label="Siniestralidad (%)")
    ax.set_title(title, loc="left", fontsize=18, fontweight="bold", color="#0f172a")
    ax.set_xlabel("Ramo", fontsize=12, color="#0f172a")
    ax.set_ylabel("%", fontsize=12, color="#0f172a")
    ax.set_xticklabels(x, rotation=35, ha="right", fontsize=10)

    _annotate_bars(ax, bars, fmt="pct", y_pad_ratio=0.02, fontsize=10)

    y_max = float(d["Siniestralidad"].max())
    ax.set_ylim(0, max(10.0, y_max * 1.35))

    ax.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#cbd5e1")
    fig.tight_layout()
    fig.savefig(filename, bbox_inches="tight")
    plt.close(fig)


def _save_png_evolucion_foco(foco_year_stats: pd.DataFrame, filename: str, title: str) -> None:
    dfx = foco_year_stats.copy()
    if dfx.empty:
        fig = plt.figure(figsize=(14, 7), dpi=200)
        plt.title(title)
        plt.text(0.5, 0.5, "Sin datos", ha="center", va="center")
        plt.axis("off")
        fig.savefig(filename, bbox_inches="tight")
        plt.close(fig)
        return

    order = ["Carga", "Cascos", "RC"]
    years = sorted(dfx["Año"].astype(int).unique().tolist())
    years_lbl = [str(y) for y in years]

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(15, 12.8), dpi=220, sharex=True)
    fig.suptitle(title, x=0.01, ha="left", fontsize=20, fontweight="bold", color="#0f172a")

    for ax, ramo in zip(axes, order):
        sub = dfx[dfx["RamoFoco"] == ramo].copy()
        # Asegura orden por año
        sub["Año"] = sub["Año"].astype(int)
        sub = sub.sort_values("Año")

        # Si faltan años, rellena 0
        sub = pd.DataFrame({"Año": years}).merge(sub, on="Año", how="left").fillna(0)

        idx = range(len(years))
        width = 0.38

        _apply_exec_style(ax)
        ax.set_title(f"{ramo} | Primas vs Siniestros + Siniestralidad", loc="left", fontsize=14, fontweight="bold", color="#0f172a")

        bars1 = ax.bar([i - width / 2 for i in idx], sub["Primas"].values, width=width, color="#004A8F", label="Primas (USD)")
        bars2 = ax.bar([i + width / 2 for i in idx], sub["Siniestros"].values, width=width, color="#DC2626", label="Siniestros (USD)")

        ax.set_ylabel("USD", fontsize=11)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))

        # Etiquetas barras
        _annotate_bars(ax, bars1, fmt="usd", y_pad_ratio=0.012, fontsize=9)
        _annotate_bars(ax, bars2, fmt="usd", y_pad_ratio=0.012, fontsize=9)

        # Línea siniestralidad en segundo eje
        ax2 = ax.twinx()
        ax2.plot(list(idx), sub["Siniestralidad"].values, color="#111827", marker="o", linewidth=2, label="Siniestralidad (%)")
        ax2.set_ylabel("%", fontsize=11)
        ax2.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.1f}%"))

        # Etiquetas % en cada punto
        for i, v in enumerate(sub["Siniestralidad"].values):
            ax2.text(i, float(v) + 0.5, _fmt_pct(v), ha="center", va="bottom", fontsize=9, color="#111827", clip_on=False)

        # Rangos para que no se recorte nada
        y_max = float(max(sub["Primas"].max(), sub["Siniestros"].max()))
        ax.set_ylim(0, (y_max * 1.40) if y_max > 0 else 1)
        pct_max = float(sub["Siniestralidad"].max()) if len(sub) else 0.0
        ax2.set_ylim(0, max(10.0, pct_max * 1.35))

        # Leyenda combinada (solo una, arriba)
        if ramo == "Carga":
            h1, l1 = ax.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            fig.legend(h1 + h2, l1 + l2, loc="upper left", bbox_to_anchor=(0.01, 0.965), ncol=3,
                       frameon=True, facecolor="white", edgecolor="#cbd5e1")

    axes[-1].set_xticks(list(range(len(years_lbl))))
    axes[-1].set_xticklabels(years_lbl, fontsize=11)
    axes[-1].set_xlabel("Año", fontsize=12)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(filename, bbox_inches="tight")
    plt.close(fig)


def build_ramo_stats(df_pais: pd.DataFrame) -> pd.DataFrame:
    if df_pais.empty:
        return pd.DataFrame(columns=["Ramo", "Primas", "Siniestros", "No Reporta", "Siniestralidad"])

    g = df_pais.groupby(["Ramo", "Tipo"])["USD"].sum().unstack(fill_value=0).reset_index()
    cols = g.columns.tolist()

    g["Primas"] = g.apply(lambda r: _kw_sum_row(r, cols, "Prima"), axis=1)
    g["Siniestros"] = g.apply(lambda r: _kw_sum_row(r, cols, "Siniestro"), axis=1)
    g["No Reporta"] = g.apply(lambda r: _kw_sum_row(r, cols, "No Reporta"), axis=1)
    g["Siniestralidad"] = (g["Siniestros"] / g["Primas"] * 100).fillna(0)

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
    if ratio < 50:
        return "#10B981"
    if ratio < 75:
        return "#F59E0B"
    return "#EF4444"


def _layout_exec(title: str) -> dict:
    return dict(
        title=dict(text=title, x=0.01, xanchor="left", font=dict(size=22, color="#0f172a")),
        template="plotly_white",
        font=dict(family="DejaVu Sans, Arial", size=14, color="#0f172a"),
        margin=dict(l=60, r=40, t=130, b=105),
        legend=dict(
            orientation="h",
            x=0.0,
            y=0.995,
            xanchor="left",
            yanchor="top",
            bgcolor="rgba(255,255,255,0.90)",
            bordercolor="rgba(148,163,184,0.6)",
            borderwidth=1,
            title_text="",
            font=dict(size=13, color="#0f172a"),
        ),
    )


def _add_bar_value_annotations(fig: go.Figure, x_vals, y_vals, *, fmt: str, row=None, col=None, yshift: int = 10):
    """
    Etiquetas garantizadas en export PNG (Kaleido).
    fmt: "usd" o "pct"
    """
    for x, y in zip(list(x_vals), list(y_vals)):
        try:
            y_num = float(y)
        except Exception:
            continue

        if fmt == "pct":
            label = f"{y_num:.1f}%"
        else:
            label = f"{y_num:,.0f}"

        fig.add_annotation(
            x=x,
            y=y_num,
            text=label,
            showarrow=False,
            yshift=yshift,
            font=dict(size=12, color="#0f172a"),
            row=row,
            col=col,
        )


def make_fig_primas_vs_siniestros_ramo(ramo_stats: pd.DataFrame, title: str) -> go.Figure:
    if ramo_stats.empty:
        return go.Figure(layout=_layout_exec(title))

    d = ramo_stats.copy().sort_values("Primas", ascending=False).head(15)
    y_max = float(max(d["Primas"].max(), d["Siniestros"].max()))
    y_range_top = y_max * 1.30 if y_max > 0 else 1

    fig = go.Figure()
    fig.add_trace(go.Bar(x=d["Ramo"], y=d["Primas"], name="Primas (USD)", marker_color="#004A8F"))
    fig.add_trace(go.Bar(x=d["Ramo"], y=d["Siniestros"], name="Siniestros (USD)", marker_color="#DC2626"))

    # Etiquetas “a prueba de export”
    _add_bar_value_annotations(fig, d["Ramo"], d["Primas"], fmt="usd", yshift=12)
    _add_bar_value_annotations(fig, d["Ramo"], d["Siniestros"], fmt="usd", yshift=28)

    fig.update_layout(
        **_layout_exec(title),
        barmode="group",
        bargap=0.22,
        xaxis=dict(title="Ramo", tickangle=-35, automargin=True),
        yaxis=dict(
            title="USD",
            tickformat=",.0f",
            range=[0, y_range_top],
            gridcolor="rgba(226,232,240,0.9)",
            zerolinecolor="rgba(148,163,184,0.9)",
            automargin=True,
        ),
        height=650,
    )
    return fig


def make_fig_siniestralidad_ramo(ramo_stats: pd.DataFrame, title: str) -> go.Figure:
    if ramo_stats.empty:
        return go.Figure(layout=_layout_exec(title))

    d = ramo_stats.copy().sort_values("Primas", ascending=False).head(15)
    d["Color"] = d["Siniestralidad"].apply(_get_color_siniestralidad)

    y_max = float(d["Siniestralidad"].max()) if not d.empty else 0.0
    y_range_top = max(10.0, y_max * 1.35)

    fig = go.Figure()
    fig.add_trace(go.Bar(x=d["Ramo"], y=d["Siniestralidad"], name="Siniestralidad (%)", marker_color=d["Color"]))

    # Etiquetas “a prueba de export”
    _add_bar_value_annotations(fig, d["Ramo"], d["Siniestralidad"], fmt="pct", yshift=12)

    fig.update_layout(
        **_layout_exec(title),
        xaxis=dict(title="Ramo", tickangle=-35, automargin=True),
        yaxis=dict(
            title="%",
            tickformat=".1f",
            range=[0, y_range_top],
            gridcolor="rgba(226,232,240,0.9)",
            zerolinecolor="rgba(148,163,184,0.9)",
            automargin=True,
        ),
        height=650,
    )
    return fig


def make_fig_evolucion_foco(foco_year_stats: pd.DataFrame, title: str) -> go.Figure:
    if foco_year_stats.empty:
        return go.Figure(layout=_layout_exec(title))

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.09,
        specs=[[{"secondary_y": True}], [{"secondary_y": True}], [{"secondary_y": True}]],
        subplot_titles=(
            "Carga | Primas vs Siniestros + Siniestralidad",
            "Cascos | Primas vs Siniestros + Siniestralidad",
            "RC | Primas vs Siniestros + Siniestralidad",
        ),
    )

    order = ["Carga", "Cascos", "RC"]
    for i, ramo in enumerate(order, start=1):
        d = foco_year_stats[foco_year_stats["RamoFoco"] == ramo].copy()
        x = d["Año"].astype(int).astype(str)

        y_max = float(max(d["Primas"].max(), d["Siniestros"].max())) if not d.empty else 0.0
        y_range_top = y_max * 1.35 if y_max > 0 else 1.0

        fig.add_trace(go.Bar(x=x, y=d["Primas"], name="Primas (USD)", marker_color="#004A8F", showlegend=(i == 1)), row=i, col=1, secondary_y=False)
        fig.add_trace(go.Bar(x=x, y=d["Siniestros"], name="Siniestros (USD)", marker_color="#DC2626", showlegend=(i == 1)), row=i, col=1, secondary_y=False)
        fig.add_trace(go.Scatter(x=x, y=d["Siniestralidad"], name="Siniestralidad (%)", mode="lines+markers", line=dict(color="#111827", width=2), marker=dict(size=8, color="#111827"), showlegend=(i == 1)), row=i, col=1, secondary_y=True)

        # Etiquetas “a prueba de export” por subplot
        _add_bar_value_annotations(fig, x, d["Primas"], fmt="usd", row=i, col=1, yshift=12)
        _add_bar_value_annotations(fig, x, d["Siniestros"], fmt="usd", row=i, col=1, yshift=28)
        _add_bar_value_annotations(fig, x, d["Siniestralidad"], fmt="pct", row=i, col=1, yshift=10)

        fig.update_yaxes(
            title_text="USD",
            tickformat=",.0f",
            range=[0, y_range_top],
            gridcolor="rgba(226,232,240,0.9)",
            zerolinecolor="rgba(148,163,184,0.9)",
            row=i,
            col=1,
            secondary_y=False,
        )
        fig.update_yaxes(
            title_text="%",
            tickformat=".1f",
            rangemode="tozero",
            gridcolor="rgba(226,232,240,0.6)",
            row=i,
            col=1,
            secondary_y=True,
        )

    fig.update_layout(
        **_layout_exec(title),
        barmode="group",
        bargap=0.25,
        height=1300,
    )
    fig.update_xaxes(title_text="Año", row=3, col=1)
    return fig


def save_plotly_figure(fig: go.Figure, filename: str) -> None:
    fig.write_image(filename, format="png", width=1400, height=800, scale=2)


def _df_to_context_table(df: pd.DataFrame, cols: List[str], max_rows: int) -> str:
    if df is None or df.empty:
        return "Sin datos."
    view = df[cols].head(max_rows).copy()
    return view.to_string(index=False)


def _call_ia_country(
    api_key: str,
    pais: str,
    years: List[int],
    totals: Dict[str, float],
    ramo_stats: pd.DataFrame,
    top_empresas: pd.DataFrame,
    instruccion: str,
) -> Dict[str, object]:
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
- (máximo 5 líneas ejecutivas)

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
            l = l.lstrip("-•* ").strip()
            if l:
                out.append(l)
        return out

    hallazgos = _bullets(hallazgos_blk)[:6]
    recs = _bullets(recs_blk)[:6]
    resumen = "\n".join([l.strip() for l in resumen_blk.splitlines() if l.strip()]) if resumen_blk else text.strip()

    return {"resumen": resumen, "hallazgos": hallazgos, "recomendaciones": recs}


# =========================
# EVOLUCIÓN 2022-2024 (FOCO)
# =========================
def _normalize_ramo(s: str) -> str:
    return str(s or "").strip().lower()


def _ramo_bucket(ramo: str) -> str | None:
    r = _normalize_ramo(ramo)
    if "carga" in r:
        return "Carga"
    if "casco" in r or "cascos" in r:
        return "Cascos"
    if r == "rc" or "responsabilidad" in r or "responsabilidad civil" in r or "civil" in r:
        return "RC"
    return None


def build_foco_year_stats(df_pais: pd.DataFrame, years_focus: List[int]) -> pd.DataFrame:
    if df_pais.empty:
        return pd.DataFrame(columns=["Año", "RamoFoco", "Primas", "Siniestros", "Siniestralidad"])

    d = df_pais.copy()
    d["Año"] = pd.to_numeric(d["Año"], errors="coerce")
    d = d.dropna(subset=["Año"])
    d["Año"] = d["Año"].astype(int)
    d = d[d["Año"].isin(years_focus)]

    d["RamoFoco"] = d["Ramo"].apply(_ramo_bucket)
    d = d.dropna(subset=["RamoFoco"])

    if d.empty:
        return pd.DataFrame(columns=["Año", "RamoFoco", "Primas", "Siniestros", "Siniestralidad"])

    piv = d.groupby(["Año", "RamoFoco", "Tipo"])["USD"].sum().unstack(fill_value=0).reset_index()
    cols = piv.columns.tolist()

    piv["Primas"] = piv.apply(lambda r: _kw_sum_row(r, cols, "Prima"), axis=1)
    piv["Siniestros"] = piv.apply(lambda r: _kw_sum_row(r, cols, "Siniestro"), axis=1)
    piv["Siniestralidad"] = (piv["Siniestros"] / piv["Primas"] * 100).fillna(0)

    out = piv[["Año", "RamoFoco", "Primas", "Siniestros", "Siniestralidad"]].copy()

    grid = pd.MultiIndex.from_product([years_focus, ["Carga", "Cascos", "RC"]], names=["Año", "RamoFoco"]).to_frame(index=False)
    out = grid.merge(out, on=["Año", "RamoFoco"], how="left").fillna(0)

    return out.sort_values(["RamoFoco", "Año"])


def generate_pdf_consolidado_por_pais(
    df_filtrado: pd.DataFrame,
    api_key: str,
    instruccion: str,
    report_title: str = "INFORME CONSOLIDADO POR PAÍS 2026",
    subtitle: str = "ALSUM INTELLIGENCE",
    top_ramos: int = 12,
    top_empresas: int = 10,
) -> bytes:
    required = {"País", "Ramo", "Compañía", "Tipo", "USD", "Año"}
    missing = [c for c in required if c not in df_filtrado.columns]
    if missing:
        raise ValueError(f"Faltan columnas requeridas para informe por país: {missing}")

    dfx = df_filtrado.copy()
    dfx["USD"] = _to_num(dfx["USD"])

    years = sorted(pd.to_numeric(dfx["Año"], errors="coerce").dropna().astype(int).unique().tolist())
    paises = sorted(dfx["País"].astype(str).unique().tolist())

    pdf = utils.UltimatePDF()
    pdf.cover_page(report_title, subtitle)

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
            "(3) opacidad por No Reporta (si aplica), (4) 2 prioridades estratégicas."
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

    # Años foco (los que tú pediste)
    years_focus = [y for y in [2022, 2023, 2024] if y in years]
    if not years_focus:
        years_focus = years  # fallback: usa los disponibles

    for pais in paises:
        df_pais = dfx[dfx["País"].astype(str) == str(pais)].copy()
        if df_pais.empty:
            continue

        totals = _compute_totals_from_long(df_pais)
        ramo_stats = build_ramo_stats(df_pais).head(top_ramos)
        top_emp = build_top_empresas(df_pais, top_n=top_empresas)
        ia = _call_ia_country(api_key, pais, years, totals, ramo_stats, top_emp, instruccion)

        pdf.add_page()
        pdf.section_title(f"País: {pais}")

        pdf.add_table(
            data=[
                ["KPI", "Valor"],
                ["Primas (USD)", f"{totals['Primas']:,.0f}"],
                ["Siniestros (USD)", f"{totals['Siniestros']:,.0f}"],
                ["Siniestralidad (%)", f"{totals['Siniestralidad']:.1f}%"],
                ["No Reporta (USD)", f"{totals['No Reporta']:,.0f}"],
            ],
            col_widths=[60, 120],
        )

        pdf.add_section("Resumen Ejecutivo del País", ia.get("resumen", ""))

        if ia.get("hallazgos"):
            pdf.key_findings(ia["hallazgos"])

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
            table_ramo = [["Ramo", "Primas", "Siniestros", "Siniestr. %"]]
            for _, r in ramo_stats.iterrows():
                table_ramo.append(
                    [
                        str(r["Ramo"]),
                        f"{float(r['Primas']):,.0f}",
                        f"{float(r['Siniestros']):,.0f}",
                        f"{float(r['Siniestralidad']):.1f}%",
                    ]
                )
            pdf.add_table(table_ramo, col_widths=[85, 35, 35, 25])

        # ===== GRÁFICAS (PNG robusto SIN Kaleido) =====
        tmp_files = []
        try:
            foco_year = build_foco_year_stats(df_pais, years_focus=years_focus)

            f1 = tempfile.NamedTemporaryFile(suffix=".png", delete=False); f1.close()
            f2 = tempfile.NamedTemporaryFile(suffix=".png", delete=False); f2.close()
            f3 = tempfile.NamedTemporaryFile(suffix=".png", delete=False); f3.close()

            tmp_files.extend([f1.name, f2.name, f3.name])

            _save_png_primas_vs_siniestros_ramo(
                ramo_stats,
                f1.name,
                title=f"{pais} | Primas vs Siniestros por Ramo (Top {min(top_ramos, len(ramo_stats))})",
            )
            _save_png_siniestralidad_ramo(
                ramo_stats,
                f2.name,
                title=f"{pais} | Siniestralidad (%) por Ramo (Top {min(top_ramos, len(ramo_stats))})",
            )
            _save_png_evolucion_foco(
                foco_year,
                f3.name,
                title=f"{pais} | Evolución {years_focus[0]}–{years_focus[-1]} (Carga / Cascos / RC)",
            )

            pdf.section_title("Visualizaciones Clave del País (Etiquetadas)")
            pdf.add_image_section("Primas vs Siniestros por Ramo", f1.name, w=180)
            pdf.add_image_section("Siniestralidad (%) por Ramo", f2.name, w=180)

            # Tabla + evolución foco
            if not foco_year.empty:
                pdf.section_title(f"Comparativo {years_focus[0]}–{years_focus[-1]} | Ramos Foco (Carga/Cascos/RC)")
                table_foco = [["Ramo Foco", "Año", "Primas", "Siniestros", "Siniestr. %"]]
                for _, r in foco_year.iterrows():
                    table_foco.append(
                        [
                            str(r["RamoFoco"]),
                            str(int(r["Año"])),
                            f"{float(r['Primas']):,.0f}",
                            f"{float(r['Siniestros']):,.0f}",
                            f"{float(r['Siniestralidad']):.1f}%",
                        ]
                    )
                pdf.add_table(table_foco, col_widths=[35, 20, 45, 45, 35])

            pdf.add_image_section("Evolución anual | Primas vs Siniestros + Siniestralidad", f3.name, w=180)

        finally:
            for p in tmp_files:
                try:
                    os.unlink(p)
                except Exception:
                    pass

        if ia.get("recomendaciones"):
            pdf.recommendations(ia["recomendaciones"])

    pdf.annex(
        "Metodología: consolidación por país a partir de Primas/Siniestros (USD) filtrados por Años, Ramos, Afiliación y Compañías. "
        "Las conclusiones de IA se generan sobre agregados (no micro-datos) y deben validarse contra contexto regulatorio/local."
    )

    return bytes(pdf.output(dest="S"))