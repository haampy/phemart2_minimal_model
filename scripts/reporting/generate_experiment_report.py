#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RunRecord:
    experiment: str
    seed: int
    path: Path
    metrics: Dict[str, float]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Aggregate multi-seed experiment outputs into a scientific + aesthetic HTML report."
    )
    p.add_argument(
        "--input-root",
        type=str,
        default="experiments/minimal_multitask_suite",
        help="Root folder containing test_metrics.json under */seed_*/",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default="reports/latest",
        help="Folder to save report assets and html.",
    )
    p.add_argument(
        "--title",
        type=str,
        default="PheMART2 Experiment Report",
        help="Report title.",
    )
    p.add_argument(
        "--primary-metric",
        type=str,
        default="main.mrr",
        help="Primary metric for leaderboard sorting.",
    )
    return p.parse_args()


def flatten_metrics(obj: Dict[str, object]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for task, val in obj.items():
        if not isinstance(val, dict):
            continue
        for m, x in val.items():
            if isinstance(x, (int, float)):
                out[f"{task}.{m}"] = float(x)
    return out


def find_run_records(input_root: Path) -> List[RunRecord]:
    files = sorted(input_root.rglob("test_metrics.json"))
    out: List[RunRecord] = []
    for fp in files:
        rel = fp.relative_to(input_root)
        parts = rel.parts
        seed_idx = None
        for i, part in enumerate(parts):
            if part.startswith("seed_"):
                seed_idx = i
                break
        if seed_idx is None:
            continue

        seed_raw = parts[seed_idx].replace("seed_", "")
        try:
            seed = int(seed_raw)
        except ValueError:
            continue

        if seed_idx == 0:
            experiment = input_root.name
        else:
            experiment = "/".join(parts[:seed_idx])

        with fp.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        out.append(
            RunRecord(
                experiment=experiment,
                seed=seed,
                path=fp,
                metrics=flatten_metrics(raw),
            )
        )
    return out


def records_to_dataframe(records: List[RunRecord]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for r in records:
        row: Dict[str, object] = {
            "experiment": r.experiment,
            "seed": r.seed,
            "metrics_path": str(r.path),
        }
        row.update(r.metrics)
        rows.append(row)
    if not rows:
        return pd.DataFrame(columns=["experiment", "seed", "metrics_path"])
    return pd.DataFrame(rows)


def metric_direction(metric_name: str) -> str:
    lower_keywords = ["mae", "rmse", "loss", "error"]
    if any(k in metric_name.lower() for k in lower_keywords):
        return "lower"
    return "higher"


def aggregate(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    base_cols = {"experiment", "seed", "metrics_path"}
    metric_cols = [c for c in df.columns if c not in base_cols]
    metric_cols = [c for c in metric_cols if pd.api.types.is_numeric_dtype(df[c])]
    if not metric_cols:
        return pd.DataFrame(), []

    rows: List[Dict[str, object]] = []
    for exp, g in df.groupby("experiment", sort=False):
        row: Dict[str, object] = {"experiment": exp, "n_seeds": int(g["seed"].nunique())}
        for m in metric_cols:
            vals = pd.to_numeric(g[m], errors="coerce").dropna().to_numpy(dtype=float)
            if vals.size == 0:
                continue
            mean = float(vals.mean())
            std = float(vals.std(ddof=1)) if vals.size > 1 else 0.0
            ci95 = float(1.96 * std / math.sqrt(vals.size)) if vals.size > 1 else 0.0
            row[f"{m}__mean"] = mean
            row[f"{m}__std"] = std
            row[f"{m}__ci95"] = ci95
        rows.append(row)

    out = pd.DataFrame(rows)
    return out, metric_cols


def format_pm(mean: float, std: float) -> str:
    return f"{mean:.4f} ± {std:.4f}"


def build_task_tables(summary: pd.DataFrame, metric_cols: List[str]) -> Dict[str, pd.DataFrame]:
    tasks = sorted(set(m.split(".", 1)[0] for m in metric_cols if "." in m))
    tables: Dict[str, pd.DataFrame] = {}
    for task in tasks:
        task_metrics = [m for m in metric_cols if m.startswith(f"{task}.")]
        rows: List[Dict[str, object]] = []
        for _, r in summary.iterrows():
            out = {"experiment": r["experiment"], "n_seeds": int(r["n_seeds"])}
            for m in task_metrics:
                mean_key = f"{m}__mean"
                std_key = f"{m}__std"
                if mean_key in r and pd.notna(r[mean_key]):
                    out[m] = format_pm(float(r[mean_key]), float(r.get(std_key, 0.0)))
            rows.append(out)
        tables[task] = pd.DataFrame(rows)
    return tables


def save_barplot_primary(summary: pd.DataFrame, primary_metric: str, out_png: Path) -> None:
    mean_col = f"{primary_metric}__mean"
    std_col = f"{primary_metric}__std"
    if mean_col not in summary.columns:
        return

    x = summary["experiment"].tolist()
    y = summary[mean_col].to_numpy(dtype=float)
    e = summary[std_col].to_numpy(dtype=float) if std_col in summary.columns else np.zeros_like(y)

    fig, ax = plt.subplots(figsize=(11, 4.8), dpi=140)
    colors = ["#2D6A4F"] * len(x)
    best_idx = int(np.argmax(y))
    colors[best_idx] = "#F4A261"
    ax.bar(x, y, yerr=e, color=colors, capsize=4, edgecolor="#163A32", linewidth=0.8)
    ax.set_title(f"Primary Metric: {primary_metric}", fontsize=13, pad=10)
    ax.set_ylabel(primary_metric)
    ax.set_xlabel("Experiment")
    ax.grid(axis="y", alpha=0.22)
    ax.set_axisbelow(True)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def save_seed_strip(df: pd.DataFrame, primary_metric: str, out_png: Path) -> None:
    if primary_metric not in df.columns:
        return
    d = df[["experiment", "seed", primary_metric]].copy()
    d = d.dropna(subset=[primary_metric])
    if d.empty:
        return

    exps = list(dict.fromkeys(d["experiment"].tolist()))
    exp_to_x = {e: i for i, e in enumerate(exps)}

    fig, ax = plt.subplots(figsize=(11, 4.4), dpi=140)
    for exp in exps:
        sub = d[d["experiment"] == exp]
        x0 = exp_to_x[exp]
        jitter = np.linspace(-0.12, 0.12, num=len(sub))
        ax.scatter(
            np.full(len(sub), x0, dtype=float) + jitter,
            sub[primary_metric].to_numpy(dtype=float),
            s=42,
            alpha=0.9,
            color="#287271",
            edgecolors="#0B1D26",
            linewidths=0.5,
        )
        ax.plot([x0 - 0.18, x0 + 0.18], [sub[primary_metric].mean(), sub[primary_metric].mean()], color="#E76F51", linewidth=2)

    ax.set_xticks(range(len(exps)))
    ax.set_xticklabels(exps, rotation=20, ha="right")
    ax.set_ylabel(primary_metric)
    ax.set_xlabel("Experiment")
    ax.set_title(f"Seed Stability ({primary_metric})", fontsize=13, pad=10)
    ax.grid(axis="y", alpha=0.22)
    ax.set_axisbelow(True)
    plt.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def build_leaderboard(summary: pd.DataFrame, metric_cols: List[str], primary_metric: str) -> pd.DataFrame:
    cols = ["experiment", "n_seeds"]
    for m in metric_cols:
        mean_key = f"{m}__mean"
        std_key = f"{m}__std"
        if mean_key in summary.columns:
            summary[f"{m}__pm"] = summary.apply(
                lambda r: format_pm(float(r[mean_key]), float(r.get(std_key, 0.0))),
                axis=1,
            )
            cols.append(f"{m}__pm")

    lb = summary[cols].copy()
    sort_col = f"{primary_metric}__mean"
    if sort_col in summary.columns:
        ascending = metric_direction(primary_metric) == "lower"
        lb = lb.loc[summary.sort_values(sort_col, ascending=ascending).index]
    return lb


def html_escape(x: object) -> str:
    s = str(x)
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def dataframe_to_html_table(df: pd.DataFrame, class_name: str = "tbl") -> str:
    if df.empty:
        return '<p class="muted">No data.</p>'
    headers = "".join(f"<th>{html_escape(c)}</th>" for c in df.columns)
    body_rows = []
    for _, row in df.iterrows():
        cells = "".join(f"<td>{html_escape(row[c])}</td>" for c in df.columns)
        body_rows.append(f"<tr>{cells}</tr>")
    body = "\n".join(body_rows)
    return f'<table class="{class_name}"><thead><tr>{headers}</tr></thead><tbody>{body}</tbody></table>'


def save_html_report(
    title: str,
    input_root: Path,
    out_html: Path,
    leaderboard: pd.DataFrame,
    task_tables: Dict[str, pd.DataFrame],
    run_df: pd.DataFrame,
    primary_metric: str,
    assets_rel: Dict[str, str],
) -> None:
    total_runs = len(run_df)
    experiments = run_df["experiment"].nunique() if not run_df.empty else 0
    seeds = run_df["seed"].nunique() if not run_df.empty else 0

    sections = []
    sections.append(
        f"""
        <section class="card hero">
          <div>
            <h1>{html_escape(title)}</h1>
            <p class="muted">Source: <code>{html_escape(str(input_root))}</code></p>
          </div>
          <div class="stats">
            <div><span>{experiments}</span><label>Experiments</label></div>
            <div><span>{seeds}</span><label>Seeds</label></div>
            <div><span>{total_runs}</span><label>Runs</label></div>
            <div><span>{html_escape(primary_metric)}</span><label>Primary Metric</label></div>
          </div>
        </section>
        """
    )

    if "bar_primary" in assets_rel:
        sections.append(
            f"""
            <section class="card">
              <h2>Primary Leaderboard</h2>
              <img src="{html_escape(assets_rel['bar_primary'])}" alt="primary metric barplot" />
            </section>
            """
        )

    if "seed_strip" in assets_rel:
        sections.append(
            f"""
            <section class="card">
              <h2>Seed Stability</h2>
              <img src="{html_escape(assets_rel['seed_strip'])}" alt="seed strip plot" />
            </section>
            """
        )

    sections.append(
        f"""
        <section class="card">
          <h2>Experiment Leaderboard (mean ± std)</h2>
          {dataframe_to_html_table(leaderboard)}
          <p class="muted">Direction rule: higher is better except error metrics (MAE/RMSE/loss).</p>
        </section>
        """
    )

    for task, tdf in sorted(task_tables.items()):
        sections.append(
            f"""
            <section class="card">
              <h2>Task: {html_escape(task)}</h2>
              {dataframe_to_html_table(tdf)}
            </section>
            """
        )

    sections.append(
        f"""
        <section class="card">
          <h2>Run-Level Details</h2>
          {dataframe_to_html_table(run_df.sort_values(['experiment', 'seed']))}
        </section>
        """
    )

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{html_escape(title)}</title>
  <style>
    :root {{
      --bg: #f3f6f4;
      --ink: #102a25;
      --muted: #4f6b63;
      --card: #ffffff;
      --line: #d7e2dc;
      --accent: #2D6A4F;
      --accent2: #E76F51;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--ink);
      background:
        radial-gradient(1300px 600px at -10% -10%, #e7efe9 0%, transparent 60%),
        radial-gradient(900px 450px at 110% 0%, #f8e4d8 0%, transparent 55%),
        var(--bg);
      font-family: "IBM Plex Sans", "Avenir Next", "Segoe UI", sans-serif;
      line-height: 1.45;
    }}
    .wrap {{ max-width: 1200px; margin: 30px auto 56px; padding: 0 20px; }}
    .card {{
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 18px;
      box-shadow: 0 8px 24px rgba(16, 42, 37, 0.06);
      padding: 20px 22px;
      margin-bottom: 18px;
    }}
    .hero {{
      display: grid;
      grid-template-columns: 1.3fr 1fr;
      gap: 16px;
      align-items: start;
    }}
    h1 {{
      margin: 0 0 8px 0;
      font-family: "Source Serif 4", "Georgia", serif;
      font-size: 2rem;
      letter-spacing: 0.2px;
    }}
    h2 {{ margin: 0 0 12px 0; font-size: 1.15rem; }}
    .muted {{ color: var(--muted); margin: 0; }}
    .stats {{
      display: grid;
      grid-template-columns: repeat(2, minmax(120px, 1fr));
      gap: 10px;
    }}
    .stats div {{
      background: linear-gradient(145deg, #f8fbf9, #edf4f0);
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 10px 12px;
    }}
    .stats span {{
      display: block;
      font-size: 1.25rem;
      font-weight: 700;
      color: var(--accent);
    }}
    .stats label {{
      font-size: 0.82rem;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.06em;
    }}
    img {{
      width: 100%;
      border-radius: 10px;
      border: 1px solid var(--line);
      background: #fff;
    }}
    table.tbl {{
      width: 100%;
      border-collapse: collapse;
      overflow-x: auto;
      display: block;
    }}
    table.tbl th, table.tbl td {{
      border-bottom: 1px solid #e4ede8;
      padding: 8px 10px;
      text-align: left;
      white-space: nowrap;
      font-size: 0.9rem;
    }}
    table.tbl thead th {{
      font-size: 0.78rem;
      text-transform: uppercase;
      letter-spacing: 0.04em;
      color: #35564d;
      background: #f3f8f5;
      position: sticky;
      top: 0;
      z-index: 1;
    }}
    @media (max-width: 900px) {{
      .hero {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    {''.join(sections)}
  </div>
</body>
</html>
"""

    out_html.write_text(html, encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root).resolve()
    out_dir = Path(args.output_dir).resolve()
    assets_dir = out_dir / "assets"
    out_dir.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(parents=True, exist_ok=True)

    records = find_run_records(input_root)
    if not records:
        raise ValueError(f"No test_metrics.json found under: {input_root}")

    run_df = records_to_dataframe(records)
    summary_df, metric_cols = aggregate(run_df)
    if summary_df.empty:
        raise ValueError("No numeric metrics detected from discovered test_metrics.json files.")

    mean_col_primary = f"{args.primary_metric}__mean"
    if mean_col_primary not in summary_df.columns:
        available = sorted(c.replace("__mean", "") for c in summary_df.columns if c.endswith("__mean"))
        raise ValueError(
            f"Primary metric '{args.primary_metric}' not found. Available: {available}"
        )

    ascending = metric_direction(args.primary_metric) == "lower"
    summary_df = summary_df.sort_values(mean_col_primary, ascending=ascending).reset_index(drop=True)

    leaderboard = build_leaderboard(summary_df.copy(), metric_cols, args.primary_metric)
    task_tables = build_task_tables(summary_df.copy(), metric_cols)

    run_csv = out_dir / "run_level_metrics.csv"
    summary_csv = out_dir / "summary_metrics.csv"
    leaderboard_csv = out_dir / "leaderboard.csv"
    run_df.to_csv(run_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)
    leaderboard.to_csv(leaderboard_csv, index=False)

    bar_png = assets_dir / "primary_bar.png"
    strip_png = assets_dir / "seed_strip.png"
    save_barplot_primary(summary_df, args.primary_metric, bar_png)
    save_seed_strip(run_df, args.primary_metric, strip_png)

    assets_rel = {}
    if bar_png.exists():
        assets_rel["bar_primary"] = str(bar_png.relative_to(out_dir))
    if strip_png.exists():
        assets_rel["seed_strip"] = str(strip_png.relative_to(out_dir))

    out_html = out_dir / "report.html"
    save_html_report(
        title=args.title,
        input_root=input_root,
        out_html=out_html,
        leaderboard=leaderboard,
        task_tables=task_tables,
        run_df=run_df,
        primary_metric=args.primary_metric,
        assets_rel=assets_rel,
    )

    print(f"input_root={input_root}")
    print(f"runs={len(run_df)} experiments={run_df['experiment'].nunique()}")
    print(f"output_html={out_html}")
    print(f"run_csv={run_csv}")
    print(f"summary_csv={summary_csv}")
    print(f"leaderboard_csv={leaderboard_csv}")


if __name__ == "__main__":
    main()
