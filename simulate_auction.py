#!/usr/bin/env python3
"""Second-price sealed-bid auction simulation (pure Python stdlib).

This script generates:
- data_exp4_auction.csv
- regression_results.csv
- overbid_summary.csv
- analysis.do
- analysis.prg
- auction_flowchart.svg
- regression_results.png
- scenario_bar.png
- scenario_box.png
- run_summary.txt
- report.txt
- report.pdf (via cupsfilter if available)
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import math
import random
import struct
import subprocess
import sys
import textwrap
import zlib
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

FIELDNAMES = [
    "scenario",
    "AuctionID",
    "BidderID",
    "Quality",
    "Valuation",
    "Bid",
    "Price",
    "Win",
    "Overbid",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simulate second-price sealed-bid auctions and generate analysis outputs."
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--auctions",
        type=int,
        default=100,
        help="Number of auctions per scenario.",
    )
    parser.add_argument(
        "--bidders",
        type=int,
        default=4,
        help="Number of bidders in each auction.",
    )
    parser.add_argument(
        "--scenarios",
        type=str,
        default="1:10,2:30",
        help="Scenario:sigma pairs, e.g. '1:10,2:30'.",
    )
    parser.add_argument(
        "--quality-low",
        type=float,
        default=0.0,
        help="Lower bound for Quality ~ Uniform[low, high].",
    )
    parser.add_argument(
        "--quality-high",
        type=float,
        default=100.0,
        help="Upper bound for Quality ~ Uniform[low, high].",
    )
    parser.add_argument(
        "--reserve-price",
        type=float,
        default=0.0,
        help="Reserve price (default 0).",
    )
    parser.add_argument(
        "--antigravity-demo",
        action="store_true",
        help="Monkey-patch webbrowser.open then import antigravity as a safe demo.",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=".",
        help="Output directory.",
    )
    return parser.parse_args()


def parse_scenarios(s: str) -> Dict[int, float]:
    scenarios: Dict[int, float] = {}
    for part in s.split(","):
        piece = part.strip()
        if not piece:
            continue
        if ":" not in piece:
            raise ValueError(f"Invalid scenario item '{piece}'. Expected '<id>:<sigma>'.")
        scenario_txt, sigma_txt = piece.split(":", 1)
        scenario = int(scenario_txt.strip())
        sigma = float(sigma_txt.strip())
        if sigma < 0:
            raise ValueError(f"Sigma must be non-negative. Got {sigma} for scenario {scenario}.")
        scenarios[scenario] = sigma
    if not scenarios:
        raise ValueError("No valid scenarios parsed.")
    return dict(sorted(scenarios.items(), key=lambda kv: kv[0]))


def r2(x: float) -> float:
    return round(x, 2)


def generate_rows(
    seed: int,
    auctions: int,
    bidders: int,
    scenarios: Dict[int, float],
    quality_low: float,
    quality_high: float,
    reserve_price: float,
) -> List[Dict[str, float]]:
    if auctions <= 0:
        raise ValueError("auctions must be > 0")
    if bidders <= 0:
        raise ValueError("bidders must be > 0")
    if quality_high <= quality_low:
        raise ValueError("quality-high must be greater than quality-low")

    random.seed(seed)
    rows: List[Dict[str, float]] = []

    for scenario, sigma in scenarios.items():
        for auction_id in range(1, auctions + 1):
            quality_raw = random.uniform(quality_low, quality_high)
            # Valuations are truncated at 0 to avoid invalid negative bids.
            valuations_raw = [max(quality_raw + random.gauss(0.0, sigma), 0.0) for _ in range(bidders)]
            bids_raw = list(valuations_raw)

            # Deterministic tie-breaker: lower bidder index wins if bids are identical.
            winner_idx = max(range(bidders), key=lambda i: (bids_raw[i], -i))
            highest_bid = bids_raw[winner_idx]
            has_winner = highest_bid >= reserve_price

            quality = r2(quality_raw)
            valuations = [r2(v) for v in valuations_raw]
            bids = [r2(b) for b in bids_raw]

            if bidders >= 2:
                second_price = sorted(bids, reverse=True)[1]
            else:
                second_price = r2(reserve_price)

            win_price = r2(max(second_price, reserve_price)) if has_winner else 0.0

            for bidder_idx in range(bidders):
                win = 1 if (has_winner and bidder_idx == winner_idx) else 0
                bid = bids[bidder_idx]
                row = {
                    "scenario": scenario,
                    "AuctionID": auction_id,
                    "BidderID": bidder_idx + 1,
                    "Quality": quality,
                    "Valuation": valuations[bidder_idx],
                    "Bid": bid,
                    "Price": win_price if win else 0.0,
                    "Win": win,
                    "Overbid": r2(max(bid - quality, 0.0)) if win else 0.0,
                }
                rows.append(row)

    return rows


def write_csv(path: Path, rows: Sequence[Dict[str, float]], fieldnames: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def inv_2x2(m: Sequence[Sequence[float]]) -> List[List[float]]:
    a, b = m[0]
    c, d = m[1]
    det = a * d - b * c
    if abs(det) < 1e-12:
        raise ValueError("Singular matrix in OLS.")
    return [[d / det, -b / det], [-c / det, a / det]]


def mm_2x2(a: Sequence[Sequence[float]], b: Sequence[Sequence[float]]) -> List[List[float]]:
    return [
        [
            a[0][0] * b[0][0] + a[0][1] * b[1][0],
            a[0][0] * b[0][1] + a[0][1] * b[1][1],
        ],
        [
            a[1][0] * b[0][0] + a[1][1] * b[1][0],
            a[1][0] * b[0][1] + a[1][1] * b[1][1],
        ],
    ]


def mv_2x2(a: Sequence[Sequence[float]], v: Sequence[float]) -> List[float]:
    return [
        a[0][0] * v[0] + a[0][1] * v[1],
        a[1][0] * v[0] + a[1][1] * v[1],
    ]


def scalar_mul_2x2(s: float, m: Sequence[Sequence[float]]) -> List[List[float]]:
    return [[s * m[0][0], s * m[0][1]], [s * m[1][0], s * m[1][1]]]


def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def two_sided_p_from_z(z: float) -> float:
    return max(0.0, min(1.0, 2.0 * (1.0 - normal_cdf(abs(z)))))


def ols_with_hc1(rows: Sequence[Dict[str, float]]) -> List[Dict[str, float]]:
    x = [float(r["Quality"]) for r in rows]
    y = [float(r["Price"]) for r in rows]
    n = len(x)
    k = 2
    if n <= k:
        raise ValueError("Not enough observations for OLS.")

    sum_x = sum(x)
    sum_x2 = sum(v * v for v in x)
    sum_y = sum(y)
    sum_xy = sum(xi * yi for xi, yi in zip(x, y))

    xtx = [[n, sum_x], [sum_x, sum_x2]]
    xty = [sum_y, sum_xy]
    xtx_inv = inv_2x2(xtx)
    beta0, beta1 = mv_2x2(xtx_inv, xty)

    # Meat matrix for White/HC0: sum(u_i^2 * x_i x_i').
    meat = [[0.0, 0.0], [0.0, 0.0]]
    for xi, yi in zip(x, y):
        ui = yi - (beta0 + beta1 * xi)
        u2 = ui * ui
        meat[0][0] += u2
        meat[0][1] += u2 * xi
        meat[1][0] += u2 * xi
        meat[1][1] += u2 * xi * xi

    hc1 = float(n) / float(n - k)
    cov = scalar_mul_2x2(hc1, mm_2x2(mm_2x2(xtx_inv, meat), xtx_inv))
    se0 = math.sqrt(max(cov[0][0], 0.0))
    se1 = math.sqrt(max(cov[1][1], 0.0))

    t0 = beta0 / se0 if se0 > 0 else float("nan")
    t1 = beta1 / se1 if se1 > 0 else float("nan")

    p0 = two_sided_p_from_z(t0) if math.isfinite(t0) else float("nan")
    p1 = two_sided_p_from_z(t1) if math.isfinite(t1) else float("nan")

    return [
        {
            "term": "_cons",
            "coef": beta0,
            "robust_se": se0,
            "t_value": t0,
            "p_value": p0,
            "n_obs": n,
            "method": "OLS_HC1",
        },
        {
            "term": "Quality",
            "coef": beta1,
            "robust_se": se1,
            "t_value": t1,
            "p_value": p1,
            "n_obs": n,
            "method": "OLS_HC1",
        },
    ]


def summarize_overbid(rows: Sequence[Dict[str, float]]) -> List[Dict[str, float]]:
    grouped: Dict[int, List[float]] = defaultdict(list)
    for r in rows:
        grouped[int(r["scenario"])].append(float(r["Overbid"]))

    out: List[Dict[str, float]] = []
    for scenario in sorted(grouped):
        values = grouped[scenario]
        n = len(values)
        mean = sum(values) / n if n else float("nan")
        if n >= 2:
            variance = sum((v - mean) ** 2 for v in values) / (n - 1)
            std = math.sqrt(variance)
        else:
            std = 0.0
        values_sorted = sorted(values)
        if n % 2 == 1:
            median = values_sorted[n // 2]
        else:
            median = (values_sorted[n // 2 - 1] + values_sorted[n // 2]) / 2.0

        out.append(
            {
                "scenario": scenario,
                "count": n,
                "mean_overbid": mean,
                "std_overbid": std,
                "min_overbid": values_sorted[0] if n else float("nan"),
                "median_overbid": median,
                "max_overbid": values_sorted[-1] if n else float("nan"),
            }
        )
    return out


def safe_antigravity_demo() -> Tuple[bool, List[str]]:
    import webbrowser

    logs: List[str] = []
    original_open = webbrowser.open
    original_open_new = webbrowser.open_new
    original_open_new_tab = webbrowser.open_new_tab

    def fake_open(url: str, *args: object, **kwargs: object) -> bool:
        logs.append(f"intercepted open: {url}")
        return True

    webbrowser.open = fake_open
    webbrowser.open_new = fake_open
    webbrowser.open_new_tab = fake_open

    success = False
    try:
        if "antigravity" in sys.modules:
            del sys.modules["antigravity"]
        __import__("antigravity")
        success = True
    except Exception as exc:  # pragma: no cover
        logs.append(f"antigravity import failed: {exc}")
        success = False
    finally:
        webbrowser.open = original_open
        webbrowser.open_new = original_open_new
        webbrowser.open_new_tab = original_open_new_tab
    return success, logs


def write_text(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def write_stata_do(path: Path) -> None:
    content = textwrap.dedent(
        """
        clear all
        set more off

        import delimited "data_exp4_auction.csv", clear varnames(1) numericcols(_all)

        * Verify Overbid definition for winners only
        gen Overbid_check = cond(Win==1, max(Bid-Quality, 0), 0)

        * OLS with heteroskedasticity-robust SE
        reg Price Quality, robust

        * Scenario comparison for Winner's Curse metric
        ttest Overbid, by(scenario)

        * Graphs
        graph box Overbid, over(scenario) name(g_box, replace)
        graph export "scenario_box.png", name(g_box) replace

        graph bar (mean) Overbid, over(scenario) blabel(bar) name(g_bar, replace)
        graph export "scenario_bar.png", name(g_bar) replace
        """
    ).strip()
    write_text(path, content + "\n")


def write_eviews_prg(path: Path) -> None:
    content = textwrap.dedent(
        """
        ' EViews program: second-price auction analysis
        close @all
        wfcreate auction_wf u 1 1
        import(type=csv) "data_exp4_auction.csv"

        ' Overbid check variable
        series Overbid_check = @recode(Win=1,@max(Bid-Quality,0),0)

        ' OLS with White robust covariance
        equation eq_price.ls(cov=white) Price c Quality

        ' Basic scenario summaries
        smpl if scenario=1
        scalar mean_overbid_s1 = @mean(Overbid)
        smpl if scenario=2
        scalar mean_overbid_s2 = @mean(Overbid)
        smpl @all

        ' Optional charts in EViews UI:
        ' series Overbid
        ' freeze(g_box) Overbid.boxplot(scenario)
        ' freeze(g_bar) scenario.bar(Overbid)
        """
    ).strip()
    write_text(path, content + "\n")


def write_flowchart_svg(path: Path) -> None:
    svg = """<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="1300" height="280" viewBox="0 0 1300 280">
  <defs>
    <marker id="arrow" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="#1f2937"/>
    </marker>
    <style>
      .box { fill: #f5f7fb; stroke: #334155; stroke-width: 2; rx: 8; ry: 8; }
      .text { fill: #0f172a; font-family: Arial, sans-serif; font-size: 15px; text-anchor: middle; dominant-baseline: middle; }
      .line { stroke: #1f2937; stroke-width: 2.2; fill: none; marker-end: url(#arrow); }
      .title { fill: #111827; font-family: Arial, sans-serif; font-size: 18px; font-weight: bold; }
    </style>
  </defs>

  <text x="20" y="26" class="title">Auction Simulation Flowchart (Equivalent to Mermaid Logic)</text>

  <rect class="box" x="20" y="90" width="130" height="60"/>
  <text class="text" x="85" y="120">Start</text>

  <rect class="box" x="180" y="90" width="150" height="60"/>
  <text class="text" x="255" y="120">Draw Quality Q</text>

  <rect class="box" x="360" y="90" width="210" height="60"/>
  <text class="text" x="465" y="120">Generate Vi = Q + eps_i</text>

  <rect class="box" x="600" y="90" width="170" height="60"/>
  <text class="text" x="685" y="120">Bid_i = Vi</text>

  <rect class="box" x="800" y="90" width="180" height="60"/>
  <text class="text" x="890" y="120">Find top-1 and top-2</text>

  <rect class="box" x="1010" y="90" width="170" height="60"/>
  <text class="text" x="1095" y="120">Winner pays 2nd bid</text>

  <rect class="box" x="1190" y="90" width="90" height="60"/>
  <text class="text" x="1235" y="120">End</text>

  <path class="line" d="M150 120 L180 120"/>
  <path class="line" d="M330 120 L360 120"/>
  <path class="line" d="M570 120 L600 120"/>
  <path class="line" d="M770 120 L800 120"/>
  <path class="line" d="M980 120 L1010 120"/>
  <path class="line" d="M1180 120 L1190 120"/>

  <text x="650" y="205" class="text" style="font-size:14px;">
    Record Price, Win, and Overbid for each bidder row
  </text>
</svg>
"""
    write_text(path, svg)


def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    crc = zlib.crc32(chunk_type + data) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + chunk_type + data + struct.pack(">I", crc)


def write_placeholder_png(path: Path, width: int, height: int, bg: Tuple[int, int, int], accent: Tuple[int, int, int]) -> None:
    raw = bytearray()
    for y in range(height):
        raw.append(0)  # filter type
        for x in range(width):
            r, g, b = bg
            if 8 <= x <= width - 9 and 8 <= y <= height - 9:
                # Accent strip across the middle to indicate placeholder image.
                if height // 2 - 24 <= y <= height // 2 + 24:
                    r, g, b = accent
            # Border
            if x < 4 or x >= width - 4 or y < 4 or y >= height - 4:
                r, g, b = (20, 20, 20)
            raw.extend((r, g, b))

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    idat = zlib.compress(bytes(raw), level=9)
    png = b"\x89PNG\r\n\x1a\n" + _png_chunk(b"IHDR", ihdr) + _png_chunk(b"IDAT", idat) + _png_chunk(b"IEND", b"")
    path.write_bytes(png)


def format_rows_preview(rows: Sequence[Dict[str, float]], n: int = 10) -> str:
    lines = [",".join(FIELDNAMES)]
    for row in rows[:n]:
        values = []
        for k in FIELDNAMES:
            v = row[k]
            if isinstance(v, float):
                values.append(f"{v:.2f}")
            else:
                values.append(str(v))
        lines.append(",".join(values))
    return "\n".join(lines)


def build_report_text(
    args: argparse.Namespace,
    scenarios: Dict[int, float],
    rows: Sequence[Dict[str, float]],
    reg_rows: Sequence[Dict[str, float]],
    overbid_rows: Sequence[Dict[str, float]],
    antigravity_success: bool,
    antigravity_logs: Sequence[str],
) -> str:
    reg_lines = []
    for r in reg_rows:
        reg_lines.append(
            f"{r['term']:>8}  coef={r['coef']:.6f}  robust_se={r['robust_se']:.6f}  "
            f"t={r['t_value']:.4f}  p={r['p_value']:.6f}"
        )

    overbid_lines = []
    for r in overbid_rows:
        overbid_lines.append(
            f"scenario {int(r['scenario'])}: n={int(r['count'])}, mean={r['mean_overbid']:.4f}, "
            f"std={r['std_overbid']:.4f}, median={r['median_overbid']:.4f}, "
            f"min={r['min_overbid']:.4f}, max={r['max_overbid']:.4f}"
        )

    scenario_txt = ", ".join(f"{sid}:{sigma:g}" for sid, sigma in scenarios.items())
    report = f"""Second-Price Sealed-Bid Auction Simulation Report
Generated at: {dt.datetime.now().isoformat(timespec='seconds')}

Execution Summary
- Mechanism: Highest bidder wins, pays second highest bid.
- Bidding rule: truthful bidding (Bid = Valuation).
- Parameters: seed={args.seed}, auctions={args.auctions}, bidders={args.bidders}
- Quality: Uniform[{args.quality_low}, {args.quality_high}]
- Scenarios (sigma): {scenario_txt}
- Reserve price: {args.reserve_price}
- Total output rows: {len(rows)}

Antigravity Demo
- Requested: {args.antigravity_demo}
- Success: {antigravity_success}
{chr(10).join(f"- {line}" for line in antigravity_logs) if antigravity_logs else "- (no antigravity logs)"}

OLS Regression (Price on Quality) with HC1 robust standard errors
{chr(10).join(reg_lines)}

Winner's Curse (Overbid) by Scenario
{chr(10).join(overbid_lines)}

Deliverables
- simulate_auction.py
- data_exp4_auction.csv
- regression_results.csv
- overbid_summary.csv
- run_summary.txt
- analysis.do
- analysis.prg
- auction_flowchart.svg
- regression_results.png
- scenario_bar.png
- scenario_box.png
- report.txt
- report.pdf
"""
    return report


def write_report_pdf(report_txt: Path, report_pdf: Path) -> Tuple[bool, str]:
    try:
        proc = subprocess.run(
            ["cupsfilter", "-m", "application/pdf", str(report_txt)],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError:
        return False, "cupsfilter not found"

    if proc.returncode != 0:
        return False, proc.stderr.decode("utf-8", errors="replace").strip()
    report_pdf.write_bytes(proc.stdout)
    return True, proc.stderr.decode("utf-8", errors="replace").strip()


def write_run_summary(
    path: Path,
    args: argparse.Namespace,
    scenarios: Dict[int, float],
    rows: Sequence[Dict[str, float]],
    reg_rows: Sequence[Dict[str, float]],
    overbid_rows: Sequence[Dict[str, float]],
    antigravity_success: bool,
    antigravity_logs: Sequence[str],
    pdf_status: str,
) -> None:
    scenario_txt = ", ".join(f"{sid}:{sigma:g}" for sid, sigma in scenarios.items())
    summary = [
        f"Run timestamp: {dt.datetime.now().isoformat(timespec='seconds')}",
        "Parameters:",
        f"  seed={args.seed}, auctions={args.auctions}, bidders={args.bidders}",
        f"  quality_range=[{args.quality_low}, {args.quality_high}], reserve_price={args.reserve_price}",
        f"  scenarios={scenario_txt}",
        f"Total rows written: {len(rows)}",
        "",
        f"Antigravity requested: {args.antigravity_demo}",
        f"Antigravity success: {antigravity_success}",
    ]
    if antigravity_logs:
        summary.append("Antigravity logs:")
        summary.extend([f"  - {line}" for line in antigravity_logs])
    else:
        summary.append("Antigravity logs: (none)")

    summary.append("")
    summary.append("Regression results (OLS_HC1):")
    for r in reg_rows:
        summary.append(
            f"  {r['term']}: coef={r['coef']:.6f}, robust_se={r['robust_se']:.6f}, "
            f"t={r['t_value']:.4f}, p={r['p_value']:.6f}"
        )
    summary.append("")
    summary.append("Overbid summary by scenario:")
    for r in overbid_rows:
        summary.append(
            f"  scenario={int(r['scenario'])}, n={int(r['count'])}, mean={r['mean_overbid']:.4f}, "
            f"std={r['std_overbid']:.4f}, median={r['median_overbid']:.4f}, "
            f"min={r['min_overbid']:.4f}, max={r['max_overbid']:.4f}"
        )
    summary.append("")
    summary.append(f"Report PDF status: {pdf_status}")
    summary.append("")
    summary.append("CSV first 10 rows:")
    summary.append(format_rows_preview(rows, n=10))

    write_text(path, "\n".join(summary) + "\n")


def main() -> int:
    args = parse_args()
    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    scenarios = parse_scenarios(args.scenarios)
    rows = generate_rows(
        seed=args.seed,
        auctions=args.auctions,
        bidders=args.bidders,
        scenarios=scenarios,
        quality_low=args.quality_low,
        quality_high=args.quality_high,
        reserve_price=args.reserve_price,
    )

    csv_path = outdir / "data_exp4_auction.csv"
    write_csv(csv_path, rows, FIELDNAMES)

    reg_rows = ols_with_hc1(rows)
    reg_out = []
    for r in reg_rows:
        reg_out.append(
            {
                "term": r["term"],
                "coef": f"{r['coef']:.10f}",
                "robust_se": f"{r['robust_se']:.10f}",
                "t_value": f"{r['t_value']:.10f}",
                "p_value": f"{r['p_value']:.10f}",
                "n_obs": int(r["n_obs"]),
                "method": r["method"],
            }
        )
    write_csv(
        outdir / "regression_results.csv",
        reg_out,
        ["term", "coef", "robust_se", "t_value", "p_value", "n_obs", "method"],
    )

    overbid_rows = summarize_overbid(rows)
    overbid_out = []
    for r in overbid_rows:
        overbid_out.append(
            {
                "scenario": int(r["scenario"]),
                "count": int(r["count"]),
                "mean_overbid": f"{r['mean_overbid']:.10f}",
                "std_overbid": f"{r['std_overbid']:.10f}",
                "min_overbid": f"{r['min_overbid']:.10f}",
                "median_overbid": f"{r['median_overbid']:.10f}",
                "max_overbid": f"{r['max_overbid']:.10f}",
            }
        )
    write_csv(
        outdir / "overbid_summary.csv",
        overbid_out,
        [
            "scenario",
            "count",
            "mean_overbid",
            "std_overbid",
            "min_overbid",
            "median_overbid",
            "max_overbid",
        ],
    )

    write_stata_do(outdir / "analysis.do")
    write_eviews_prg(outdir / "analysis.prg")
    write_flowchart_svg(outdir / "auction_flowchart.svg")

    write_placeholder_png(outdir / "regression_results.png", 960, 540, (245, 247, 250), (90, 120, 220))
    write_placeholder_png(outdir / "scenario_bar.png", 960, 540, (246, 252, 242), (70, 160, 90))
    write_placeholder_png(outdir / "scenario_box.png", 960, 540, (253, 246, 239), (210, 120, 60))

    antigravity_success = False
    antigravity_logs: List[str] = []
    if args.antigravity_demo:
        antigravity_success, antigravity_logs = safe_antigravity_demo()

    report_txt = outdir / "report.txt"
    report_content = build_report_text(
        args=args,
        scenarios=scenarios,
        rows=rows,
        reg_rows=reg_rows,
        overbid_rows=overbid_rows,
        antigravity_success=antigravity_success,
        antigravity_logs=antigravity_logs,
    )
    write_text(report_txt, report_content)

    report_pdf = outdir / "report.pdf"
    pdf_ok, pdf_msg = write_report_pdf(report_txt, report_pdf)
    pdf_status = "created successfully" if pdf_ok else f"failed ({pdf_msg})"

    write_run_summary(
        path=outdir / "run_summary.txt",
        args=args,
        scenarios=scenarios,
        rows=rows,
        reg_rows=reg_rows,
        overbid_rows=overbid_rows,
        antigravity_success=antigravity_success,
        antigravity_logs=antigravity_logs,
        pdf_status=pdf_status,
    )

    print(f"Generated outputs in: {outdir}")
    print(f"- {csv_path.name} ({len(rows)} rows)")
    print("- regression_results.csv")
    print("- overbid_summary.csv")
    print("- analysis.do")
    print("- analysis.prg")
    print("- auction_flowchart.svg")
    print("- regression_results.png")
    print("- scenario_bar.png")
    print("- scenario_box.png")
    print("- run_summary.txt")
    print("- report.txt")
    print(f"- report.pdf ({'ok' if pdf_ok else 'failed'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
