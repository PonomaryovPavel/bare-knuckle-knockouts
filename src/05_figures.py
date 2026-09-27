"""Три графика прямо из посчитанных данных."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

DATA, FIG, UFC = Path("data"), Path("figures"), Path("/home/claude/ufc/data")
SURF, INK, MUTED, RULE = "#fcfcfb", "#16130f", "#6b645c", "#e0dbd4"
RED, SLATE = "#d1344b", "#4a5d78"
FONT = "ui-sans-serif,-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif"
W, H = 720, 300
L, RGT, T, B = 62, 18, 62, 46


def ru(x: float, d: int = 1) -> str:
    """Число с запятой — заменять точки во всей разметке нельзя, ломаются координаты."""
    return f"{x:.{d}f}".replace(".", ",")


def esc(t: str) -> str:
    return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p, d = k / n, 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def head(title: str, sub: str, legend=None) -> list[str]:
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
         f'role="img" aria-labelledby="t d" font-family="{FONT}">',
         f'<title id="t">{esc(title)}</title><desc id="d">{esc(sub)}</desc>',
         f'<rect width="{W}" height="{H}" fill="{SURF}"/>',
         f'<text x="{L}" y="22" font-size="13.5" font-weight="600" fill="{INK}">{esc(title)}</text>',
         f'<text x="{L}" y="39" font-size="11" fill="{MUTED}">{esc(sub)}</text>']
    for i, (color, text, dashed) in enumerate(legend or []):
        x = L + i * 300
        dash = ' stroke-dasharray="5 3"' if dashed else ""
        s.append(f'<line x1="{x}" y1="52" x2="{x+22}" y2="52" stroke="{color}" stroke-width="2.4"{dash}/>')
        s.append(f'<text x="{x+28}" y="55.5" font-size="10.5" fill="{MUTED}">{esc(text)}</text>')
    return s


def ypix(v: float, ymax: float) -> float:
    return H - B - (v / ymax) * (H - B - T)


def grid(ymax: float, ticks, xlabels, xpos, unit="%") -> list[str]:
    s = []
    for t in ticks:
        y = ypix(t, ymax)
        s.append(f'<line x1="{L}" y1="{y:.1f}" x2="{W-RGT}" y2="{y:.1f}" stroke="{RULE}"/>')
        s.append(f'<text x="{L-8}" y="{y+3.5:.1f}" font-size="10.5" fill="{MUTED}" '
                 f'text-anchor="end">{t:g}{unit}</text>')
    for lab, x in zip(xlabels, xpos):
        for j, part in enumerate(lab.split("\n")):
            s.append(f'<text x="{x:.1f}" y="{H-B+17+j*14}" font-size="11" fill="{INK}" '
                     f'text-anchor="middle">{esc(part)}</text>')
    return s


def fig_exposure(c: dict) -> str:
    e, r = c["exposure"], c["ratios"]
    bars = [("BKFC\nголые кулаки", e["bkfc"]["per100"], RED, 0.92),
            ("UFC\nвне партера", e["ufc"]["per100_standing"], SLATE, 0.62),
            ("UFC\nвсё время боя", e["ufc"]["per100"], SLATE, 0.32)]
    ymax = 15
    s = head("Нокауты на сто минут в ринге",
             f"на голых кулаках в {ru(r['per_minute_standing'])} раза чаще, чем в перчатках; "
             f"у ММА вычтено время контроля в партере")
    xs = [L + 110 + i * 172 for i in range(3)]
    s += grid(ymax, [0, 5, 10, 15], [b[0] for b in bars], xs, unit="")
    for x, (_, v, col, op) in zip(xs, bars):
        s.append(f'<rect x="{x-36:.0f}" y="{ypix(v,ymax):.1f}" width="72" '
                 f'height="{H-B-ypix(v,ymax):.1f}" fill="{col}" opacity="{op}"/>')
        s.append(f'<text x="{x}" y="{ypix(v,ymax)-9:.1f}" font-size="13" fill="{INK}" '
                 f'text-anchor="middle" font-weight="600">{ru(v)}</text>')
    return "\n".join(s) + "\n</svg>\n"


def fig_dose(bk: pd.DataFrame, uf: pd.DataFrame) -> str:
    ymax = 55
    xs = [L + 80 + i * (W - L - RGT - 160) / 2 for i in range(3)]
    s = head("След пропущенного нокаута",
             "риск быть нокаутированным в очередном бою по числу прошлых нокаутных поражений",
             [(RED, "голые кулаки, BKFC", False), (SLATE, "перчатки, UFC", True)])
    s += grid(ymax, [0, 20, 40], ["ни одного", "один", "два и больше"], xs)
    for d, col, dashed in ((bk, RED, False), (uf, SLATE, True)):
        g = d.assign(c=d.prior_ko_losses.clip(upper=2)).groupby("c").ko_loss.agg(["sum", "size"])
        pts, dash = [], (' stroke-dasharray="5 3"' if dashed else "")
        for i, x in enumerate(xs):
            k, n = int(g.loc[i, "sum"]), int(g.loc[i, "size"])
            p = 100 * k / n
            lo, hi = wilson(k, n)
            pts.append(f"{x:.1f},{ypix(p,ymax):.1f}")
            s.append(f'<line x1="{x:.1f}" y1="{ypix(100*lo,ymax):.1f}" x2="{x:.1f}" '
                     f'y2="{ypix(100*hi,ymax):.1f}" stroke="{col}" stroke-width="1.4" opacity=".45"/>')
        s.insert(5, f'<polyline points="{" ".join(pts)}" fill="none" stroke="{col}" '
                    f'stroke-width="2.2"{dash}/>')
        for i, x in enumerate(xs):
            k, n = int(g.loc[i, "sum"]), int(g.loc[i, "size"])
            p = 100 * k / n
            s.append(f'<circle cx="{x:.1f}" cy="{ypix(p,ymax):.1f}" r="4" fill="{SURF}" '
                     f'stroke="{col}" stroke-width="2.2"/>')
            s.append(f'<text x="{x:.1f}" y="{ypix(p,ymax)-13:.1f}" font-size="10.5" fill="{col}" '
                     f'text-anchor="middle" font-weight="600">{ru(p)}</text>')
    return "\n".join(s) + "\n</svg>\n"


def fig_returns(R: dict, U: dict) -> str:
    global B
    B, ymax = 62, 55
    bars = [("после нокаута\nголые кулаки", R["raw_ko"], RED, 0.92),
            ("после иного\nголые кулаки", R["raw_oth"], RED, 0.38),
            ("после нокаута\nперчатки", U["raw_ko"], SLATE, 0.8),
            ("после решения\nперчатки", U["raw_dec"], SLATE, 0.35)]
    s = head("Первый бой после поражения",
             "риск проиграть нокаутом в зависимости от того, чем кончилось прошлое поражение")
    xs = [L + 80 + i * 158 for i in range(4)]
    s += grid(ymax, [0, 20, 40], [b[0] for b in bars], xs)
    for x, (_, d, col, op) in zip(xs, bars):
        v = 100 * d["risk"]
        lo, hi = wilson(d["ko"], d["n"])
        s.append(f'<rect x="{x-30:.0f}" y="{ypix(v,ymax):.1f}" width="60" '
                 f'height="{H-B-ypix(v,ymax):.1f}" fill="{col}" opacity="{op}"/>')
        s.append(f'<line x1="{x}" y1="{ypix(100*lo,ymax):.1f}" x2="{x}" y2="{ypix(100*hi,ymax):.1f}" '
                 f'stroke="{INK}" stroke-width="1.4"/>')
        s.append(f'<text x="{x}" y="{ypix(100*hi,ymax)-8:.1f}" font-size="12" fill="{INK}" '
                 f'text-anchor="middle" font-weight="600">{ru(v)}%</text>')
        s.append(f'<text x="{x}" y="{H-B+46}" font-size="9.5" fill="{MUTED}" '
                 f'text-anchor="middle">{d["ko"]} из {d["n"]}</text>')
    out = "\n".join(s) + "\n</svg>\n"
    B = 46
    return out


def main() -> None:
    FIG.mkdir(exist_ok=True)
    c = json.loads((DATA / "compare.json").read_text())
    R = json.loads((DATA / "results.json").read_text())
    U = json.loads((Path("/home/claude/ufc/data") / "results.json").read_text())
    bk = pd.read_parquet(DATA / "panel.parquet")
    uf = pd.read_parquet(UFC / "panel.parquet")
    for name, svg in (("exposure.svg", fig_exposure(c)),
                      ("dose.svg", fig_dose(bk, uf)),
                      ("returns.svg", fig_returns(R, U))):
        (FIG / name).write_text(svg)
        print(f"{name}: {len(svg)} байт")


if __name__ == "__main__":
    main()
