"""Голые кулаки против перчаток: экспозиция и подпись повторного нокаута.

Две части. Первая — сколько нокаутов приходится на минуту в ринге, с честным
знаменателем: в ММА почти половина времени уходит на контроль в партере, и
эту часть надо вычесть, иначе сравнение завышено. Вторая — одна модель на
объединённых данных с взаимодействием по виду спорта: отличается ли подпись
повторного нокаута на голых кулаках от подписи в перчатках.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model_core import fit, hr  # noqa: E402

DATA, UFC = Path("data"), Path("/home/claude/ufc")
CTRL = "fight_no + log_layoff + prior_win_rate + big + era"
OUT: dict = {}


def mmss(x: object) -> float:
    m = re.match(r"^\s*(\d+):(\d{2})\s*$", str(x))
    return int(m.group(1)) * 60 + int(m.group(2)) if m else np.nan


def exposure() -> None:
    """Нокауты на сто минут боя."""
    b = pd.read_parquet(DATA / "bouts.parquet")
    b = b[b.date.notna()]
    ko_b, min_b, n_b = int(b.kind.isin(["ko", "tko"]).sum()), b.seconds.sum() / 60, len(b)

    st = pd.read_csv(UFC / "data_raw/ufc_fight_stats.csv")
    for c in ("EVENT", "BOUT"):
        st[c] = st[c].astype(str).str.strip()
    st["ctrl"] = st["CTRL"].map(mmss)
    per = st.groupby(["EVENT", "BOUT"], as_index=False).ctrl.sum(min_count=1)
    per["fight_id"] = per.EVENT.str.cat(per.BOUT, sep=" | ")
    u = pd.read_parquet(UFC / "data/fights.parquet").drop_duplicates("fight_id")
    u = u[u.year >= 2001].merge(per[["fight_id", "ctrl"]], on="fight_id", how="left")
    u["ctrl"] = u.ctrl.fillna(0).clip(upper=u.seconds)
    ko_u, n_u = int((u.clean & u.METHOD.eq("KO/TKO")).sum()), len(u)
    min_u_all, min_u_stand = u.seconds.sum() / 60, (u.seconds - u.ctrl).sum() / 60

    rows = [("BKFC, голые кулаки", n_b, ko_b, min_b, min_b),
            ("UFC, перчатки", n_u, ko_u, min_u_all, min_u_stand)]
    print(f"{'':22} {'боёв':>6} {'нокаутов':>9} {'доля боёв':>10} "
          f"{'на 100 мин боя':>15} {'на 100 мин вне партера':>23}")
    for name, n, k, m_all, m_st in rows:
        print(f"{name:22} {n:6} {k:9} {100*k/n:9.1f}% {100*k/m_all:15.2f} {100*k/m_st:23.2f}")
    OUT["exposure"] = {
        "bkfc": {"bouts": n_b, "ko": ko_b, "minutes": float(min_b),
                 "share": ko_b / n_b, "per100": 100 * ko_b / min_b},
        "ufc": {"bouts": n_u, "ko": ko_u, "minutes": float(min_u_all),
                "minutes_standing": float(min_u_stand), "share": ko_u / n_u,
                "per100": 100 * ko_u / min_u_all, "per100_standing": 100 * ko_u / min_u_stand,
                "control_share": float(u.ctrl.sum() / u.seconds.sum())},
    }
    e = OUT["exposure"]
    OUT["ratios"] = {"per_bout": e["bkfc"]["share"] / e["ufc"]["share"],
                     "per_minute": e["bkfc"]["per100"] / e["ufc"]["per100"],
                     "per_minute_standing": e["bkfc"]["per100"] / e["ufc"]["per100_standing"]}
    r = OUT["ratios"]
    print(f"\nво сколько раз чаще на голых кулаках: по доле боёв {r['per_bout']:.2f}, "
          f"на минуту боя {r['per_minute']:.2f}, на минуту вне партера {r['per_minute_standing']:.2f}")
    print(f"доля времени под контролем в партере в UFC: {100*e['ufc']['control_share']:.1f}%")


def harmonise(d: pd.DataFrame, sport: str, big: set[str]) -> pd.DataFrame:
    out = pd.DataFrame({
        "ko_loss": d.ko_loss.astype(int),
        "prior_ko_losses": d.prior_ko_losses,
        "prior_other_losses": d.prior_other_losses,
        "fight_no": d.fight_no,
        "log_layoff": np.log(d.layoff.clip(lower=7)),
        "prior_win_rate": (d.prior_wins / (d.fight_no - 1)).fillna(0),
        "big": d.division.isin(big).astype(int),
        "year": d.date.dt.year if np.issubdtype(d.date.dtype, np.datetime64) else d.year,
        "fighter": sport + " | " + d.fighter,
        "sport": sport,
    })
    return out.dropna()


def compare() -> None:
    b = pd.read_parquet(DATA / "panel.parquet")
    u = pd.read_parquet(UFC / "data/panel.parquet")
    B = harmonise(b, "bkfc", {"Light Heavyweight", "Cruiserweight", "Heavyweight", "Ironweight"})
    U = harmonise(u, "ufc", {"Light Heavyweight", "Heavyweight"})
    for tag, uu in (("весь UFC с 2001", U), ("UFC c 2018, тот же период", U[U.year >= 2018])):
        d = pd.concat([B, uu], ignore_index=True)
        d["bare"] = (d.sport == "bkfc").astype(int)
        d["era"] = d.year - 2020
        m = fit("ko_loss ~ (prior_ko_losses + prior_other_losses) * bare + bare * "
                "(fight_no + log_layoff + prior_win_rate + big + era)", d, groups="fighter")
        print(f"\n{tag}: {len(d)} строк, {int(d.ko_loss.sum())} нокаутов "
              f"(кулачка {len(B)}/{int(B.ko_loss.sum())}, перчатки {len(uu)}/{int(uu.ko_loss.sum())})")
        for name, lab in (("prior_ko_losses", "нокауты, в перчатках"),
                          ("prior_ko_losses:bare", "насколько сильнее на голых кулаках"),
                          ("prior_other_losses", "контроль, в перчатках"),
                          ("prior_other_losses:bare", "контроль, отличие кулачки")):
            o, lo, hi, p = hr(m, name)
            print(f"  {lab:<38} {o:.3f} [{lo:.3f}–{hi:.3f}]  p={p:.4f}")
            OUT[f"pool/{tag}/{name}"] = {"est": o, "lo": lo, "hi": hi, "p": p}


def main() -> None:
    exposure()
    print("\n" + "=" * 78)
    print("ПОДПИСЬ ПОВТОРНОГО НОКАУТА: ОДНА МОДЕЛЬ НА ДВУХ ВИДАХ")
    compare()
    (DATA / "compare.json").write_text(json.dumps(OUT, ensure_ascii=False, indent=1))
    print(f"\nчисла записаны в {DATA/'compare.json'}")


if __name__ == "__main__":
    main()
