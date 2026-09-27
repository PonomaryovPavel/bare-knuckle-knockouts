"""Повторные нокауты на голых кулаках: та же конструкция, что и в UFC."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model_core import fit, holm, hr  # noqa: E402

DATA = Path("data")
RNG = np.random.default_rng(20260927)
R: dict = {}


def say(res, name: str, label: str, key: str, scale: str = "ОШ") -> None:
    o, lo, hi, p = hr(res, name)
    print(f"  {label:<44} {scale} {o:.3f} [{lo:.3f}–{hi:.3f}]  p={p:.4f}")
    R[key] = {"est": o, "lo": lo, "hi": hi, "p": p, "label": label}


def bootstrap(formula: str, d: pd.DataFrame, names: list[str], reps: int = 400) -> dict:
    pos = {f: np.flatnonzero((d.fighter == f).to_numpy()) for f in d.fighter.unique()}
    ids = np.array(list(pos))
    out = {n: [] for n in names}
    for _ in range(reps):
        idx = np.concatenate([pos[f] for f in RNG.choice(ids, size=len(ids), replace=True)])
        try:
            r = fit(formula, d.iloc[idx])
        except Exception:
            continue
        for n in names:
            if n in r.params.index:
                out[n].append(r.params[n])
    return {n: (float(np.exp(np.percentile(v, 2.5))), float(np.exp(np.percentile(v, 97.5))), len(v))
            for n, v in out.items() if v}


def main() -> None:
    p = pd.read_parquet(DATA / "panel.parquet")
    p["year"] = p.date.dt.year
    p["era"] = p.year - 2022
    p["log_layoff"] = np.log(p.layoff.clip(lower=7))
    p["opp_ko10"] = 10 * p.groupby("opponent").ko_win.transform("mean").fillna(0)
    p["prior_win_rate"] = p.prior_wins / (p.fight_no - 1)
    p["big"] = p.division.isin(["Light Heavyweight", "Cruiserweight", "Heavyweight", "Ironweight"])
    need = ["log_layoff", "prior_win_rate", "opp_ko10"]
    before = len(p)
    p = p.dropna(subset=need)
    CTRL = "fight_no + log_layoff + prior_win_rate + big + era"
    tests = []

    R["panel"] = {"fights": len(p), "fighters": int(p.fighter.nunique()),
                  "ko": int(p.ko_loss.sum()), "risk": float(p.ko_loss.mean()),
                  "dropped": before - len(p),
                  "year_min": int(p.year.min()), "year_max": int(p.year.max())}
    print(f"панель: {len(p)} боёв, {p.fighter.nunique()} бойцов, {int(p.ko_loss.sum())} нокаутов "
          f"({100*p.ko_loss.mean():.1f}% на бой), {int(p.year.min())}—{int(p.year.max())}, "
          f"отсеяно {before-len(p)}")

    print("\n1. СВОЙСТВО: накопленные нокауты, рядом отрицательный контроль")
    m1 = fit(f"ko_loss ~ prior_ko_losses + prior_other_losses + {CTRL}", p, groups="fighter")
    say(m1, "prior_ko_losses", "каждый пропущенный нокаут", "trait_ko")
    say(m1, "prior_other_losses", "каждое поражение не нокаутом", "trait_other")
    say(m1, "fight_no", "каждый бой стажа", "fight_no")
    say(m1, "big", "тяжёлые категории", "big")
    c = np.zeros(len(m1.params))
    c[list(m1.params.index).index("prior_ko_losses")] = 1
    c[list(m1.params.index).index("prior_other_losses")] = -1
    dd = float(c @ m1.params)
    se = float(np.sqrt(c @ m1.cov_params().to_numpy() @ c))
    R["swap"] = {"est": float(np.exp(dd)), "lo": float(np.exp(dd - 1.96 * se)),
                 "hi": float(np.exp(dd + 1.96 * se)), "p": float(2 * stats.norm.sf(abs(dd / se))),
                 "label": "замена поражения не нокаутом на нокаут"}
    print(f"  {'замена поражения не нокаутом на нокаут':<44} ОШ {R['swap']['est']:.3f} "
          f"[{R['swap']['lo']:.3f}–{R['swap']['hi']:.3f}]  p={R['swap']['p']:.4f}")
    tests += [("свойство: накопленные нокауты", R["trait_ko"]["p"]),
              ("контроль: поражения не нокаутом", R["trait_other"]["p"])]

    print("\n2. ПЕРВЫЙ БОЙ ПОСЛЕ ПОРАЖЕНИЯ")
    p = p.sort_values(["fighter", "date"])
    pr = p.groupby("fighter")
    p["prev_kind"] = pr["kind"].shift(1)
    p["prev_won"] = pr["won"].shift(1)
    p["prev_decided"] = pr["decided"].shift(1)
    ret = p[(p.prev_decided == True) & (p.prev_won == False)].copy()
    ret["prev_ko"] = ret.prev_kind.isin(["ko", "tko"])
    for lab, sel in (("после нокаута", ret.prev_ko), ("после иного поражения", ~ret.prev_ko)):
        s = ret[sel]
        print(f"  {lab:<28} {len(s):4} боёв, риск {100*s.ko_loss.mean():.1f}% "
              f"({int(s.ko_loss.sum())} нокаутов), медианная пауза {s.layoff.median():.0f} дней")
        R[f"raw_{'ko' if 'нокаута' in lab else 'oth'}"] = {
            "n": len(s), "ko": int(s.ko_loss.sum()), "risk": float(s.ko_loss.mean()),
            "layoff_med": float(s.layoff.median())}
    m2 = fit(f"ko_loss ~ prev_ko + {CTRL}", ret, groups="fighter")
    say(m2, "prev_ko", "нокаут против иного поражения", "ret_ko")
    m2b = fit(f"ko_loss ~ prev_ko + prior_ko_losses + prior_other_losses + {CTRL}",
              ret, groups="fighter")
    say(m2b, "prev_ko", "то же при равном числе прошлых нокаутов", "ret_ko_adj")
    tests.append(("возвращение после нокаута", R["ret_ko"]["p"]))

    print("\n3. ПАУЗА")
    m3 = fit(f"ko_loss ~ prev_ko * log_layoff + fight_no + prior_win_rate + big + era",
             ret, groups="fighter")
    say(m3, "prev_ko:log_layoff", "удвоение паузы после нокаута", "layoff_x")
    for lim in (45, 60, 90, 180):
        R[f"ko_return_under_{lim}"] = float((ret.loc[ret.prev_ko, "layoff"] < lim).mean())
    print(f"  возвращений после нокаута быстрее 45 дней: {100*R['ko_return_under_45']:.1f}%, "
          f"60: {100*R['ko_return_under_60']:.1f}%, 90: {100*R['ko_return_under_90']:.1f}%")
    tests.append(("пауза после нокаута", R["layoff_x"]["p"]))

    print("\n4. УСТОЙЧИВОСТЬ")
    for tag, d in (("вся панель", p), ("без спорных победителей", p[~p.winner_assumed]),
                   ("исход шире: плюс остановки врача", p.assign(
                       ko_loss=p.decided & ~p.won & p.kind.isin(["ko", "tko", "doctor"]))),
                   ("только 2022 и позже", p[p.year >= 2022]),
                   ("тяжёлые категории", p[p.big]), ("остальные категории", p[~p.big]),
                   ("с третьего боя в BKFC", p[p.fight_no >= 3])):
        ctrl = CTRL.replace(" + big", "") if "категории" in tag else CTRL
        try:
            r = fit(f"ko_loss ~ prior_ko_losses + prior_other_losses + {ctrl}", d, groups="fighter")
            o, lo, hi, pv = hr(r, "prior_ko_losses")
            print(f"  {tag:<36} ОШ {o:.3f} [{lo:.3f}–{hi:.3f}]  n={len(d)}, нокаутов={int(d.ko_loss.sum())}")
            R[f"sens/{tag}"] = {"est": o, "lo": lo, "hi": hi, "p": pv, "n": len(d)}
        except Exception as e:
            print(f"  {tag:<36} не сошлось: {type(e).__name__}")

    print("\n5. БУТСТРЭП ПО КАРЬЕРАМ")
    R["boot"] = {}
    for n, (lo, hi, k) in bootstrap(f"ko_loss ~ prior_ko_losses + prior_other_losses + {CTRL}",
                                    p, ["prior_ko_losses", "prior_other_losses"]).items():
        print(f"  {n:<24} [{lo:.3f}–{hi:.3f}]  ({k} повторов)")
        R["boot"][n] = {"lo": lo, "hi": hi, "reps": k}

    print("\n6. ПОПРАВКА ХОЛМА")
    R["holm"] = []
    for name, raw, adj in holm(tests):
        print(f"  {name:<40} p={raw:.4f}  после Холма {adj:.4f}")
        R["holm"].append({"name": name, "p": raw, "adj": adj})

    (DATA / "results.json").write_text(json.dumps(R, ensure_ascii=False, indent=1))
    print(f"\nчисла записаны в {DATA/'results.json'}")


if __name__ == "__main__":
    main()
