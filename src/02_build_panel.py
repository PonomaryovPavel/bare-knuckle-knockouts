"""Панель «боец-бой» по BKFC и оценка того, что на ней вообще можно померить."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("data")


def main() -> None:
    b = pd.read_parquet(OUT / "bouts.parquet")
    b = b[b.date.notna()].copy()
    b["bout_id"] = b.event.str.cat(b.winner.str.cat(b.loser, sep=" vs "), sep=" | ")

    rows = []
    for side, other, won in (("winner", "loser", True), ("loser", "winner", False)):
        d = b[["bout_id", "event", "date", "division", "weight_class", "kind", "round",
               "seconds", "decided", "winner_assumed", side, other]].copy()
        d = d.rename(columns={side: "fighter", other: "opponent"})
        d["won"] = won
        rows.append(d)
    f = pd.concat(rows, ignore_index=True)

    f["ko_loss"] = f.decided & ~f.won & f.kind.isin(["ko", "tko"])
    f["ko_win"] = f.decided & ~f.ko_loss & f.won & f.kind.isin(["ko", "tko"])
    f["clean_loss"] = f.decided & ~f.won
    f["other_loss"] = f.clean_loss & ~f.ko_loss

    f = f.sort_values(["fighter", "date", "bout_id"]).reset_index(drop=True)
    g = f.groupby("fighter", sort=False)
    f["fight_no"] = g.cumcount() + 1
    for src, dst in (("ko_loss", "prior_ko_losses"), ("other_loss", "prior_other_losses"),
                     ("won", "prior_wins"), ("ko_win", "prior_ko_wins")):
        v = f[src].astype("int32")
        f[dst] = v.groupby(f.fighter, sort=False).cumsum() - v
    v = f.seconds.fillna(0)
    f["cum_seconds"] = v.groupby(f.fighter, sort=False).cumsum() - v
    f["prev_date"] = g["date"].shift(1)
    f["layoff"] = (f.date - f.prev_date).dt.days
    f["age_proxy"] = np.nan          # даты рождения в источнике нет

    f.to_parquet(OUT / "panel_all.parquet", index=False)

    p = f[(f.fight_no > 1) & f.decided].copy()
    print(f"строк боец-бой: {len(f)}, бойцов: {f.fighter.nunique()}, боёв: {f.bout_id.nunique()}")
    print(f"группа риска (есть предыдущий бой в BKFC, исход однозначен): {len(p)}, "
          f"бойцов {p.fighter.nunique()}, нокаутов {int(p.ko_loss.sum())} "
          f"({100*p.ko_loss.mean():.1f}% на бой)")
    print("\nсколько боёв провёл боец в BKFC:")
    print(g.size().value_counts().sort_index().head(10).to_string())
    print(f"медиана {g.size().median():.0f}, максимум {g.size().max()}")
    print("\nриск нокаута по числу прошлых нокаутных поражений:")
    q = p.assign(c=p.prior_ko_losses.clip(upper=2))
    print(q.groupby("c").ko_loss.agg(n="size", ko="sum",
          risk=lambda s: round(100*s.mean(), 1)).to_string())
    print("\nто же по числу прочих поражений (контроль):")
    q2 = p.assign(c=p.prior_other_losses.clip(upper=2))
    print(q2.groupby("c").ko_loss.agg(n="size", ko="sum",
          risk=lambda s: round(100*s.mean(), 1)).to_string())

    # какой эффект эта выборка в принципе способна заметить
    n, k = len(p), int(p.ko_loss.sum())
    pbar = k / n
    sd = p.prior_ko_losses.std()
    se = 1 / np.sqrt(n * pbar * (1 - pbar) * sd ** 2)
    print(f"\nоценка мощности: n={n}, событий={k}, разброс счётчика sd={sd:.2f}")
    print(f"  ожидаемая стандартная ошибка логарифма ОШ ≈ {se:.4f}")
    print(f"  при 80% мощности различим эффект от ОШ {np.exp(2.80*se):.2f}")
    print(f"  в UFC найдено 1.21 — {'хватает' if np.exp(2.80*se) <= 1.21 else 'НЕ хватает'}")
    p.to_parquet(OUT / "panel.parquet", index=False)


if __name__ == "__main__":
    main()
