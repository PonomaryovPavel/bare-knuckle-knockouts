"""Разбор карточек боёв BKFC из викитекста годовых статей английской Википедии.

Источник: en.wikipedia.org, API action=parse&prop=wikitext, лицензия CC BY-SA.
Выгрузка лежит в data_raw и дальше не трогается, чтобы прогон был повторяемым.

Что здесь важно знать про источник. Победитель отделён от проигравшего словом
«def.», но часть строк записана через «vs.» — и это не только ничьи: так
оформлены и обычные нокауты. Для таких строк победителем считается тот, кто
записан первым, и строка помечается: её можно исключить отдельной проверкой.
Бои без раунда и времени — отменённые или нетранслируемые, они выбрасываются.
"""
from __future__ import annotations

import html
import json
import re
from difflib import get_close_matches
from pathlib import Path

import pandas as pd

RAW, OUT = Path("data_raw"), Path("data")
YEARS = range(2018, 2027)
MONTHS = ("January February March April May June July August September "
          "October November December").split()
ROUND_SECONDS = 120          # регламент BKFC: раунд две минуты
DIVISIONS = ("Strawweight", "Flyweight", "Bantamweight", "Featherweight", "Lightweight",
             "Welterweight", "Middleweight", "Light Heavyweight", "Cruiserweight",
             "Ironweight", "Heavyweight", "Catchweight", "Openweight")


def clean(s: str) -> str:
    """Снять разметку: сноски, флаги, ссылки, выравнивание, сущности."""
    s = re.sub(r"<ref[^>]*/>", "", s)
    s = re.sub(r"<ref.*?</ref>", "", s, flags=re.S)
    s = re.sub(r"\{\{\s*flagicon[^}]*\}\}", "", s, flags=re.I)
    s = re.sub(r"\{\{\s*small\s*\|([^}]*)\}\}", r"\1", s, flags=re.I)
    s = re.sub(r"\[\[[^\]|]*\|([^\]]*)\]\]", r"\1", s)
    s = re.sub(r"\[\[([^\]]*)\]\]", r"\1", s)
    s = re.sub(r"<[^>]+>", "", s)
    s = html.unescape(s).replace(" ", " ")
    s = s.replace("'''", "").replace("''", "").replace("|}", " ")
    s = re.sub(r"^\s*(align|style|colspan|rowspan|scope|bgcolor)\s*=\s*[^|]*\|", "", s)
    return re.sub(r"\s+", " ", s).strip()


def key(s: str) -> str:
    """Ключ для сопоставления названий турнира: якоря пишут через подчёркивания."""
    return re.sub(r"[^a-z0-9]", "", clean(s).replace("_", " ").lower())


def event_dates(text: str) -> dict[str, str]:
    out = {}
    block = text.split("==List of events==", 1)
    block = (block[1] if len(block) > 1 else text).split("\n==", 1)[0]
    for row in block.split("\n|-"):
        # ссылка вида [[#Якорь|Подпись]]: заголовок секции совпадает то с одним,
        # то с другим, поэтому дата кладётся под оба ключа
        link = re.search(r"\[\[#([^|\]]+)(?:\|([^\]]+))?\]\]", row)
        if not link:
            continue
        # даты записаны двумя способами: шаблоном {{dts|2024|May|31}} и текстом «June 8, 2024»
        tpl = re.search(r"\{\{dts\|(\d{4})\|(\w+)\|(\d{1,2})\}\}", row)
        txt = re.search(r"\n\|\s*(" + "|".join(MONTHS) + r")\s+(\d{1,2}),?\s+(\d{4})", row)
        if tpl and tpl.group(2) in MONTHS:
            y, mon, d = tpl.groups()
        elif txt:
            mon, d, y = txt.groups()
        else:
            continue
        iso = f"{y}-{MONTHS.index(mon)+1:02d}-{int(d):02d}"
        for part in link.groups():
            if part:
                out[key(part)] = iso
    return out


def split_events(text: str) -> list[tuple[str, str]]:
    body = text.split("==List of events==", 1)
    body = body[1] if len(body) > 1 else text
    parts = re.split(r"\n==\s*(?!=)(.+?)\s*==\s*\n", body)
    return [(clean(parts[i]), parts[i + 1]) for i in range(1, len(parts) - 1, 2)]


def parse_card(section: str) -> list[list[str]]:
    rows = []
    for tbl in re.findall(r"\{\|.*?\n\|\}", section, flags=re.S):
        if "def." not in tbl and "vs." not in tbl:
            continue
        for chunk in re.split(r"\n\|-", tbl):
            cells = [clean(c) for c in re.split(r"\n\|(?!\})", chunk)[1:]]
            if len(cells) >= 7 and cells[2] in ("def.", "vs."):
                rows.append((cells + [""] * 8)[:8])
    return rows


def division(w: str) -> str:
    for d in DIVISIONS:
        if d.lower() in w.lower():
            return d
    return "Unknown"


def method_kind(m: str) -> str:
    low = m.lower()
    if low.startswith("ko"):
        return "ko"
    if "doctor" in low or "cut" in low:
        return "doctor"
    if low.startswith("tko") or low.startswith("rtd") or "retirement" in low or "corner" in low:
        return "tko"
    if "decision" in low:
        return "decision"
    if "draw" in low:
        return "draw"
    if "no contest" in low or low.startswith("nc"):
        return "nc"
    if low.startswith("dq") or "disqualification" in low:
        return "dq"
    return "other"


def mmss(s: str) -> float:
    m = re.match(r"^\s*(\d+):(\d{1,2})\s*$", s)
    return int(m.group(1)) * 60 + int(m.group(2)) if m else float("nan")


def main() -> None:
    OUT.mkdir(exist_ok=True)
    rows, checks, unmatched, fuzzy = [], [], [], []
    for year in YEARS:
        path = RAW / f"bkfc_{year}.json"
        if not path.exists():
            continue
        text = json.loads(path.read_text())["parse"]["wikitext"]
        dates = event_dates(text)
        n_ev = 0
        for name, section in split_events(text):
            card = parse_card(section)
            if not card:
                continue
            n_ev += 1
            date = dates.get(key(name))
            if date is None:
                # якорь и заголовок иногда расходятся («BKFC 67» против «BKFC 67 Denver»)
                near = get_close_matches(key(name), list(dates), n=2, cutoff=0.88)
                if len(near) == 1:
                    date = dates[near[0]]
                    fuzzy.append((year, name, near[0]))
                else:
                    unmatched.append((year, name))
            for wc, a, sep, b, method, rnd, tm, notes in card:
                rows.append({"year": year, "event": name, "date": date,
                             "weight_class": wc, "first": a, "second": b, "sep": sep,
                             "method_raw": method, "round_raw": rnd, "time_raw": tm,
                             "notes": notes})
        declared = re.search(r"\|\s*total_events\s*=\s*(\d+)", text)
        checks.append({"year": year, "events_parsed": n_ev,
                       "events_declared": int(declared.group(1)) if declared else None})

    d = pd.DataFrame(rows)
    d["division"] = d.weight_class.map(division)
    d["kind"] = d.method_raw.map(method_kind)
    d["round"] = pd.to_numeric(d.round_raw, errors="coerce")
    d["time_s"] = d.time_raw.map(mmss)
    d["cancelled"] = d.event.str.contains("cancel|postpon", case=False)
    n_all = len(d)
    # бой без раунда или времени не состоялся либо его результат не записан
    d = d[~d.cancelled & d["round"].notna() & d.time_s.notna()].copy()
    d["date"] = pd.to_datetime(d.date)
    # в источнике встречается время вроде «5:00» при двухминутном раунде —
    # это опечатка, время приводится к концу раунда и строка помечается
    d["time_fixed"] = d.time_s > ROUND_SECONDS
    d.loc[d.time_fixed, "time_s"] = ROUND_SECONDS
    d["seconds"] = (d["round"] - 1) * ROUND_SECONDS + d.time_s
    d["decided"] = d.kind.isin(["ko", "tko", "doctor", "decision", "dq"])
    d["winner_assumed"] = d.sep.eq("vs.") & d.decided
    d["winner"] = d["first"]
    d["loser"] = d["second"]

    # редакторы иногда вписывают один бой дважды — в основной кард и в прелиминари
    dup_keys = ["event", "first", "second", "method_raw", "round_raw", "time_raw"]
    n_dup = int(d.duplicated(dup_keys).sum())
    d = d.drop_duplicates(dup_keys)

    d = d.sort_values(["date", "event"]).reset_index(drop=True)
    d.to_parquet(OUT / "bouts.parquet", index=False)

    ch = pd.DataFrame(checks)
    print(ch.to_string(index=False))
    print(f"\nдублей одного боя в статье: {n_dup}")
    print(f"строк разобрано: {n_all}, осталось после отсева: {len(d)} "
          f"(отменённые и без результата: {n_all - len(d)})")
    print(f"турниров: {d.event.nunique()}, период: {d.date.min().date()} — {d.date.max().date()}")
    print(f"без даты турнира: {int(d.date.isna().sum())}  (несопоставленных названий: {len(unmatched)})")
    for u in unmatched[:5]:
        print("   ", u)
    ok_order = all(g.dropna(subset=["date"]).date.is_monotonic_increasing
                   for _, g in d.groupby("year"))
    print(f"названия сведены по близости: {len(fuzzy)}")
    for f in fuzzy:
        print("   ", f)
    print(f"порядок дат внутри года не нарушен: {ok_order}")
    print(f"время исправлено как опечатка источника: {int(d.time_fixed.sum())}")
    print(f"дивизион Unknown: {int(d.division.eq('Unknown').sum())}")
    print(f"победитель определён по порядку записи: {int(d.winner_assumed.sum())} "
          f"({100*d.winner_assumed.mean():.1f}%)")
    print("\nисходы:")
    print(d.kind.value_counts().to_string())


if __name__ == "__main__":
    main()
