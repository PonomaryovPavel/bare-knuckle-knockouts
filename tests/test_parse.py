"""Каждый тест закрывает дефект, который реально был при разборе источника."""
import importlib.util
import json
import re
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("parse", ROOT / "src" / "01_parse_wikitext.py")
P = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(P)


@pytest.fixture(scope="module")
def bouts():
    return pd.read_parquet(ROOT / "data" / "bouts.parquet")


@pytest.fixture(scope="module")
def panel():
    return pd.read_parquet(ROOT / "data" / "panel.parquet")


def test_clean_strips_markup():
    assert P.clean("{{flagicon|USA}} [[Robbie Peralta]]") == "Robbie Peralta"
    assert P.clean("[[Seminole Hard Rock|Hard Rock]]") == "Hard Rock"
    assert P.clean("Heavyweight 120&nbsp;kg") == "Heavyweight 120 kg"
    assert P.clean("align=center|def.") == "def."
    assert P.clean("<ref name='x'/>Welterweight") == "Welterweight"


def test_anchor_and_display_text_give_the_same_key():
    """Половина ссылок в списке турниров записана через подчёркивания,
    и подпись ссылки не всегда совпадает с якорем."""
    assert P.key("BKFC_63_Sturgis:_Hart_vs._Starling") == P.key("BKFC 63 Sturgis: Hart vs. Starling")


def test_both_date_formats_are_read():
    """Даты записаны то шаблоном {{dts}}, то обычным текстом."""
    row = ("\n|-\n| 1\n| [[#A|A]]\n| {{dts|2024|May|31}}\n|-\n|2\n|[[#B|B]]\n|June 8, 2024\n")
    got = P.event_dates("==List of events==" + row + "\n==X==\n")
    assert got[P.key("A")] == "2024-05-31"
    assert got[P.key("B")] == "2024-06-08"


def test_every_bout_has_a_date_or_is_a_known_gap(bouts):
    missing = bouts[bouts.date.isna()]
    assert len(missing) / len(bouts) < 0.02, f"без даты {len(missing)} из {len(bouts)}"


def test_dates_run_forward_within_a_year(bouts):
    d = bouts.dropna(subset=["date"])
    for year, g in d.groupby("year"):
        assert g.date.is_monotonic_increasing, year


def test_time_never_exceeds_the_round(bouts):
    """В источнике встречается «5:00» при двухминутном раунде — опечатка,
    которая иначе ушла бы в расчёт экспозиции."""
    assert (bouts.time_s <= P.ROUND_SECONDS).all()
    assert bouts.time_fixed.sum() > 0, "проверка потеряла смысл, опечаток больше нет"


def test_vs_separator_is_not_only_draws(bouts):
    """Разделителем «vs.» записаны и обычные нокауты, поэтому победителя
    приходится брать по порядку записи — такие строки помечены."""
    assert bouts.winner_assumed.sum() > 0
    assert bouts.loc[bouts.winner_assumed, "kind"].isin(["ko", "tko", "doctor", "decision", "dq"]).all()
    assert bouts.winner_assumed.mean() < 0.05


def test_unfinished_bouts_are_dropped(bouts):
    """Отменённые и нетранслируемые бои идут без раунда и времени."""
    assert bouts["round"].notna().all() and bouts.time_s.notna().all()
    assert not bouts.event.str.contains("cancel", case=False).any()


def test_every_division_is_recognised(bouts):
    assert bouts.division.ne("Unknown").all()
    assert "Ironweight" in set(bouts.division), "категория Ironweight пропала из данных"


def test_exposure_is_consistent_with_round_and_time(bouts):
    expected = (bouts["round"] - 1) * P.ROUND_SECONDS + bouts.time_s
    assert (bouts.seconds - expected).abs().max() < 1e-9
    assert bouts.seconds.max() <= 6 * P.ROUND_SECONDS


def test_no_duplicated_bouts(bouts):
    """Один бой в статье бывает вписан дважды — в основной кард и в прелиминари."""
    assert not bouts.duplicated(["event", "winner", "loser", "method_raw",
                                 "round_raw", "time_raw"]).any()


def test_two_rows_per_bout_in_the_panel():
    f = pd.read_parquet(ROOT / "data" / "panel_all.parquet")
    assert f.groupby("bout_id").size().eq(2).all()
    assert not f.duplicated(["bout_id", "fighter"]).any()


def test_history_excludes_the_current_bout():
    f = pd.read_parquet(ROOT / "data" / "panel_all.parquet")
    first = f.sort_values(["fighter", "date"]).groupby("fighter").head(1)
    assert (first.prior_ko_losses == 0).all()
    assert (first.cum_seconds == 0).all()


def test_panel_only_has_repeat_fighters(panel):
    assert (panel.fight_no >= 2).all()
    assert panel.decided.all()


def test_figures_have_valid_coordinates():
    """Замена точки на запятую однажды прошла по всей разметке и сломала
    координаты: числа вида y="141,2" SVG не понимает."""
    import xml.etree.ElementTree as ET

    for svg in sorted((ROOT / "figures").glob("*.svg")):
        ET.parse(svg)
        text = svg.read_text()
        assert not re.search(r'\b(x|y|cx|cy|width|height)="[-\d]+,[\d]', text), svg.name
