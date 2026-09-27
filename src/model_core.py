"""Обёртки над statsmodels с защитой от булевых столбцов.

patsy отдаёт булев отклик матрицей из двух колонок, GLM читает её как
(успехи, неудачи) и оценивает P(False) — все коэффициенты переворачиваются.
Поэтому любой булев столбец приводится к целому до сборки матриц.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.discrete.conditional_models import ConditionalLogit


def numeric_bools(data: pd.DataFrame) -> pd.DataFrame:
    """Копия кадра, где каждый булев столбец стал 0/1."""
    cols = [c for c in data.columns if pd.api.types.is_bool_dtype(data[c])]
    return data.assign(**{c: data[c].astype("int8") for c in cols}) if cols else data


def _check_single_column_response(formula: str, data: pd.DataFrame) -> None:
    y, _ = patsy.dmatrices(formula, data, return_type="dataframe")
    if y.shape[1] != 1:
        raise ValueError(
            f"отклик '{formula.split('~')[0].strip()}' раскрылся в {y.shape[1]} колонок "
            f"({list(y.columns)}) — GLM прочтёт их как (успехи, неудачи)"
        )


def fit(formula: str, data: pd.DataFrame, groups: str | None = None, family=None):
    """GLM с биномиальным семейством; groups — столбец для кластерных ошибок."""
    d = numeric_bools(data)
    _check_single_column_response(formula, d)
    m = smf.glm(formula, data=d, family=family or sm.families.Binomial())
    if groups is None:
        return m.fit()
    return m.fit(cov_type="cluster", cov_kwds={"groups": d[groups]})


def fit_conditional(formula: str, data: pd.DataFrame, strata: str):
    """Условная логистическая регрессия со стратами (эффекты внутри бойца)."""
    d = numeric_bools(data)
    y, X = patsy.dmatrices(formula, d, return_type="dataframe")
    if y.shape[1] != 1:
        raise ValueError(f"отклик раскрылся в {y.shape[1]} колонок: {list(y.columns)}")
    X = X.drop(columns=[c for c in X.columns if c == "Intercept"])
    g = d.loc[X.index, strata]
    return ConditionalLogit(y.iloc[:, 0].to_numpy(), X, groups=g.to_numpy()).fit(disp=False), list(X.columns)


def predict(res, data: pd.DataFrame):
    """Предсказание с тем же приведением булевых столбцов, что и при подгонке."""
    return res.predict(numeric_bools(data))


def ci(res, name: str, level: float = 0.95) -> tuple[float, float]:
    lo, hi = res.conf_int(alpha=1 - level).loc[name]
    return float(lo), float(hi)


def hr(res, name: str) -> tuple[float, float, float, float]:
    """Отношение шансов/рисков и границы, плюс p."""
    lo, hi = ci(res, name)
    p = float(res.pvalues[name])
    return float(np.exp(res.params[name])), float(np.exp(lo)), float(np.exp(hi)), p


def holm(pairs: list[tuple[str, float]]) -> list[tuple[str, float, float]]:
    """Поправка Холма; возвращает (имя, сырое p, скорректированное p)."""
    order = sorted(pairs, key=lambda t: t[1])
    m, out, running = len(order), [], 0.0
    for i, (name, p) in enumerate(order):
        running = max(running, min(1.0, (m - i) * p))
        out.append((name, p, running))
    return out
