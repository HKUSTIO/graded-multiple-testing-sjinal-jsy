from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import t


def _two_sample_t_pvalue(y: np.ndarray, z: np.ndarray) -> float:
    treated = y[z == 1]
    control = y[z == 0]
    n1 = treated.shape[0]
    n0 = control.shape[0]
    s1 = float(np.var(treated, ddof=1))
    s0 = float(np.var(control, ddof=1))
    se = float(np.sqrt(s1 / n1 + s0 / n0))
    diff = float(np.mean(treated) - np.mean(control))
    if se == 0.0:
        return 1.0
    t_stat = diff / se
    df_num = (s1 / n1 + s0 / n0) ** 2
    df_den = ((s1 / n1) ** 2) / (n1 - 1) + ((s0 / n0) ** 2) / (n0 - 1)
    if df_den == 0.0:
        return 1.0
    df = df_num / df_den
    return float(2.0 * t.sf(np.abs(t_stat), df=df))


def simulate_null_pvalues(config: dict[str, Any]) -> pd.DataFrame:
    """
    Generate p-values under the complete null for L simulations.
    Return columns: sim_id, hypothesis_id, p_value.
    """
    rng = np.random.default_rng(int(config["seed_null"]))
    n = int(config["N"])
    m = int(config["M"])
    l = int(config["L"])
    p_treat = float(config["p_treat"])

    rows: list[dict[str, float | int]] = []
    for sim_id in range(l):
        z = (rng.random(n) < p_treat).astype(int)
        for hypothesis_id in range(m):
            y = rng.normal(loc=0.0, scale=1.0, size=n)
            p_value = _two_sample_t_pvalue(y=y, z=z)
            rows.append(
                {
                    "sim_id": sim_id,
                    "hypothesis_id": hypothesis_id,
                    "p_value": p_value,
                }
            )
    return pd.DataFrame(rows)


def simulate_mixed_pvalues(config: dict[str, Any]) -> pd.DataFrame:
    """
    Generate p-values under mixed true and false null hypotheses for L simulations.
    Return columns: sim_id, hypothesis_id, p_value, is_true_null.
    """
    rng = np.random.default_rng(int(config["seed_mixed"]))
    n = int(config["N"])
    m = int(config["M"])
    m0 = int(config["M0"])
    l = int(config["L"])
    p_treat = float(config["p_treat"])
    tau_alt = float(config["tau_alternative"])

    rows: list[dict[str, float | int | bool]] = []
    for sim_id in range(l):
        z = (rng.random(n) < p_treat).astype(int)
        for hypothesis_id in range(m):
            is_true_null = hypothesis_id >= (m - m0)
            effect = 0.0 if is_true_null else tau_alt
            y = rng.normal(loc=0.0, scale=1.0, size=n) + effect * z
            p_value = _two_sample_t_pvalue(y=y, z=z)
            rows.append(
                {
                    "sim_id": sim_id,
                    "hypothesis_id": hypothesis_id,
                    "p_value": p_value,
                    "is_true_null": is_true_null,
                }
            )
    return pd.DataFrame(rows)


def _uncorrected_rejections(p_values: np.ndarray, alpha: float) -> np.ndarray:
    """Reject when p <= alpha."""
    return p_values <= alpha


def bonferroni_rejections(p_values: np.ndarray, alpha: float) -> np.ndarray:
    """
    Return boolean rejection decisions under Bonferroni correction.
    """
    m = len(p_values)
    return p_values <= alpha / m


def holm_rejections(p_values: np.ndarray, alpha: float) -> np.ndarray:
    """
    Return boolean rejection decisions under Holm step-down correction.
    """
    m = len(p_values)
    order = np.argsort(p_values)
    sorted_p = p_values[order]
    reject_sorted = np.zeros(m, dtype=bool)
    for k in range(m):
        threshold = alpha / (m - k)
        if sorted_p[k] <= threshold:
            reject_sorted[k] = True
        else:
            break
    result = np.zeros(m, dtype=bool)
    result[order[: np.sum(reject_sorted)]] = True
    return result


def benjamini_hochberg_rejections(p_values: np.ndarray, alpha: float) -> np.ndarray:
    """
    Return boolean rejection decisions under Benjamini-Hochberg correction.
    """
    m = len(p_values)
    order = np.argsort(p_values)
    sorted_p = p_values[order]
    reject_sorted = np.zeros(m, dtype=bool)
    for k in range(m - 1, -1, -1):
        threshold = (k + 1) / m * alpha
        if sorted_p[k] <= threshold:
            reject_sorted[: k + 1] = True
            break
    result = np.zeros(m, dtype=bool)
    result[order[reject_sorted]] = True
    return result


def benjamini_yekutieli_rejections(p_values: np.ndarray, alpha: float) -> np.ndarray:
    """
    Return boolean rejection decisions under Benjamini-Yekutieli correction.
    """
    m = len(p_values)
    harmonic = np.sum(1.0 / np.arange(1, m + 1))
    order = np.argsort(p_values)
    sorted_p = p_values[order]
    reject_sorted = np.zeros(m, dtype=bool)
    for k in range(m - 1, -1, -1):
        threshold = (k + 1) / m * alpha / harmonic
        if sorted_p[k] <= threshold:
            reject_sorted[: k + 1] = True
            break
    result = np.zeros(m, dtype=bool)
    result[order[reject_sorted]] = True
    return result


def compute_fwer(rejections_null: np.ndarray) -> float:
    """
    Return family-wise error rate from a [L, M] rejection matrix under the complete null.
    """
    at_least_one = np.any(rejections_null, axis=1)
    return float(np.mean(at_least_one))


def compute_fdr(rejections: np.ndarray, is_true_null: np.ndarray) -> float:
    """
    Return FDR for one simulation: false discoveries among all discoveries.
    Use 0.0 when there are no rejections.
    """
    n_rejections = np.sum(rejections)
    if n_rejections == 0:
        return 0.0
    false_discoveries = np.sum(rejections & is_true_null)
    return float(false_discoveries / n_rejections)


def compute_power(rejections: np.ndarray, is_true_null: np.ndarray) -> float:
    """
    Return power for one simulation: true rejections among false null hypotheses.
    """
    n_false_nulls = np.sum(~is_true_null)
    if n_false_nulls == 0:
        return 0.0
    true_rejections = np.sum(rejections & ~is_true_null)
    return float(true_rejections / n_false_nulls)


def summarize_multiple_testing(
    null_pvalues: pd.DataFrame,
    mixed_pvalues: pd.DataFrame,
    alpha: float,
) -> dict[str, float]:
    """
    Return summary metrics:
      fwer_uncorrected, fwer_bonferroni, fwer_holm,
      fdr_uncorrected, fdr_bh, fdr_by,
      power_uncorrected, power_bh, power_by.
    """
    l_null = null_pvalues["sim_id"].nunique()
    m_null = null_pvalues["hypothesis_id"].nunique()

    reject_uncorr = np.zeros((l_null, m_null), dtype=bool)
    reject_bonf = np.zeros((l_null, m_null), dtype=bool)
    reject_holm = np.zeros((l_null, m_null), dtype=bool)

    for sim_id, group in null_pvalues.groupby("sim_id"):
        pvals = group["p_value"].values
        reject_uncorr[sim_id] = _uncorrected_rejections(pvals, alpha)
        reject_bonf[sim_id] = bonferroni_rejections(pvals, alpha)
        reject_holm[sim_id] = holm_rejections(pvals, alpha)

    fwer_uncorrected = compute_fwer(reject_uncorr)
    fwer_bonferroni = compute_fwer(reject_bonf)
    fwer_holm = compute_fwer(reject_holm)

    fdr_uncorr_list = []
    fdr_bh_list = []
    fdr_by_list = []
    power_uncorr_list = []
    power_bh_list = []
    power_by_list = []

    for sim_id, group in mixed_pvalues.groupby("sim_id"):
        pvals = group["p_value"].values
        is_true_null = group["is_true_null"].values

        rej_uncorr = _uncorrected_rejections(pvals, alpha)
        rej_bh = benjamini_hochberg_rejections(pvals, alpha)
        rej_by = benjamini_yekutieli_rejections(pvals, alpha)

        fdr_uncorr_list.append(compute_fdr(rej_uncorr, is_true_null))
        fdr_bh_list.append(compute_fdr(rej_bh, is_true_null))
        fdr_by_list.append(compute_fdr(rej_by, is_true_null))
        power_uncorr_list.append(compute_power(rej_uncorr, is_true_null))
        power_bh_list.append(compute_power(rej_bh, is_true_null))
        power_by_list.append(compute_power(rej_by, is_true_null))

    return {
        "fwer_uncorrected": fwer_uncorrected,
        "fwer_bonferroni": fwer_bonferroni,
        "fwer_holm": fwer_holm,
        "fdr_uncorrected": float(np.mean(fdr_uncorr_list)),
        "fdr_bh": float(np.mean(fdr_bh_list)),
        "fdr_by": float(np.mean(fdr_by_list)),
        "power_uncorrected": float(np.mean(power_uncorr_list)),
        "power_bh": float(np.mean(power_bh_list)),
        "power_by": float(np.mean(power_by_list)),
    }
