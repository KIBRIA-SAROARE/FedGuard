"""Causal release times for the notebook's CSEC signals.

nb.csec_signals() assigns a corroboration value to every sample of a voltage event, using the event's
peak depth and the peers' events whose onsets lie within +/-CSEC_DTALIGN. In a live site these
quantities are known only after (i) the site's own event has ended, (ii) CSEC_DTALIGN has elapsed after
the onset (no later peer can still qualify), and (iii) each matched peer's event has ended and its
15-byte descriptor has arrived. avail[i] is the first sample index at which the CSEC value of sample i is
known. Samples outside any event have value 0 and are known immediately. Scores are unchanged; a window
decision is released once every sample in it is known, so CSEC appears as decision delay.
net_delay_samples models a fixed descriptor transport delay (0 = ideal link, stated as an assumption).
"""
import numpy as np

from . import nb


def csec_with_release(tels, cals, k, net_delay_samples=0, return_clipped=False):
    corr, unc, log = nb.csec_signals(tels, cals)
    T = nb.tvec()
    n = len(T)
    E = {j: nb.detect_events(tels[j], j, cals[j])[0] for j in tels}
    avail = np.arange(n, dtype=np.int64)
    clipped = np.zeros(n, bool)
    align = nb.CFG["CSEC_DTALIGN"]
    for e in E[k]:
        rel = max(e["idx1"], int(np.searchsorted(T, e["t_onset"] + align, side="right")))
        for j in sorted(tels):
            if j == k:
                continue
            for f in E[j]:
                if abs(f["t_onset"] - e["t_onset"]) <= align:
                    rel = max(rel, f["idx1"] + net_delay_samples)
                    break
        if rel > n - 1:                      # REV (A9): release clipped to the end of the record
            clipped[e["idx0"]:e["idx1"]] = True
        rel = min(rel, n - 1)
        avail[e["idx0"]:e["idx1"]] = np.maximum(avail[e["idx0"]:e["idx1"]], rel)
    if return_clipped:
        return corr[k], unc[k], avail, log, len(E[k]), clipped
    return corr[k], unc[k], avail, log, len(E[k])
