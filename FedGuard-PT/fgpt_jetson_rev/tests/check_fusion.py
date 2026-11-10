"""The bounded fusion fit equals sklearn's LogisticRegression when no bound is active, and respects the bounds."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fgpt import nb  # noqa: E402

rng = np.random.default_rng(0)
n, d = 16000, len(nb.TERMS)
Z = rng.normal(size=(n, d))
w_true = 0.5 + np.abs(rng.normal(size=d))            # all clearly positive: no bound should be active
y = (rng.random(n) < 1 / (1 + np.exp(-(Z @ w_true - 1.0)))).astype(int)
from sklearn.linear_model import LogisticRegression  # noqa: E402
a = LogisticRegression(max_iter=100000, C=1.0, tol=1e-12).fit(Z, y)    # sklearn run to full convergence
b = nb.fit_fusion(Z, y, nonneg=True)
dw = float(np.abs(a.coef_ - b.coef_).max())
db = abs(float(a.intercept_[0] - b.intercept_[0]))
print(f"no active bound: max |w_sklearn(tol 1e-12) - w_bounded| = {dw:.2e}, |b diff| = {db:.2e}")
ok1 = dw < 1e-4 and db < 1e-4
w_true[nb.TERMS.index("|r_ups|")] = -1.5             # a check with a truly negative effect
y = (rng.random(n) < 1 / (1 + np.exp(-(Z @ w_true - 1.0)))).astype(int)
b = nb.fit_fusion(Z, y, nonneg=True)
w = b.coef_.ravel()
chk = [w[nb.TERMS.index(t)] for t in nb.CHECK_TERMS]
print("bound active: check weights", np.round(chk, 4))
ok2 = min(chk) >= 0 and abs(w[nb.TERMS.index("|r_ups|")]) < 1e-8
# physics rule: fires only above RULE_Z on the UPS term, adds RULE_BOOST, off for older bundles
Zr = np.zeros((3, d))
Zr[1, nb.TERMS.index("|r_ups|")] = nb.RULE_Z + 1e-6
Zr[2, nb.TERMS.index("|r_ups|")] = nb.RULE_Z
rule = nb.default_rule()
s0 = nb.fused_score(np.ones(d), 0.0, Zr)
s1 = nb.fused_score(np.ones(d), 0.0, Zr, rule=rule)
ok3 = (list(nb.physics_rule(Zr, rule)) == [False, True, False] and np.allclose(s1 - s0, [0, nb.RULE_BOOST, 0])
       and not nb.physics_rule(Zr, None).any() and not nb.physics_rule(Zr, nb.default_rule(False)).any())
print("physics rule:", "ok" if ok3 else "WRONG")
print("PASS" if ok1 and ok2 and ok3 else "FAIL")
sys.exit(0 if ok1 and ok2 and ok3 else 1)
