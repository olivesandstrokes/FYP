"""framework.py - a first, simple prototype (first half of the FYP).

Steps: 1) make a test problem   2) check the correlations
       3) allocate lots with a greedy method   4) check the answer

Only plain functions, loops and if-statements are used. A priority queue
(heap) is planned for the faster greedy in the second half (Chapter 3).
"""

import numpy as np

MIN_INVESTED = 0.9    # every account must invest at least 90% of its budget


# ---------- 1. Make a test problem ----------
def make_instance(N, K, rho, seed):
    """N assets, K accounts, the same correlation rho for every pair."""
    rng = np.random.default_rng(seed)              # same seed = same problem
    inst = {}
    inst["sigma"] = rng.uniform(0.10, 0.30, N)     # volatility of each asset
    inst["mu"] = 0.02 + 0.3 * inst["sigma"] + rng.normal(0, 0.01, N)  # return
    inst["price"] = rng.integers(1, 6, N).astype(float)   # cost of one lot ($k)
    inst["budget"] = rng.uniform(50, 200, K)       # budget of each account ($k)
    inst["risk_limit"] = rng.uniform(0.10, 0.22, K)  # max volatility per account
    inst["pool"] = np.ceil(1.5 * inst["budget"].sum() / (N * inst["price"]))
    corr = fix_correlation(correlation_matrix(N, rho))
    inst["cov"] = np.outer(inst["sigma"], inst["sigma"]) * corr   # Sigma = DCD
    return inst


# ---------- 2. Check the correlations ----------
def correlation_matrix(N, rho):
    """1 on the diagonal and rho everywhere else."""
    C = np.full((N, N), rho)
    np.fill_diagonal(C, 1.0)
    return C


def is_psd(C):
    """True if C has no negative eigenvalues (allowing for rounding)."""
    return np.linalg.eigvalsh(C).min() >= -1e-10


def fix_correlation(C):
    """Set negative eigenvalues to 0 and make the diagonal 1 again."""
    if is_psd(C):
        return C
    values, vectors = np.linalg.eigh(C)
    C = vectors @ np.diag(np.clip(values, 0, None)) @ vectors.T
    d = np.sqrt(np.diag(C))
    return C / np.outer(d, d)


# ---------- 3. The greedy ----------
def variance_after_add(var_k, Sx_k, cov, i, delta):
    """Equation (8): account variance after adding delta to asset i."""
    return var_k + 2 * delta * Sx_k[i] + delta**2 * cov[i, i]


def score(inst, phase, aware, i, risk_now, risk_new):
    """How good one more lot of asset i is. Higher is better."""
    if not aware:                                  # correlation-blind
        if phase == "feasibility":
            return -inst["sigma"][i]               # lowest volatility first
        return inst["mu"][i] / inst["sigma"][i]    # best return per volatility
    if phase == "feasibility":                     # least extra risk per dollar
        return -(risk_new**2 - risk_now**2) / inst["price"][i]
    return inst["mu"][i] * inst["price"][i] / max(risk_new - risk_now, 1e-6)


def greedy(inst, aware=True):
    """Buy one lot at a time.
    Phase 1: get every account to 90% invested.
    Phase 2: add more return.
    Each round we look at every allowed (account, asset) pair, score it,
    and buy the pair with the highest score."""
    N, K = len(inst["price"]), len(inst["budget"])
    lots = np.zeros((N, K), dtype=int)             # lots[i, k] = lots of asset i in account k

    for phase in ["feasibility", "return"]:
        while True:
            best_score = None                      # best pair found so far
            best_k, best_i = None, None

            for k in range(K):
                spent = inst["price"] @ lots[:, k]
                if phase == "feasibility" and spent >= MIN_INVESTED * inst["budget"][k]:
                    continue                       # already invested enough
                x = inst["price"] * lots[:, k] / inst["budget"][k]
                Sx = inst["cov"] @ x
                var = x @ Sx

                for i in range(N):
                    if lots[i].sum() >= inst["pool"][i]:
                        continue                   # none left in the pool
                    if spent + inst["price"][i] > inst["budget"][k]:
                        continue                   # cannot afford it
                    delta = inst["price"][i] / inst["budget"][k]
                    v_new = variance_after_add(var, Sx, inst["cov"], i, delta)
                    risk_new = np.sqrt(v_new)
                    if risk_new > inst["risk_limit"][k]:
                        continue                   # breaks the risk limit

                    s = score(inst, phase, aware, i, np.sqrt(var), risk_new)
                    # ">" (not ">=") keeps the first pair on a tie
                    if best_score is None or s > best_score:
                        best_score, best_k, best_i = s, k, i

            if best_score is None:
                break                              # nothing more can be bought
            lots[best_i, best_k] += 1              # buy the best lot
    return lots


# ---------- 4. Check the answer ----------
def account_risk(inst, lots, k):
    """Volatility of account k, computed from scratch."""
    x = inst["price"] * lots[:, k] / inst["budget"][k]
    return np.sqrt(x @ inst["cov"] @ x)


def check(inst, lots):
    """Re-check every rule instead of trusting the greedy."""
    feasible, div_ratio = [], []
    for k in range(len(inst["budget"])):
        x = inst["price"] * lots[:, k] / inst["budget"][k]
        risk = account_risk(inst, lots, k)
        enough = MIN_INVESTED - 1e-9 <= x.sum() <= 1 + 1e-9
        within_limit = risk <= inst["risk_limit"][k] + 1e-9
        feasible.append(enough and within_limit)
        div_ratio.append(x @ inst["sigma"] / risk if risk > 0 else 1.0)
    return {"feasible": feasible,
            "pool_ok": bool(np.all(lots.sum(axis=1) <= inst["pool"])),
            "total_return": np.sum(inst["mu"] * inst["price"] * lots.sum(axis=1)),
            "div_ratio": div_ratio}


# ---------- Run: 30 problems for each correlation level ----------
if __name__ == "__main__":
    print("rho  method  feasible  return    DR")
    for rho in [0.0, 0.3, 0.6, 0.9]:
        for aware in [False, True]:
            rows = []
            for seed in range(30):
                inst = make_instance(30, 5, rho, seed)
                r = check(inst, greedy(inst, aware))
                rows.append([np.mean(r["feasible"]), r["total_return"],
                             np.mean(r["div_ratio"])])
            f, ret, dr = np.mean(rows, axis=0)
            name = "aware" if aware else "blind"
            print(f"{rho:<4} {name:<7} {f:>7.0%} {ret:>7.1f} {dr:>5.2f}")

    inst = make_instance(30, 5, 0.9, seed=3)
    lots = greedy(inst, aware=True)
    ok = check(inst, lots)["feasible"]
    print("\nExample (rho = 0.9, problem 3), aware greedy")
    print("account  invested   risk   limit  feasible")
    for k in range(5):
        invested = inst["price"] @ lots[:, k] / inst["budget"][k]
        print(f"{k + 1:>7} {invested:>9.0%} {account_risk(inst, lots, k):>6.1%}"
              f" {inst['risk_limit'][k]:>7.1%}  {'yes' if ok[k] else 'no'}")
