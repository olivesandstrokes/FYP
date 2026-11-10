"""real_data_check.py - Section 2.4.3: the risk floor on real market data.

Data (monthly, from github.com/datasets):
  s-and-p-500/data/data.csv          S&P 500 level (Shiller, monthly average)
  gold-prices/data/monthly.csv       gold price, USD per ounce (monthly)
  bond-yields-us-10y/data/monthly.csv  US 10-year Treasury yield (monthly)
"""
import numpy as np, pandas as pd
from scipy.optimize import minimize

ALPHA, WINDOW = 0.9, 36            # 90% invested; 3-year rolling window (months)

def bond_return(y_prev, y_now):
    """Monthly return of a 10-year par bond with semiannual coupons:
    bought at par with coupon y_prev, revalued one month later at yield y_now
    with 9 years 11 months left, plus one month of coupon income (y_prev/12)."""
    n = 2 * (10 - 1 / 12)                      # semiannual periods left
    k = np.arange(1, int(np.ceil(n)) + 1) - (np.ceil(n) - n)
    price = (y_prev / 2 / (1 + y_now / 2) ** k).sum() + 1 / (1 + y_now / 2) ** k[-1]
    return price - 1 + y_prev / 12

def min_vol(cov):
    """Exact lowest volatility of a long-only, fully invested portfolio (convex QP)."""
    n = len(cov)
    res = minimize(lambda w: w @ cov @ w, np.ones(n) / n, jac=lambda w: 2 * cov @ w,
                   bounds=[(0, 1)] * n, constraints=[{'type': 'eq', 'fun': lambda w: w.sum() - 1}],
                   method='SLSQP', options={'ftol': 1e-15, 'maxiter': 500})
    return np.sqrt(res.fun), res.x

sp = pd.read_csv('s-and-p-500.csv', parse_dates=['Date']).set_index('Date')['SP500']
gold = pd.read_csv('gold-prices.csv')
gold = gold.set_index(pd.to_datetime(gold['Date'] + '-01'))['Price']
yld = pd.read_csv('bond-yields-us-10y.csv', parse_dates=['Date']).set_index('Date')['Rate'] / 100
bonds = pd.Series([bond_return(yld.iloc[i - 1], yld.iloc[i]) for i in range(1, len(yld))], index=yld.index[1:])

r = pd.concat({'stocks': sp.pct_change(), 'bonds': bonds, 'gold': gold.pct_change()}, axis=1, sort=True).dropna()
r = r[r.index >= '2000-01-01']

vol = r.std() * np.sqrt(12)                    # volatility: scale monthly by sqrt(12)
corr = r.corr()                                # correlations need no scaling
cov = r.cov().values * 12                      # covariance: scale monthly by 12
print('Months:', len(r), r.index[0].date(), 'to', r.index[-1].date())
print('Annual volatility (%):', (100 * vol).round(1).to_dict())
print('Correlations:\n', corr.round(2))
w = np.array([0.6, 0.4, 0.0])
print('60/40: %.1f%% vs weighted average %.1f%%' % (100 * np.sqrt(w @ cov @ w), 100 * w @ vol.values))
f, wf = min_vol(cov)
print('Floor, whole period: %.2f%% (weights %s)' % (100 * ALPHA * f, wf.round(2)))

rows = []
for end in range(WINDOW, len(r) + 1):
    s = r.iloc[end - WINDOW:end]
    f, _ = min_vol(s.cov().values * 12)
    rows.append((s.index[-1], s.corr().iloc[0, 1], ALPHA * f))
roll = pd.DataFrame(rows, columns=['end', 'corr_stock_bond', 'floor']).set_index('end')
roll.to_csv('rolling_floor.csv')
for d in ['2021-12', '2023-12']:
    print(d, roll.loc[d].round(4).to_dict('records'))
for a, b in [('2009-01', '2021-12'), ('2022-01', str(roll.index[-1].date()))]:
    print(a, 'to', b, ': floor above 5%% in %.0f%% of windows' % (100 * (roll.loc[a:b, 'floor'] > 0.05).mean()))
print('Floor range: %.2f%% to %.2f%%' % (100 * roll.floor.min(), 100 * roll.floor.max()))
