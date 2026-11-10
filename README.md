# Correlation-Aware Heuristic Optimisation for Multi-Portfolio Asset Allocation

Code for my NTU EEE Final Year Project (interim stage), supervised by Dr Lim Wei Quan.

The project asks: when a firm cannot serve every client from one shared pool of assets,
how much of the shortfall comes from the assets (their correlations) and how much from
the allocation method?

## Files

| File | What it does | Report section |
| --- | --- | --- |
| `framework.py` | Prototype: test-problem generator, PSD check and repair, two-phase greedy (correlation-aware and correlation-blind), and an evaluator | 2.4.1, 2.5, Appendix A |
| `test_framework.py` | Seven unit tests for the prototype | 2.6 |
| `real_data_check.py` | Risk floor on real monthly data for stocks, bonds and gold | 2.4.3 |

## How to run

Python 3 with the packages in `requirements.txt`:

```
pip install -r requirements.txt
python framework.py          # prints the results in Table A1 and Listing A2
python -m pytest -q          # runs the seven unit tests
```

For the real-data check, first download these three files into this folder
(from https://github.com/datasets), saving them under the names shown:

| Save as | Download from |
| --- | --- |
| `s-and-p-500.csv` | https://raw.githubusercontent.com/datasets/s-and-p-500/main/data/data.csv |
| `gold-prices.csv` | https://raw.githubusercontent.com/datasets/gold-prices/main/data/monthly.csv |
| `bond-yields-us-10y.csv` | https://raw.githubusercontent.com/datasets/bond-yields-us-10y/main/data/monthly.csv |

Then run `python real_data_check.py`. It prints every number quoted in Section 2.4.3.
These datasets are updated over time, so newer downloads can give slightly different numbers.
The report used data up to September 2026, downloaded on 10 November 2026.
