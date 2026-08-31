# futures_trading_strategies (private research repo)

Personal research notebooks and experiments for systematic futures trading strategies.

The reusable library lives in the separate public repo **sysstrat**. Two ways to install it:

**Active development** (this machine — editable local checkout):

```bash
pip install -e ../sysstrat
```

**From GitHub** (any machine — pinned to a release tag):

```bash
pip install "sysstrat @ git+https://github.com/SoulDancer27/sysstrat.git@v0.1.0"
```

If the package repo is private, use the SSH form instead:

```bash
pip install "sysstrat @ git+ssh://git@github.com/SoulDancer27/sysstrat.git@v0.1.0"
```

## Layout

- `data/` — private market data CSVs (not for redistribution)
- `my_tests/` — experiment notebooks
- `scripts/` — CSV conversion helpers
- `architecture_status.md` — architecture snapshot for the sysstrat package
- root `*.ipynb` — standalone experiment notebooks

## Notes

- `my_tests/legacy/` notebooks use the old pre-architecture API and are kept for reference only.
