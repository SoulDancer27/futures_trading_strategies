# futures_trading_strategies (private research repo)

Personal research notebooks and experiments for systematic futures trading strategies.

The reusable library lives in the separate public repo **sysstrat**. Install it editable
before running the notebooks:

```bash
pip install -e ../sysstrat
```

## Layout

- `data/` — private market data CSVs (not for redistribution)
- `my_tests/` — experiment notebooks
- `scripts/` — CSV conversion helpers
- `architecture_status.md` — architecture snapshot for the sysstrat package
- root `*.ipynb` — standalone experiment notebooks

## Notes

- `my_tests/legacy/` notebooks use the old pre-architecture API and are kept for reference only.
