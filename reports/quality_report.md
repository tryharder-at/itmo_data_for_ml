# Data Quality Report

Generated: 2026-03-28T01:16:15

## Raw Dataset
- Rows: 2000
- Missing: 0 rows (0.0%)
- Duplicates: 0 (0.0%)
- Outliers: 152 (7.6%)
- Imbalance ratio: 1.11x

## Strategy Applied
| Issue | Strategy |
|-------|---------|
| Missing | drop |
| Duplicates | drop |
| Outliers | clip_iqr |
| Imbalance | undersample |

## After Cleaning
- Rows: 1896
- Removed: 104
- Distribution: {'negative': np.int64(948), 'positive': np.int64(948)}

## LLM Analysis (Claude)
❌ ANTHROPIC_API_KEY not found. Create a .env file with ANTHROPIC_API_KEY=sk-ant-...
