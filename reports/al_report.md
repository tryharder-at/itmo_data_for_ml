# Active Learning Report

Generated: 2026-03-28T01:20:56

## Protocol
- Initial: 50 | Pool: 190 | Test: 60
- Iterations: 10 × 20 examples

## Entropy Strategy Results
|   iteration |   n_labeled |   accuracy |   f1_macro |
|------------:|------------:|-----------:|-----------:|
|           0 |          50 |     0.6167 |     0.3814 |
|           1 |          70 |     0.65   |     0.4982 |
|           2 |          90 |     0.6667 |     0.5342 |
|           3 |         110 |     0.7    |     0.6429 |
|           4 |         130 |     0.7167 |     0.6851 |
|           5 |         150 |     0.7333 |     0.6591 |
|           6 |         170 |     0.7333 |     0.6717 |
|           7 |         190 |     0.75   |     0.6738 |
|           8 |         210 |     0.7167 |     0.6135 |
|           9 |         230 |     0.6667 |     0.509  |
|          10 |         240 |     0.65   |     0.4695 |

## Final Metrics
| Metric | Value |
|--------|-------|
| Accuracy | 0.65 |
| F1-macro | 0.4695 |
| F1-positive | 0.16 |

## LLM Analysis (Claude)
❌ ANTHROPIC_API_KEY not set in .env
