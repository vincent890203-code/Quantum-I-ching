# Feature Selection Funnel — Deep Code Audit

## Task 1: Feature Count Audit

### 1.1 Raw Market Data

**Source:** `data_loader.py` (yfinance) → OHLCV.

| Feature | Used in Models? |
|--------|------------------|
| Open   | No              |
| High   | No              |
| Low    | No              |
| Close  | Yes (Model A, B, C) |
| Volume | Yes (Model A, B, C) |

**Count:** 5 raw; 2 (Close, Volume) used as model inputs.

---

### 1.2 Technical Indicators

**Source:** `market_encoder._calculate_technical_indicators`.

| Feature           | Definition                    | Used in Models?        |
|-------------------|-------------------------------|------------------------|
| Daily_Return      | Close.pct_change()            | Yes (all)              |
| Volume_MA20       | Volume.rolling(20).mean()     | No (encoding only)     |
| RVOL              | Volume / Volume_MA20          | Yes (Model A, B, C)    |
| RVOL_Percentile   | Rolling RVOL rank             | No (encoding only)     |

**Count:** 4 technical; 2 (Daily_Return, RVOL) used as model inputs.

---

### 1.3 I-Ching Features

**Source:** `data_processor.extract_iching_features` (from `Ritual_Sequence`).

**Encoder also produces:** `Ritual_Sequence`, `Hexagram_Binary`, `Future_Hex_ID`, `Num_Moving_Lines`. These are used for oracle/UI, not as numeric XGBoost features.

**Extracted numeric features (5):**

| Feature            | Description                          |
|--------------------|--------------------------------------|
| Yang_Count_Main    | 主卦陽爻數 (7, 9)                    |
| Yang_Count_Future  | 變卦陽爻數                           |
| Moving_Lines_Count | 動爻數 (6, 9)                        |
| Energy_Delta       | Yang_Count_Future − Yang_Count_Main  |
| Conflict_Score     | \|上卦和 − 下卦和\|                  |

**One-hot hexagrams?** No.  
**Line positions 1–6?** No (we use aggregates only).  
**Element types?** No.

**Count:** 5 I-Ching numeric features.

---

### 1.4 Potential vs Model A vs Model D

| Stage                       | Count | Features |
|----------------------------|-------|----------|
| **Potential (all generated)** | 14 | Open, High, Low, Close, Volume, Daily_Return, Volume_MA20, RVOL, RVOL_Percentile, Yang_Count_Main, Yang_Count_Future, Moving_Lines_Count, Energy_Delta, Conflict_Score |
| **Model A (initial screen)**  | 9  | Close, Volume, RVOL, Daily_Return, Yang_Count_Main, Yang_Count_Future, Moving_Lines_Count, Energy_Delta, Conflict_Score |
| **Model C (importance filter)** | 6 | Close, Volume, RVOL, Daily_Return, Moving_Lines_Count, Energy_Delta |
| **Model D (Pure Quantum)**     | 3  | Daily_Return, Moving_Lines_Count, Energy_Delta |

**Note:** RSI, MACD not implemented. Model B uses only 4 numerical features (baseline).

---

## Funnel Stages (for Visualization)

1. **Stage 1 — Raw Data Universe:** 14 potential features (OHLCV + tech + I-Ching).
2. **Stage 2 — Model A:** 9 features (first trained screen).
3. **Stage 3 — Feature Importance Filter:** 6 features (Model C; drop 3 I-Ching).
4. **Stage 4 — Model D (Pure Quantum):** 3 core alpha signals.
