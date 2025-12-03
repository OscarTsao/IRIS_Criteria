# Dataset Description: DSM-5 Criteria Matching

## Overview

The dataset contains 14,840 samples from 1,484 social media posts, each evaluated against 10 criteria (A.1-A.10).

**Key Statistics:**
- **Posts**: 1,484 unique social media posts
- **Total Samples**: 14,840 (1,484 posts × 10 criteria)
- **Class Distribution**: 90.7% negative (13,459), 9.3% positive (1,381)
- **Source**: Social media mental health subreddits

---

## DSM-5 Criteria (A.1 - A.9)

These are the official DSM-5 Major Depressive Disorder diagnostic criteria:

| ID | Criterion | Description |
|----|-----------|-------------|
| **A.1** | Depressed mood | Sad, empty, hopeless feelings most of the day |
| **A.2** | Anhedonia | Loss of interest/pleasure in activities |
| **A.3** | Weight change | Significant weight/appetite changes |
| **A.4** | Sleep disturbance | Insomnia or hypersomnia |
| **A.5** | Psychomotor changes | Agitation or retardation (observable) |
| **A.6** | Fatigue | Loss of energy nearly every day |
| **A.7** | Guilt/worthlessness | Excessive guilt or feelings of worthlessness |
| **A.8** | Concentration problems | Difficulty thinking or concentrating |
| **A.9** | Suicidal ideation | Thoughts of death or suicide |

**Positive Rate by Criterion:**
- A.1: ~12-15% (depressed mood most common)
- A.2: ~8-10% (anhedonia)
- A.4: ~10-12% (sleep disturbance)
- A.9: ~5-8% (suicidal ideation)
- Others: ~3-7%

---

## A.10: Non-DSM-5 Clinical Criterion (Special Case)

**Purpose**: A.10 is a **synthetic criterion** for negative discrimination.

### Definition
**"Non-DSM-5 clinical or positive discriminations"**

A.10 captures posts that:
1. ✅ **Have clinical/mental health content**
2. ❌ **Do NOT match any DSM-5 MDD criteria (A.1-A.9)**

### Use Case

This criterion helps the model learn to distinguish:
- **MDD symptoms** (A.1-A.9) from
- **Other mental health conditions** (anxiety, ADHD, ASD, PTSD, etc.)
- **General clinical discussions** that aren't MDD

### Examples of A.10 Content

Posts labeled as A.10 (typically with groundtruth=0) include:
- **Other disorders**: "I was diagnosed with ADHD, C-PTSD, and possibly ASD..."
- **Treatment discussions**: "I switched to Trazodone and gave up dairy..."
- **General mental health**: "How did you choose your new name?" (trans identity)
- **Non-clinical social**: "How DARE they feel bad when someone tells a racist joke?"

### Statistics

- **Total A.10 samples**: 1,484 (same as other criteria)
- **Positive labels**: 86 (5.8%)
- **Negative labels**: 1,398 (94.2%)

**Note**: A.10 has a slightly lower positive rate than true DSM-5 criteria because it serves as a "catch-all" for clinical content that doesn't meet MDD criteria.

---

## Data Structure

### Input Files

**1. criteria_matching_groundtruth.csv**
```csv
post_id,post,DSM5_symptom,groundtruth
s_1270_9,"Im not sleeping...",A.1,0
s_1270_9,"Im not sleeping...",A.2,0
s_1270_9,"Im not sleeping...",A.4,1
...
```

**Columns:**
- `post_id`: Unique identifier for each post
- `post`: Raw text of the social media post
- `DSM5_symptom`: Criterion ID (A.1-A.10)
- `groundtruth`: Binary label (0=no match, 1=match)

**2. MDD_Criteria.json**
```json
{
  "diagnosis": "Major Depressive Disorder",
  "criteria": [
    {"id": "A.1", "text": "Depressed mood..."},
    ...
    {"id": "A.10", "text": "Non-DSM-5 clinical..."}
  ]
}
```

---

## Class Imbalance

The dataset is **highly imbalanced**:
- **Negative**: 90.7% (13,459 samples)
- **Positive**: 9.3% (1,381 samples)

**Ratio**: ~9.7:1 (negative:positive)

### Implications for Training

1. **Loss Function**: Use Focal Loss or Weighted BCE
   - Focal Loss (γ=2.0, α=0.093) **recommended**
   - Weighted BCE (pos_weight=9.82) alternative

2. **Evaluation Metrics**:
   - **Don't use accuracy** (misleading on imbalanced data)
   - **Use F1, Precision, Recall, Sensitivity**
   - **AUC-PR** more informative than AUC-ROC

3. **Class Weights**:
   - Negative class: weight = 0.54
   - Positive class: weight = 5.37

---

## Data Split Strategy

### K-Fold Cross-Validation

**Critical**: Use **grouped K-fold** by `post_id` to prevent data leakage.

```python
# CORRECT: Group by post_id
splits = create_kfold_splits(df, n_folds=5, group_column="post_id")
```

**Why?** Each post appears 10 times (once per criterion). If you don't group by post, the same post could appear in both train and validation sets, leading to inflated performance.

### Split Statistics (5-fold)

- **Fold size**: ~2,968 samples (297 posts × 10 criteria)
- **Train set**: ~11,872 samples per fold
- **Val set**: ~2,968 samples per fold

---

## Label Distribution Analysis

### Per-Criterion Positive Rates

Based on the dataset:
- **A.1 (Depressed mood)**: Highest positive rate (~12-15%)
- **A.4 (Sleep)**: High positive rate (~10-12%)
- **A.2 (Anhedonia)**: Moderate (~8-10%)
- **A.9 (Suicidal)**: Lower (~5-8%)
- **A.10 (Non-DSM-5)**: ~5.8% (86/1484)

### Clinical Interpretation

The positive rate reflects:
1. **Symptom prevalence** in the source population (mental health subreddit users)
2. **Annotation guidelines** (strict adherence to DSM-5 criteria)
3. **Post selection bias** (posts from depression-related subreddits)

---

## Data Quality Notes

1. **Self-reported**: Posts are self-reported, not clinically diagnosed
2. **Annotation**: Labels based on DSM-5 criteria by clinical experts
3. **Noise**: Natural language ambiguity and subjective interpretation
4. **A.10 special handling**: Serves dual purpose (criterion + negative discriminator)

---

## Usage in IRIS Model

The IRIS model processes data as follows:

1. **Input**: Concatenated `post + criterion_text`
2. **Retrieval**: Top-k chunks retrieved per query vector
3. **Classification**: Binary prediction (match/no match)

**Example**:
```
Post: "I can't sleep and feel hopeless every day"
Criterion A.4: "Insomnia or hypersomnia nearly every day"
Label: 1 (match)

Criterion A.10: "Non-DSM-5 clinical..."
Label: 0 (this IS an MDD symptom, not non-DSM-5)
```

---

## References

- **DSM-5**: American Psychiatric Association (2013)
- **IRIS Paper**: ACL 2025 (Interpretable Retrieval-Augmented Classification)
- **Source**: Reddit mental health communities
