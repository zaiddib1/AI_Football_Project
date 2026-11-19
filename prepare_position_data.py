"""Audit the full player table, export position-classification CSVs, and compare feature views.

Run from any directory using the project Python environment. No imputation, one-hot
encoding, or scaling is learned before the cross-validation split.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, classification_report,
    confusion_matrix, f1_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
SEED = 42
LABELS = ['DF', 'MF', 'FW']
COUNTS = ['Sh', 'SoT', 'Crs', 'Int', 'TklW', 'Fls', 'Fld', 'Off',
          'Gls', 'Ast', 'PK', 'PKatt', 'CrdY', 'CrdR', '2CrdY', 'OG']
COUNT_DESCRIPTIONS = {
    'Sh': 'Shots', 'SoT': 'Shots on target', 'Crs': 'Crosses',
    'Int': 'Interceptions', 'TklW': 'Tackles won (not all attempted tackles)',
    'Fls': 'Fouls committed', 'Fld': 'Fouls drawn', 'Off': 'Offsides',
    'Gls': 'Goals', 'Ast': 'Assists', 'PK': 'Penalty goals',
    'PKatt': 'Penalty attempts', 'CrdY': 'Yellow cards', 'CrdR': 'Red cards',
    '2CrdY': 'Second yellow cards', 'OG': 'Own goals',
}
COMMON_NUM = ['Age', 'Min']
CATEGORICAL = ['Comp']
IDENTIFIERS = ['Player', 'Rk', 'Squad']
KEEPER_FIELDS = {'GA', 'GA90', 'SoTA', 'Saves', 'Save%', 'W', 'D', 'L',
                 'CS', 'CS%', 'PKA', 'PKsv', 'PKm'}
ALGEBRAIC = {'G+A', 'G-PK', 'G+A-PK', 'G/Sh', 'G/SoT', 'SoT%', 'Sh/90', 'SoT/90', '90s'}


def numeric(series: pd.Series, *, age: bool = False) -> pd.Series:
    text = series.astype('string').str.strip().str.replace(',', '', regex=False)
    if age:
        text = text.str.extract(r'^([0-9]+(?:\.[0-9]+)?)', expand=False)
    return pd.to_numeric(text, errors='coerce').astype(float).replace([np.inf, -np.inf], np.nan)


def audit_source(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    schema = pd.DataFrame({
        'column': raw.columns, 'dtype': raw.dtypes.astype(str).to_numpy(),
        'missing_n': raw.isna().sum().to_numpy(),
        'missing_percent': raw.isna().mean().mul(100).to_numpy(),
        'distinct_nonmissing': raw.nunique().to_numpy(),
    })
    merged = []
    for col in raw.columns:
        if '_stats_' not in col:
            continue
        base = col.split('_stats_', 1)[0]
        if base not in raw:
            continue
        overlap = raw[base].notna() & raw[col].notna()
        left, right = raw.loc[overlap, base], raw.loc[overlap, col]
        if base in ['Age', 'Born', 'MP', 'Starts', 'Min', '90s', 'Gls', 'PK', 'PKatt', 'CrdY', 'CrdR', 'Rk']:
            equal = np.isclose(numeric(left, age=base == 'Age'), numeric(right, age=base == 'Age'))
        else:
            equal = left.astype('string').str.strip().eq(right.astype('string').str.strip()).to_numpy()
        merged.append({'column': col, 'base': base, 'overlap_n': int(overlap.sum()),
                       'equal_on_overlap_percent': float(np.mean(equal)*100) if overlap.any() else np.nan})
    identical = []
    for i, left in enumerate(raw.columns):
        for right in raw.columns[i+1:]:
            if raw[left].equals(raw[right]):
                identical.append({'column_a': left, 'column_b': right})
    return schema, pd.DataFrame(merged), pd.DataFrame(identical, columns=['column_a', 'column_b'])


def prepare(raw: pd.DataFrame, min_minutes: int = 180):
    required = set(COMMON_NUM + CATEGORICAL + COUNTS + ['Player', 'Pos', 'Born'])
    missing = required-set(raw.columns)
    if missing:
        raise ValueError(f'Full statistics CSV is missing required columns: {sorted(missing)}')
    frame = raw.copy(deep=True)
    frame['_source_row'] = np.arange(len(frame))
    steps = []

    def retain(mask, decision):
        nonlocal frame
        previous_n = len(frame)
        frame = frame.loc[mask].copy()
        steps.append({'decision': decision, 'removed_n': previous_n-len(frame), 'remaining_n': len(frame)})

    steps.append({'decision': 'Original full CSV', 'removed_n': 0, 'remaining_n': len(frame)})
    retain(~raw.duplicated(), 'Remove exact duplicate source records')
    # Only the first listed position defines the target; secondary positions are not inputs.
    frame['Target'] = frame['Pos'].astype('string').str.split(',', n=1).str[0].str.strip().str.upper()
    retain(frame.Target.ne('GK').fillna(True), 'Exclude primary-position goalkeepers')
    retain(frame.Target.isin(LABELS), 'Exclude missing or unsupported primary positions')
    frame['Min'] = numeric(frame['Min'])
    retain(frame.Min.notna() & frame.Min.gt(0), 'Exclude missing, nonfinite, or nonpositive minutes')
    retain(frame.Min.ge(min_minutes), f'Retain at least {min_minutes} minutes played')
    frame['_player_group'] = frame.Player.astype('string').str.strip().replace('', pd.NA)
    retain(frame._player_group.notna(), 'Exclude missing player grouping identity')

    conversion_rows = []
    for column in ['Age'] + COUNTS:
        original = frame[column].copy()
        parsed = numeric(original, age=column == 'Age')
        invalid = parsed.le(0) if column == 'Age' else parsed.lt(0)
        parsed = parsed.mask(invalid)
        frame[column] = parsed
        conversion_rows.append({'column': column,
                                'new_missing_after_invalid_value_handling': int((original.notna() & parsed.isna()).sum()),
                                'eligible_missing_n': int(parsed.isna().sum())})
    for column in CATEGORICAL:
        text = frame[column].astype('string').str.strip().replace('', pd.NA)
        frame[column] = text.astype(object).where(text.notna(), np.nan)
    frame = frame.reset_index(drop=True)
    for column in COUNTS:
        # Use exact minutes instead of the already-rounded supplied 90s field.
        frame[f'{column}_p90'] = frame[column] * 90.0 / frame['Min']

    raw_columns = COMMON_NUM + CATEGORICAL + COUNTS
    rate_columns = COMMON_NUM + CATEGORICAL + [f'{column}_p90' for column in COUNTS]
    raw_view = frame[raw_columns + ['Target']].copy()
    rate_view = frame[rate_columns + ['Target']].copy()
    assert frame.Target.isin(LABELS).all() and frame.Min.ge(min_minutes).all()
    assert not any(c.startswith('Pos') or c in IDENTIFIERS + ['Born'] for c in raw_columns+rate_columns)
    assert raw_view.Target.equals(rate_view.Target)
    assert not np.isinf(rate_view.select_dtypes('number').to_numpy()).any()
    return frame, raw_view, rate_view, pd.DataFrame(steps), pd.DataFrame(conversion_rows)


def make_pipeline(model_name: str, representation: str) -> Pipeline:
    count_cols = COUNTS if representation == 'raw_counts' else [f'{c}_p90' for c in COUNTS]
    numerical_steps = [('impute', SimpleImputer(strategy='median', keep_empty_features=True))]
    if model_name == 'Logistic regression':
        numerical_steps.append(('scale', StandardScaler()))
    preprocess = ColumnTransformer([
        ('numeric', Pipeline(numerical_steps), COMMON_NUM + count_cols),
        ('categorical', Pipeline([
            ('impute', SimpleImputer(strategy='most_frequent', keep_empty_features=True)),
            ('encode', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
        ]), CATEGORICAL),
    ], remainder='drop')
    if model_name == 'Logistic regression':
        estimator = LogisticRegression(C=1., max_iter=2000, class_weight='balanced')
    elif model_name == 'Random forest':
        estimator = RandomForestClassifier(n_estimators=200, min_samples_leaf=3,
            class_weight='balanced', random_state=SEED, n_jobs=1)
    elif model_name == 'Majority baseline':
        estimator = DummyClassifier(strategy='most_frequent')
    else:
        raise ValueError(model_name)
    return Pipeline([('preprocess', preprocess), ('model', estimator)])


def compare(frame: pd.DataFrame, raw_view: pd.DataFrame, rate_view: pd.DataFrame):
    y = frame.Target.to_numpy()
    groups = frame._player_group.to_numpy()
    splits = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED).split(raw_view, y, groups))
    folds = np.zeros(len(frame), dtype=int)
    audit = []
    for fold, (train, valid) in enumerate(splits, 1):
        overlap = set(groups[train]) & set(groups[valid])
        assert not overlap
        assert set(y[train]) == set(LABELS) and set(y[valid]) == set(LABELS)
        folds[valid] += 1
        row = {'fold': fold, 'n_train': len(train), 'n_valid': len(valid), 'overlapping_player_names': len(overlap)}
        row.update({f'valid_{label}_n': int((y[valid] == label).sum()) for label in LABELS})
        audit.append(row)
    assert (folds == 1).all()
    rows, predictions = [], pd.DataFrame({'Target': y})
    for fold, (_, valid) in enumerate(splits, 1):
        predictions.loc[valid, 'fold'] = fold
    predictions['fold'] = predictions['fold'].astype(int)
    for representation, data in [('raw_counts', raw_view), ('per90', rate_view)]:
        X = data.drop(columns='Target')
        for model_name in ['Majority baseline', 'Logistic regression', 'Random forest']:
            oof = np.full(len(frame), '', dtype=object)
            template = make_pipeline(model_name, representation)
            for fold, (train, valid) in enumerate(splits, 1):
                estimator = clone(template)
                with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=1):
                    warnings.simplefilter('always', ConvergenceWarning)
                    estimator.fit(X.iloc[train], y[train])
                    pred = estimator.predict(X.iloc[valid])
                convergence = [w for w in caught if issubclass(w.category, ConvergenceWarning)]
                for warning in caught:
                    if not issubclass(warning.category, ConvergenceWarning):
                        warnings.warn(str(warning.message), warning.category)
                oof[valid] = pred
                per_class = classification_report(y[valid], pred, labels=LABELS, output_dict=True, zero_division=0)
                row = {'representation': representation, 'model': model_name, 'fold': fold,
                    'macro_f1': f1_score(y[valid], pred, labels=LABELS, average='macro', zero_division=0),
                    'balanced_accuracy': balanced_accuracy_score(y[valid], pred),
                    'accuracy': accuracy_score(y[valid], pred),
                    'convergence_warnings': len(convergence)}
                for label in LABELS:
                    row[f'{label}_f1'] = per_class[label]['f1-score']
                    row[f'{label}_recall'] = per_class[label]['recall']
                rows.append(row)
            assert np.isin(oof, LABELS).all()
            predictions[f'{representation}__{model_name}'] = oof
    results = pd.DataFrame(rows)
    return results, predictions, pd.DataFrame(audit)


def decisions(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in raw.columns:
        if column == 'Pos':
            action, reason = 'Target only', 'First listed position becomes DF/MF/FW in Target; original label removed.'
        elif column.startswith('Pos_'):
            action, reason = 'Drop', 'Duplicate position label directly reveals the target.'
        elif column in IDENTIFIERS or column.startswith('Rk_'):
            action, reason = 'Drop', 'Player/team identifier or arbitrary row rank; Player used privately only for fold grouping.'
        elif '_stats_keeper' in column or column in KEEPER_FIELDS:
            action, reason = 'Drop', 'Keeper-specific variable, structurally missing/irrelevant for outfield cohort.'
        elif '_stats_' in column:
            action, reason = 'Drop', 'Merged-table copy; retain the canonical source variable when selected.'
        elif column == 'Born':
            action, reason = 'Drop', 'Birth year strongly overlaps Age; no match/reference date supports extra age precision.'
        elif column in COMMON_NUM + CATEGORICAL:
            action, reason = 'Retain in both views', 'Shared player age, exposure, or league covariate.'
        elif column in COUNTS:
            action, reason = 'Raw view; convert in per90 view', 'Event count retained once per representation; per90 = count × 90 / Min.'
        elif column in ALGEBRAIC:
            action, reason = 'Drop', 'Derived/redundant or rounded statistic; canonical counts and exact-Min rates avoid duplicate information.'
        elif column == 'Nation':
            action, reason = 'Drop', 'Nationality is outside the controlled behavioral comparison; not needed as a predictor.'
        else:
            action, reason = 'Drop', 'Additional playing-time or team-context statistic omitted from the explicit feature allowlist; Min supplies common exposure.'
        rows.append({'source_column': column, 'action': action, 'reason': reason})
    return pd.DataFrame(rows)


def md_table(frame: pd.DataFrame, include_index: bool = False) -> str:
    view = frame.reset_index() if include_index else frame.copy()
    def text(value):
        if pd.isna(value):
            return '—'
        return str(value).replace('|', '\\|').replace('\n', ' ')
    return '\n'.join([
        '| ' + ' | '.join(map(str, view.columns)) + ' |',
        '| ' + ' | '.join(['---']*len(view.columns)) + ' |',
        *['| ' + ' | '.join(text(v) for v in row) + ' |' for row in view.itertuples(index=False, name=None)],
    ])


def plots(frame, results, summary, predictions, report_dir):
    sns.set_theme(style='whitegrid', palette='colorblind')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), layout='constrained')
    counts = frame.Target.value_counts().reindex(LABELS)
    axes[0].bar(counts.index, counts, color=['#0072B2', '#009E73', '#D55E00'])
    for i, value in enumerate(counts):
        axes[0].text(i, value+8, f'{value:,}\n{value/len(frame):.1%}', ha='center')
    axes[0].set(title='Primary positions after the 180-minute filter', xlabel='Primary position', ylabel='Player records')
    axes[0].set_ylim(0, counts.max()*1.17)
    x = np.arange(2)
    for j, representation in enumerate(['raw_counts', 'per90']):
        tab = summary.query('representation == @representation').set_index('model').loc[['Logistic regression', 'Random forest']]
        axes[1].bar(x+(j-.5)*.34, tab.macro_f1_mean, yerr=tab.macro_f1_std,
                    width=.32, capsize=3, label='Raw counts' if j == 0 else 'Per 90 minutes')
    axes[1].set_xticks(x, ['Logistic regression', 'Random forest'])
    axes[1].set(title='Same grouped folds, different feature representation', ylabel='Macro F1 (five-fold mean ± SD)', ylim=(0, 1))
    axes[1].legend()
    fig.savefig(report_dir/'class_balance_and_feature_comparison.png', dpi=160, bbox_inches='tight')
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2), layout='constrained')
    for ax, representation in zip(axes, ['raw_counts', 'per90']):
        pred = predictions[f'{representation}__Logistic regression']
        matrix = confusion_matrix(frame.Target, pred, labels=LABELS)
        sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=LABELS, yticklabels=LABELS, ax=ax)
        label = 'raw counts' if representation == 'raw_counts' else 'per-90 rates'
        ax.set(title=f'Logistic regression: {label}', xlabel='Predicted position', ylabel='True position')
    fig.savefig(report_dir/'logistic_oof_confusion_matrices.png', dpi=160, bbox_inches='tight')
    plt.close(fig)


def write_report(raw, frame, rate_view, flow, conversion, schema, merged, identical,
                 results, summary, class_balance, paired, report_dir, source_path):
    missing_raw = schema.query('missing_n > 0')
    age_born = raw[['Age', 'Born']].apply(numeric).dropna()
    age_born_corr = age_born.corr().loc['Age', 'Born']
    source_dupes = int(raw.duplicated().sum())
    player_pair_dupes = int(raw.duplicated(['Player', 'Squad']).sum()) if 'Squad' in raw else None
    repeated_names = int(raw.Player.astype('string').str.strip().duplicated().sum())
    repeated_retained = int(frame._player_group.duplicated().sum())
    majority = class_balance['count'].idxmax()
    minority = class_balance['count'].idxmin()
    ratio = class_balance.loc[majority, 'count']/class_balance.loc[minority, 'count']
    source_pos = raw.Pos.astype('string').str.contains(',', regex=False).sum()
    eligible_pos = frame.Pos.astype('string').str.contains(',', regex=False).sum()
    dtype_counts = raw.dtypes.astype(str).value_counts().rename_axis('dtype').rename('columns').reset_index()
    display_summary = summary[['representation', 'model']].copy()
    for metric in ['macro_f1', 'balanced_accuracy', 'accuracy']:
        display_summary[metric] = [f'{m:.3f} ± {s:.3f}' for m, s in zip(summary[f'{metric}_mean'], summary[f'{metric}_std'])]
    comparison_text = []
    for name in ['Logistic regression', 'Random forest']:
        tab = summary.query('model == @name').set_index('representation')
        delta = tab.loc['per90', 'macro_f1_mean']-tab.loc['raw_counts', 'macro_f1_mean']
        pairs = paired.query('model == @name')
        wins = int((pairs['per90_minus_raw_macro_f1'] > 0).sum())
        comparison_text.append(f'- **{name}:** per-90 minus raw-count macro F1 is **{delta:+.3f}**; per-90 wins {wins}/5 paired folds.')
    metrics = summary.query('model != "Majority baseline"')
    best = metrics.loc[metrics.macro_f1_mean.idxmax()]
    rates_by_class = results.query('representation == "per90" and model == "Logistic regression"')
    class_metrics = pd.DataFrame({
        'Position': LABELS,
        'Per90 logistic recall (mean)': [f'{rates_by_class[f"{label}_recall"].mean():.3f}' for label in LABELS],
        'Per90 logistic F1 (mean)': [f'{rates_by_class[f"{label}_f1"].mean():.3f}' for label in LABELS],
    })
    description_rows = [{'Raw count': c, 'Per90 field': f'{c}_p90', 'Meaning': COUNT_DESCRIPTIONS[c]} for c in COUNTS]
    report = f'''# Position data preparation and raw-count versus per-90 experiment

## Deliverables and main finding

`data/players_position_cleaned_min180.csv` is the primary **per-90** dataset:
**{len(rate_view):,} rows × {rate_view.shape[1]} columns**, with target `Target` in DF/MF/FW.
`data/players_position_raw_counts_min180.csv` contains the identical players in the identical
order with raw event counts, for the controlled comparison. Both exclude player identifiers,
original/duplicate position labels, birth year, keeper statistics, and redundant merged fields.
Keep `Target` as **y**, never as an input feature. Neither file is globally imputed or scaled.

The strongest observed combination is **{best['model']} with {best['representation']}**
(mean macro F1 **{best['macro_f1_mean']:.3f}**). This is an exploratory comparison of fixed
configurations, not an independent final test or a claim that normalization always helps.

## 1. Source audit

Source: `{source_path.name}` (the full CSV extracted from `archive.zip`).
It contains **{len(raw):,} rows and {len(raw.columns)} columns**. It has {source_dupes}
exact duplicate records, {player_pair_dupes} repeated Player/Squad pairs, and {repeated_names}
additional rows sharing a stripped player name. Distinct club records are retained;
collapsing them without a documented aggregation rule would discard valid observations.
After filtering, {repeated_retained} extra records share names; these names are grouped in CV.

{md_table(dtype_counts)}

**{len(missing_raw)} columns contain missing values.** Keeper columns are mostly structurally
missing for outfield players. Ratios such as `G/SoT` and `SoT%` are often undefined when their
denominator is zero. The raw table has {int(raw.Age.isna().sum())} missing ages and
{int(raw.Born.isna().sum())} missing birth years. After filtering, **{int(rate_view.Age.isna().sum())}
ages remain missing**; they are retained for fold-local imputation. Audit files list every
column's original type, missing count, and cardinality, all {len(merged)} comparable merged
copies, and {len(identical)} exact same-type identical-column pairs. A suffix alone is not
proof of equality: `merged_column_audit.csv` measures agreement on overlapping nonmissing rows.

## 2. Every row-selection decision

{md_table(flow)}

The steps above are sequential, so removals do not double-count players. `Pos` is split at
the first comma, whitespace is stripped, and the first token becomes `Target`. Thus
`MF,FW` becomes MF, while `FW,MF` becomes FW. This changes no rows by itself:
{int(source_pos)} source records have multiple positions, and {int(eligible_pos)} retained
records receive their first listed role. Primary GK records are excluded. Rows whose
target or exposure cannot be established are excluded rather than having labels/minutes
invented. Exact duplicates are identified from the original source columns; the private
audit row index is excluded from that comparison.
There is no complete-case deletion based on predictors, outlier trimming, resampling,
deduplication by name alone, or label balancing by discarding majority examples.

## 3. Column choices, dates, and target leakage

- **Target:** preserve only the three-class `Target`. Drop `Pos`, `Pos_stats_keeper`,
  `Pos_stats_shooting`, `Pos_stats_playing_time`, `Pos_stats_misc`, and any other position
  copies from X. They directly disclose the answer. They are also absent from the exports.
- **Identifiers:** remove Player, row ranks, and Squad from predictors and exports. Player
  names are used only privately to separate repeated-player records across CV folds.
  Nationality is omitted as out-of-scope context, not described as a unique identifier.
- **Age versus birth year:** their Pearson correlation is **{age_born_corr:.4f}**. They encode
  almost the same underlying age information, so retain Age and drop Born and all copies.
  They are not claimed to be exact numerical duplicates: birthdays/reference dates can
  account for a one-year difference. There is **no explicit match-date column**, so date
  parsing is unnecessary. Treating birth year as a match date, or inventing a reference
  date to recompute age, would add unsupported assumptions.
- **Exposure:** retain exact Min as a numerical covariate in both feature sets and use it
  to calculate rates. Drop the rounded `90s` and redundant playing-time summaries.
- **Event statistics:** retain the 16 canonical counts listed below, or their recalculated
  rates, once per experiment. Drop existing rounded `Sh/90` and `SoT/90`, the merged copies,
  redundant sums/differences such as G+A and G-PK, and unstable derived conversion ratios.
- **Keeper/team context:** drop keeper fields, including structurally missing values after
  excluding GK. Additional team-outcome and playing-time columns are outside the explicit
  allowlist; using all 102 fields would mix redundant summaries into this focused comparison.
- **Goals and assists are valid inputs here:** the target is **position**, not goals/assists.
  Their correlation with a role is useful behavioral information, not direct target leakage.
  They would be leakage when predicting contemporaneous goal contributions instead.

Column exclusion and engineering change the number of columns, not the {len(frame):,}
eligible rows. `column_decisions.csv` records a decision and reason for **every original
column**. `feature_conversion_audit.csv` records invalid-to-missing conversions. Invalid or
negative event counts become missing predictors; zero counts stay zero. Nonfinite values
become missing. Missing, nonpositive, nonfinite, or unparseable ages remain missing as well.

## 4. Per-90 features

For each count **c**, use `c_p90 = c * 90 / Min`. The retained minimum is
{frame.Min.min():.0f} minutes, so no rate divides by zero. Exact minutes avoid rounding error
from the supplied `90s`. A missing count remains missing; it is never replaced by zero
without evidence. `TklW` measures **tackles won**, not every attempted tackle.

{md_table(pd.DataFrame(description_rows))}

The primary CSV contains Age, Min, Comp, these 16 rates, and Target. The comparison CSV
contains the same three common covariates, 16 original counts, and Target. Per-90 values
describe event intensity and reduce exposure-driven differences, but two 90s of playing
time can still give noisy rates. No rate clipping is performed. Because **Min is included
in both views**, counts and rates are mathematically convertible: the experiment tests a
representation/inductive-bias change, not the addition of new information.

## 5. Class imbalance and missing-data pipelines

{md_table(class_balance.reset_index().assign(percent=lambda t: t['percent'].map(lambda x: f'{x:.2f}%')))}

{majority} is {ratio:.2f} times as frequent as {minority}. A majority-only classifier can
reach {class_balance.loc[majority, 'percent']:.2f}% accuracy while never identifying the
other two roles. **Macro F1** gives each role equal weight and is the primary metric;
**balanced accuracy** averages class recalls. Per-class recall/F1 and the held-out confusion
matrices expose whether minority forwards are being missed.

Both learned models use `class_weight="balanced"`, recalculated from each training fold.
There is no oversampling/undersampling. Weighting may trade overall accuracy for minority
recall and does not guarantee equal performance. Natural validation prevalence is preserved.

Each sklearn Pipeline contains a ColumnTransformer. Numerical predictors receive
training-median imputation (`keep_empty_features=True`). Logistic regression then uses
StandardScaler, which is appropriate for its regularized coefficients. Random forests
receive imputation but no numerical scaling because their split thresholds do not require
common units. League (`Comp`) receives most-frequent imputation and OneHotEncoder with
`handle_unknown="ignore"`. Its categories, imputation values, and scaling moments are fitted
**only on the training fold**. A new validation league maps to an all-zero categorical vector.
All-missing training columns are preserved by the imputer rather than changing the schema.

The CSV intentionally keeps missing numerical values and the readable league label.
Exporting globally imputed/scaled values or fitting encoding before CV would contaminate
evaluation. Downstream users should fit the supplied pipeline on their training data;
the new CSV alone cannot contain a split-independent, already-fitted preprocessing step.

## 6. Raw counts versus per-90: calculated results

Use `StratifiedGroupKFold(5, shuffle=True, random_state=42)`, stratified by Target and
grouped by stripped Player name. All feature views and models use the **same saved indices**.
Repeated names never occur in both train and validation. Namesakes may be conservatively
grouped; a persistent player identifier would improve this design. Out-of-fold predictions
cover every retained row once. `fold_assignments.csv` permits reusing those splits with
either exported CSV, without exposing player identities.

Fixed models: logistic regression C=1, max_iter=2000; random forest 200 trees,
min_samples_leaf=3, seed=42; plus a most-frequent DummyClassifier. No hyperparameter search
or selection on these same validation predictions is used. Fold SDs are descriptive,
not confidence intervals, and overlapping training sets make folds dependent.

{md_table(display_summary)}

{chr(10).join(comparison_text)}

The average macro-F1 gains are small relative to the reported fold variation and are not
consistent wins in every fold. They do not establish statistically significant superiority.
Other metrics can move differently: check balanced accuracy and the class-specific scores
before choosing a representation for a particular application.

{md_table(class_metrics)}

![Class balance and feature comparison](class_balance_and_feature_comparison.png)

![Out-of-fold logistic confusion matrices](logistic_oof_confusion_matrices.png)

The primary CSV uses per-90 features as requested, even if a raw-count configuration scores
better. Report the empirical winner rather than assuming normalization improves accuracy.
The results identify football roles from same-season behavior and do not establish future
performance or causal effects. Primary-label assignment simplifies genuinely hybrid roles;
recorded position can also depend on the source's labeling convention.

## Reproduce and inspect the audit

From the outer FootballAIProject directory:

```sh
.venv/bin/python AI_Football_Project/prepare_position_data.py
```

The original extracted CSV, archive, existing test.py, and earlier notebooks are unchanged.
`preprocessing_manifest.json` records the source hash, versions, feature allowlists, and
output hashes. No identifiers or duplicate target columns enter either exported feature set.
Full supporting tables: [source schema](source_schema.csv), [merged copies](merged_column_audit.csv),
[identical columns](identical_column_pairs.csv), [column decisions](column_decisions.csv),
[feature conversions](feature_conversion_audit.csv), [fold metrics](raw_vs_per90_fold_metrics.csv),
[paired differences](raw_vs_per90_paired_differences.csv), and [fold audit](fold_audit.csv).

Method references: [sklearn pipelines](https://scikit-learn.org/stable/modules/compose.html),
[stratified grouped validation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html),
and [balanced accuracy](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html).
'''
    (report_dir/'preprocessing_report.md').write_text(report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT/'data'/'players_data-2026_2027.csv')
    args = parser.parse_args()
    source_path = args.source.resolve()
    data_dir = ROOT/'data'
    report_dir = ROOT/'preprocessing_report'
    data_dir.mkdir(exist_ok=True)
    report_dir.mkdir(exist_ok=True)
    source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
    raw = pd.read_csv(source_path, encoding='utf-8-sig', low_memory=False)
    schema, merged, identical = audit_source(raw)
    frame, raw_view, rate_view, flow, conversion = prepare(raw)
    results, predictions, fold_audit = compare(frame, raw_view, rate_view)
    summary = results.groupby(['representation', 'model'], sort=False).agg(
        macro_f1_mean=('macro_f1', 'mean'), macro_f1_std=('macro_f1', 'std'),
        balanced_accuracy_mean=('balanced_accuracy', 'mean'), balanced_accuracy_std=('balanced_accuracy', 'std'),
        accuracy_mean=('accuracy', 'mean'), accuracy_std=('accuracy', 'std')).reset_index()
    class_balance = frame.Target.value_counts().reindex(LABELS).rename('count').to_frame()
    class_balance.index.name = 'Target'
    class_balance['percent'] = 100*class_balance['count']/len(frame)
    paired = results.pivot(index=['model', 'fold'], columns='representation', values='macro_f1').reset_index()
    paired['per90_minus_raw_macro_f1'] = paired['per90']-paired['raw_counts']
    outputs = {
        'per90': data_dir/'players_position_cleaned_min180.csv',
        'raw_counts': data_dir/'players_position_raw_counts_min180.csv',
    }
    rate_view.to_csv(outputs['per90'], index=False)
    raw_view.to_csv(outputs['raw_counts'], index=False)
    tables = {
        'source_schema': schema, 'merged_column_audit': merged, 'identical_column_pairs': identical,
        'column_decisions': decisions(raw), 'row_filter_audit': flow, 'feature_conversion_audit': conversion,
        'class_balance': class_balance.reset_index(), 'raw_vs_per90_fold_metrics': results,
        'raw_vs_per90_summary': summary, 'raw_vs_per90_paired_differences': paired,
        'fold_audit': fold_audit, 'out_of_fold_predictions': predictions,
        'fold_assignments': pd.DataFrame({'cleaned_row': np.arange(len(frame)), 'fold': predictions['fold']}),
        'eligible_descriptive_statistics': frame[COMMON_NUM+COUNTS+[f'{c}_p90' for c in COUNTS]].describe().T.reset_index(names='feature'),
    }
    for name, table in tables.items():
        table.to_csv(report_dir/f'{name}.csv', index=False)
    plots(frame, results, summary, predictions, report_dir)
    write_report(raw, frame, rate_view, flow, conversion, schema, merged, identical,
                 results, summary, class_balance, paired, report_dir, source_path)
    manifest = {
        'created_at_utc': datetime.now(timezone.utc).isoformat(), 'source': str(source_path),
        'source_sha256': source_hash, 'source_shape': list(raw.shape), 'eligible_n': len(frame),
        'minimum_minutes': 180, 'target': 'Target', 'classes': LABELS,
        'feature_sets': {'raw_counts': raw_view.drop(columns='Target').columns.tolist(),
                         'per90': rate_view.drop(columns='Target').columns.tolist()},
        'count_statistics': COUNTS, 'rate_formula': 'count * 90 / Min',
        'cv': 'StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)',
        'grouping': 'stripped Player names, not exported as predictors', 'seed': SEED,
        'convergence_warnings': int(results.convergence_warnings.sum()),
        'versions': {'python': platform.python_version(), **{p: importlib.metadata.version(p)
                     for p in ['pandas', 'numpy', 'scikit-learn', 'matplotlib', 'seaborn']}},
        'outputs': {name: {'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
                    for name, p in outputs.items()},
    }
    assert hashlib.sha256(source_path.read_bytes()).hexdigest() == source_hash
    (report_dir/'preprocessing_manifest.json').write_text(json.dumps(manifest, indent=2))
    print(flow.to_string(index=False))
    print('\nClass balance:\n'+class_balance.round(3).to_string())
    print('\nRaw versus per90:\n'+summary.round(4).to_string(index=False))
    print(f'\nPrimary CSV: {outputs["per90"]}')
    print(f'Report: {report_dir / "preprocessing_report.md"}')
    print(f'Convergence warnings: {manifest["convergence_warnings"]}')


if __name__ == '__main__':
    main()
