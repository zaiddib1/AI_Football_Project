"""Validate a WhoScored export before joining new statistics to the frozen experiment.

This script reads an export and its provenance JSON; it does not bypass site access
controls. It never updates the source CSV or any existing notebook.
"""
from __future__ import annotations

import argparse
from datetime import date, datetime, timezone
import hashlib
import json
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd

from enrich_player_biometrics import normalized, validated_clean_views

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
REPORTS = ROOT / 'whoscored_enrichment'
RAW = DATA / 'players_data-2026_2027.csv'
FROZEN_SHA256 = 'a404aac3864ef5a1e51b68af7d8e0b66bb5da505276977aca4c9411f53833b69'
SEASON = '2026/2027'
# Publication time is an upper bound, not a claim about the last included match.
SNAPSHOT_PUBLISHED = '2026-11-02T08:18:27.330+00:00'
COMPETITIONS = {
    'eng Premier League', 'es La Liga', 'de Bundesliga', 'it Serie A', 'fr Ligue 1',
}
COUNT_METRICS = {
    'key_passes', 'passes', 'accurate_passes', 'successful_dribbles',
    'aerial_duels_won', 'clearances', 'blocks', 'dispossessions',
    'poor_controls', 'long_balls', 'through_balls',
}
PERCENT_METRICS = {'pass_completion_pct'}
MODEL_METRICS = {'xg'}  # xA requires a separately verified field and definition.
METRICS = COUNT_METRICS | PERCENT_METRICS | MODEL_METRICS
IDENTITY = {'Player', 'Squad', 'Comp', 'MP', 'Min', 'source_player_id'}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def season_label(value):
    value = str(value).strip().replace('-', '/')
    return '2026/2027' if value in {'2026/2027', '2026/27'} else value


def validate_metadata(meta):
    required = {'source', 'season', 'competition', 'scope', 'cutoff_date',
                'retrieved_at_utc', 'source_urls', 'verification_notes', 'metrics'}
    missing = required - meta.keys()
    if missing:
        raise ValueError(f'Missing provenance fields: {sorted(missing)}')
    if meta['source'] != 'WhoScored' or season_label(meta['season']) != SEASON:
        raise ValueError('Only explicitly identified WhoScored 2026/2027 exports are accepted.')
    if meta['competition'] not in COMPETITIONS or meta['scope'] != 'domestic_league_only':
        raise ValueError('Specify one supported domestic league; exclude cups and all-competition totals.')
    if not isinstance(meta['cutoff_date'], str):
        raise ValueError('Source reporting cutoff is unknown; retrieval time must not be substituted for it.')
    cutoff = date.fromisoformat(meta['cutoff_date'])
    if not date(2026, 7, 1) <= cutoff <= date.fromisoformat(SNAPSHOT_PUBLISHED[:10]):
        raise ValueError('Export cutoff is outside the frozen snapshot window; later data cannot be attached.')
    retrieved = datetime.fromisoformat(meta['retrieved_at_utc'].replace('Z', '+00:00'))
    if retrieved.tzinfo is None or retrieved.date() < cutoff:
        raise ValueError('Retrieval timestamp must have a timezone and cannot precede the cutoff.')
    if not isinstance(meta['source_urls'], list) or not meta['source_urls']:
        raise ValueError('Provide the actual season/statistics page URLs, not an unlabelled download.')
    for url in meta['source_urls']:
        parsed = urlparse(url)
        if parsed.scheme != 'https' or parsed.hostname not in {'whoscored.com', 'www.whoscored.com'}:
            raise ValueError('Each source URL must be an HTTPS WhoScored page.')
    if not str(meta['verification_notes']).strip():
        raise ValueError('Document how the season, competition, cutoff and units were checked on the page.')
    if not isinstance(meta['metrics'], dict) or not meta['metrics']:
        raise ValueError('Declare the column and units for every imported metric.')
    for metric, spec in meta['metrics'].items():
        if metric not in METRICS or not isinstance(spec, dict):
            raise ValueError(f'Unsupported metric {metric!r}; do not guess source definitions.')
        if not spec.get('column') or not spec.get('definition'):
            raise ValueError(f'{metric}: supply its source column and definition.')
        expected_units = {'percentage'} if metric in PERCENT_METRICS else {'total', 'per90'}
        if spec.get('units') not in expected_units:
            raise ValueError(f'{metric}: units must be {sorted(expected_units)}. Per-game values are rejected.')
    columns = [spec['column'] for spec in meta['metrics'].values()]
    if len(set(columns)) != len(columns) or set(columns) & IDENTITY:
        raise ValueError('Metric columns must be distinct and separate from identity/exposure fields.')
    return meta


def numeric_column(series, name, *, integer=False):
    text = series.astype('string').str.replace(',', '', regex=False).str.strip()
    empty = series.isna() | text.isin(['', '-', '—'])
    values = pd.to_numeric(text.mask(empty), errors='coerce').astype(float)
    if ((~empty & values.isna()) | (values.notna() & ~np.isfinite(values)) | values.lt(0)).any():
        raise ValueError(f'{name}: unexpected nonnumeric, infinite or negative values.')
    if integer and (values.dropna() % 1).ne(0).any():
        raise ValueError(f'{name}: expected integer totals.')
    return values


def prepare_export(frame, meta):
    validate_metadata(meta)
    frame = frame.rename(columns=meta.get('columns', {})).copy()
    required = IDENTITY | {spec['column'] for spec in meta['metrics'].values()}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f'Export missing columns: {sorted(missing)}')
    if frame.empty or frame.columns.duplicated().any():
        raise ValueError('Export is empty or contains duplicate column labels.')
    if not frame.Comp.eq(meta['competition']).all():
        raise ValueError('Export competition values disagree with its provenance.')
    if 'Season' in frame and not frame.Season.map(season_label).eq(SEASON).all():
        raise ValueError('Export rows contain a different season.')
    if frame.source_player_id.isna().any():
        raise ValueError('Every source record needs a stable player ID.')
    for col in ['Player', 'Squad']:
        if frame[col].isna().any() or frame[col].astype('string').str.strip().eq('').any():
            raise ValueError(f'{col} cannot be missing.')
    for col in ['MP', 'Min']:
        frame[col] = numeric_column(frame[col], col, integer=True)
        if frame[col].isna().any() or frame[col].le(0).any():
            raise ValueError(f'{col}: positive, unrounded exposure totals are required.')
    if frame.Min.gt(90 * frame.MP).any():
        raise ValueError('Minutes exceed 90 per league appearance; check the exposure convention.')
    if frame.duplicated(['source_player_id', 'Comp', 'Squad']).any():
        raise ValueError('Repeated source records: combine categories by verified ID before importing.')
    aliases = meta.get('club_aliases', {})
    if not isinstance(aliases, dict) or any(not str(v).strip() for v in aliases.values()):
        raise ValueError('Club aliases must explicitly map source club names to raw CSV club names.')
    frame['match_name'] = frame.Player.map(normalized)
    frame['match_club'] = frame.Squad.replace(aliases).map(normalized)
    features = []
    for metric, spec in meta['metrics'].items():
        unit = spec['units']
        values = numeric_column(frame[spec['column']], metric,
                                integer=metric in COUNT_METRICS and unit == 'total')
        if metric in PERCENT_METRICS:
            if values.dropna().gt(100).any():
                raise ValueError(f'{metric}: percentages must be on a 0–100 scale.')
            output = f'ws_{metric}'
            frame[output] = values
            features.append(output)
        else:
            rate = f'ws_{metric}_p90'
            frame[rate] = values * 90 / frame.Min if unit == 'total' else values
            features.append(rate)
            if unit == 'total':
                total = f'ws_{metric}_total'
                frame[total] = values
                features.append(total)
    return frame, features


def merge_export(base, export, features):
    """Names and clubs must match exactly after normalization; corroborate exposure."""
    result = base.copy(deep=True)
    if any(c in base for c in features):
        raise ValueError('One of the imported feature columns already exists.')
    for col in features:
        result[col] = np.nan
    audit = []
    groups = {}
    for i, row in export.iterrows():
        groups.setdefault((row.match_name, row.match_club, row.Comp), []).append(i)
    base_keys = pd.DataFrame({
        'name': base.Player.map(normalized), 'club': base.Squad.map(normalized), 'Comp': base.Comp})
    duplicates = base_keys.duplicated(keep=False)
    for i, row in base.iterrows():
        key = (normalized(row.Player), normalized(row.Squad), row.Comp)
        candidates = groups.get(key, [])
        status, external = 'unmatched', None
        if candidates and (len(candidates) != 1 or duplicates.loc[i]):
            status = 'ambiguous_identity'
        elif candidates:
            external = export.loc[candidates[0]]
            if row.MP != external.MP or row.Min != external.Min:
                status = 'exposure_mismatch'
            elif ('Born' in export and pd.notna(external.Born) and
                  (pd.isna(row.Born) or str(int(row.Born)) != str(external.Born).removesuffix('.0'))):
                status = 'birthyear_mismatch'
            else:
                status = 'matched_exact_identity_and_exposure'
                result.loc[i, features] = external[features].to_numpy(dtype=float)
        audit.append({'source_row': i, 'Player': row.Player, 'Squad': row.Squad,
                      'Comp': row.Comp, 'status': status, 'raw_MP': row.MP, 'raw_Min': row.Min,
                      'ws_player_id': external.source_player_id if external is not None else None,
                      'ws_MP': external.MP if external is not None else np.nan,
                      'ws_Min': external.Min if external is not None else np.nan})
    pd.testing.assert_frame_equal(result[base.columns], base)
    return result, pd.DataFrame(audit)


def run(export_files, metadata_files):
    if digest(RAW) != FROZEN_SHA256:
        raise ValueError('The base snapshot changed. Revalidate its season, publication date and cohort first.')
    if len(export_files) != len(metadata_files):
        raise ValueError('Provide one --metadata file for every --csv file, in the same order.')
    frames, features, sources = [], [], []
    for export_file, metadata_file in zip(export_files, metadata_files):
        metadata = validate_metadata(json.loads(Path(metadata_file).read_text()))
        source, source_features = prepare_export(pd.read_csv(export_file), metadata)
        frames.append(source)
        features.extend(c for c in source_features if c not in features)
        sources.append({'export_sha256': digest(export_file),
                        'metadata_sha256': digest(metadata_file), 'metadata': metadata})
    source = pd.concat(frames, ignore_index=True)
    cutoffs = {record['metadata']['cutoff_date'] for record in sources}
    if len(cutoffs) != 1:
        raise ValueError('Use a common cutoff date across league exports.')
    raw = pd.read_csv(RAW)
    base_path = DATA / 'players_data-2026_2027_with_biometrics.csv'
    base = pd.read_csv(base_path)
    pd.testing.assert_frame_equal(base[raw.columns], raw)
    enriched, audit = merge_export(base, source, features)
    usable = enriched[features].notna().any(axis=1)
    if not usable.any():
        raise ValueError('No verified usable matches; no enriched dataset was written. Check identities and exposure totals.')
    # Use the existing validated cohort reconstruction, avoiding a second row-filter implementation.
    clean_views, eligible = validated_clean_views(raw, enriched)
    outputs = {'full': DATA / 'players_data-2026_2027_with_biometrics_whoscored.csv'}
    tables = {'full': enriched}
    rate_features = [c for c in features if not c.endswith('_total')]
    for kind, view in clean_views.items():
        for col in ['height_cm', 'weight_kg'] + rate_features:
            if col in enriched:
                view[col] = enriched.loc[eligible.source_row, col].to_numpy()
        outputs[kind] = DATA / f'players_position_{kind}_min180_with_biometrics_whoscored.csv'
        tables[kind] = view
    REPORTS.mkdir(exist_ok=True)
    # All checks above complete before the first CSV write.
    for name, path in outputs.items():
        tables[name].to_csv(path, index=False)
    audit.to_csv(REPORTS / 'player_match_audit.csv', index=False)
    manifest = {
        'status': 'imported', 'imported_at_utc': datetime.now(timezone.utc).isoformat(),
        'raw_sha256': FROZEN_SHA256, 'raw_kaggle_version': 10,
        'raw_snapshot_published_at_utc': SNAPSHOT_PUBLISHED,
        'raw_exact_last_match_date': None,
        'sources': sources,
        'verification': 'Export provenance plus exact player/club/league and MP/Min reconciliation; no live website verification',
        'usable_records': int(usable.sum()), 'total_records': len(raw),
        'primary_usable_records': int(usable.loc[eligible.source_row].sum()),
        'match_status_counts': audit.status.value_counts().to_dict(),
        'features': features, 'ratings_or_external_positions_imported': False,
        'outputs': {name: {'file': str(path), 'sha256': digest(path)} for name, path in outputs.items()},
    }
    (REPORTS / 'import_manifest.json').write_text(json.dumps(manifest, indent=2))
    print(f'Imported usable new statistics for {usable.sum()} / {len(raw)} records.')
    for path in outputs.values():
        print(path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', required=True, type=Path, action='append', help='WhoScored player export; repeat for additional leagues')
    parser.add_argument('--metadata', required=True, type=Path, action='append', help='One provenance JSON per export, in the same order')
    args = parser.parse_args()
    try:
        run(args.csv, args.metadata)
    except (ValueError, KeyError, OSError) as error:
        parser.exit(2, f'Import stopped: {error}\n')
