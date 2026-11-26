"""Add sourced, conservatively matched height and weight; preserve the original datasets.

External positions and game ratings are never imported. Fuzzy suggestions are review-only.
The cached identity data supplies birth dates, not measurements; listed measurements come
from a pinned EAFC26 profile snapshot, with explicit official-club overrides where checked.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from difflib import SequenceMatcher
from functools import lru_cache
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import shutil
import unicodedata

import numpy as np
import pandas as pd
import pycountry
import requests

ROOT = Path(__file__).resolve().parent
DATA = ROOT/'data'
SOURCES = DATA/'biometrics_sources'
REPORTS = ROOT/'biometrics_enrichment'
PHYSICAL_HANDLE = 'flynn28/eafc26-player-database/versions/2'
PHYSICAL_URL = 'https://www.kaggle.com/datasets/flynn28/eafc26-player-database/versions/2'
PHYSICAL_SNAPSHOT = '2026-04-02'
IDENTITY_COMMIT = '2e5b27e7b28515fff7103dc069a0b1e4d428d755'
IDENTITY_URL = ('https://raw.githubusercontent.com/LakshmiKanth11/EA_FC_ANALYSIS/'
                + IDENTITY_COMMIT + '/EA_FC_PLAYERS_DATASET_AND_%20DASHBOARD/data/players.csv')
IDENTITY_README = 'https://github.com/LakshmiKanth11/EA_FC_ANALYSIS'
COUNTS = ['Sh', 'SoT', 'Crs', 'Int', 'TklW', 'Fls', 'Fld', 'Off',
          'Gls', 'Ast', 'PK', 'PKatt', 'CrdY', 'CrdR', '2CrdY', 'OG']
COUNTRY_ALIASES = {
    'england': 'ENG', 'scotland': 'SCO', 'wales': 'WAL', 'northernireland': 'NIR',
    'kosovo': 'XKX', 'republicofireland': 'IRL', 'ireland': 'IRL',
    'korearepublic': 'KOR', 'southkorea': 'KOR', 'koreadpr': 'PRK', 'northkorea': 'PRK',
    'ivorycoast': 'CIV', 'cotedivoire': 'CIV', 'drcongo': 'COD', 'congodr': 'COD',
    'congo': 'COG', 'caboverde': 'CPV', 'capeverdeislands': 'CPV', 'capeverde': 'CPV',
    'czechrepublic': 'CZE', 'chinapr': 'CHN', 'chinesetaipei': 'TWN',
    'tanzania': 'TZA', 'bolivia': 'BOL', 'venezuela': 'VEN', 'iran': 'IRN',
    'russia': 'RUS', 'syria': 'SYR', 'palestine': 'PSE', 'holland': 'NLD',
    'turkey': 'TUR', 'stlucia': 'LCA', 'stkittsandnevis': 'KNA',
}


@lru_cache(maxsize=None)
def normalized(value: str) -> str:
    text = str(value).lower().translate(str.maketrans({
        'ø': 'o', 'ł': 'l', 'đ': 'd', 'ß': 'ss', 'æ': 'ae', 'œ': 'oe', 'ð': 'd', 'þ': 'th'}))
    return ''.join(c for c in unicodedata.normalize('NFKD', text)
                   if c.isalnum() and not unicodedata.combining(c))


@lru_cache(maxsize=None)
def country_code(name: str) -> str:
    if not name:
        return ''
    key = normalized(name)
    if key in COUNTRY_ALIASES:
        return COUNTRY_ALIASES[key]
    try:
        return pycountry.countries.lookup(name).alpha_3
    except LookupError:
        return ''


def original_country(value) -> str:
    if pd.isna(value):
        return ''
    prefix = str(value).strip().split()[0].lower()
    special = {'eng': 'ENG', 'sct': 'SCO', 'wls': 'WAL', 'nir': 'NIR', 'xk': 'XKX'}
    if prefix in special:
        return special[prefix]
    entry = pycountry.countries.get(alpha_2=prefix.upper())
    return entry.alpha_3 if entry else ''


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def download_sources(refresh=False):
    SOURCES.mkdir(exist_ok=True)
    metadata_path = SOURCES/'source_manifest.json'
    previous = json.loads(metadata_path.read_text()) if metadata_path.is_file() else {}
    physical_path = SOURCES/'eafc26_men_v2.csv'
    identity_path = SOURCES/'eafc27_identity_2e5b27e7.csv'
    fetched = datetime.now(timezone.utc).isoformat()
    downloaded_keys = set()
    if refresh or not physical_path.is_file():
        import kagglehub
        downloaded = Path(kagglehub.dataset_download(PHYSICAL_HANDLE))/'EAFC26-Men.csv'
        shutil.copy2(downloaded, physical_path)
        downloaded_keys.add('physical')
    if refresh or not identity_path.is_file():
        response = requests.get(IDENTITY_URL, timeout=60)
        response.raise_for_status()
        identity_path.write_bytes(response.content)
        downloaded_keys.add('identity')
    records = {}
    for key, path, url in [('physical', physical_path, PHYSICAL_URL), ('identity', identity_path, IDENTITY_URL)]:
        digest = sha256(path)
        if not refresh and key in previous and previous[key]['sha256'] != digest:
            raise ValueError(f'Cached {key} source differs from its recorded checksum; inspect it or use --refresh-sources.')
        records[key] = {'file': str(path), 'source_url': url, 'sha256': digest,
                        'retrieved_at_utc': fetched if key in downloaded_keys else
                        previous.get(key, {}).get('retrieved_at_utc', fetched)}
    records['physical'].update({'dataset_handle': PHYSICAL_HANDLE,
                               'snapshot_date': PHYSICAL_SNAPSHOT, 'publisher_license_label': 'GPL 3'})
    records['identity'].update({'repository_commit': IDENTITY_COMMIT,
                               'publisher_snapshot_date': '2026-09-12',
                               'role': 'Names and DOB only; height/weight columns in this source are empty.'})
    metadata_path.write_text(json.dumps(records, indent=2))
    return physical_path, identity_path, records


def load_profiles(physical_path, identity_path):
    p = pd.read_csv(physical_path, usecols=['ID', 'Name', 'GENDER', 'Nation', 'Team', 'Height', 'Weight', 'url'])
    p = p.loc[p.GENDER.eq('M')].copy()
    identity = pd.read_csv(identity_path, usecols=[
        'player_id', 'first_name', 'last_name', 'common_name', 'birthdate', 'nationality', 'club', 'gender'])
    identity = identity.loc[identity.gender.eq("Men's Football")].copy()
    if p.ID.duplicated().any() or identity.player_id.duplicated().any():
        raise ValueError('External player IDs are not unique; do not perform a many-to-many join.')
    # The downloaded DOB representation is day-month-year, despite the repository's ISO description.
    identity['date_of_birth'] = pd.to_datetime(identity.birthdate, format='%d-%m-%Y', errors='coerce')
    identity['date_of_birth'] = identity.date_of_birth.fillna(pd.to_datetime(identity.birthdate, format='%Y-%m-%d', errors='coerce'))
    p = p.merge(identity, how='left', left_on='ID', right_on='player_id', validate='one_to_one')
    p['birth_year'] = p.date_of_birth.dt.year
    p['listed_height_cm'] = pd.to_numeric(p.Height.str.extract(r'([0-9]+(?:\.[0-9]+)?)\s*cm\b', expand=False), errors='coerce')
    p['listed_weight_kg'] = pd.to_numeric(p.Weight.str.extract(r'([0-9]+(?:\.[0-9]+)?)\s*kg\b', expand=False), errors='coerce')
    # Broad unit/error checks, not medical reference ranges.
    p['listed_height_cm'] = p.listed_height_cm.where(p.listed_height_cm.between(130, 230))
    p['listed_weight_kg'] = p.listed_weight_kg.where(p.listed_weight_kg.between(40, 180))
    records = {}
    exact, name_only, birth_country = defaultdict(set), defaultdict(set), defaultdict(set)
    for _, row in p.iterrows():
        player_id = int(row.ID)
        aliases = {str(row.Name)}
        if pd.notna(row.common_name):
            aliases.add(str(row.common_name))
        if pd.notna(row.first_name) and pd.notna(row.last_name):
            aliases.add(f'{row.first_name} {row.last_name}')
        keys = {normalized(n) for n in aliases if n and n != 'nan'}
        nationalities = {country_code(str(x)) for x in [row.Nation, row.nationality] if pd.notna(x)}-{''}
        clubs = {normalized(str(x)) for x in [row.Team, row.club] if pd.notna(x)}
        year = int(row.birth_year) if pd.notna(row.birth_year) else None
        records[player_id] = {
            'id': player_id, 'name': row.Name, 'aliases': sorted(aliases), 'name_keys': keys,
            'nationality': row.Nation, 'country_codes': nationalities, 'club': row.Team, 'club_keys': clubs,
            'dob': row.date_of_birth.strftime('%Y-%m-%d') if pd.notna(row.date_of_birth) else None,
            'birth_year': year, 'height_cm': row.listed_height_cm, 'weight_kg': row.listed_weight_kg,
            'profile_url': row.url,
        }
        for name in keys:
            name_only[name].add(player_id)
            if year is not None:
                for nation in nationalities:
                    exact[(name, year, nation)].add(player_id)
        if year is not None:
            for nation in nationalities:
                birth_country[(year, nation)].add(player_id)
    return records, {'exact': exact, 'name_only': name_only, 'birth_country': birth_country}, p


def choose_match(name, birth_year, nationality, club, records, index):
    if pd.isna(name) or pd.isna(birth_year) or not nationality:
        return None, 'identity_missing', []
    try:
        numeric_year = float(birth_year)
    except (ValueError, TypeError):
        return None, 'identity_missing', []
    if not np.isfinite(numeric_year) or not numeric_year.is_integer():
        return None, 'identity_missing', []
    key = normalized(str(name))
    candidates = sorted(index['exact'].get((key, int(numeric_year), nationality), set()))
    if len(candidates) == 1:
        return candidates[0], 'matched_exact_name_birthyear_country', candidates
    if len(candidates) > 1:
        club_key = normalized(str(club)) if pd.notna(club) else ''
        with_club = [i for i in candidates if club_key and club_key in records[i]['club_keys']]
        if len(with_club) == 1:
            return with_club[0], 'matched_exact_with_club_tiebreak', candidates
        return None, 'ambiguous', candidates
    known_name = index['name_only'].get(key, set())
    if known_name:
        with_dob = [i for i in known_name if records[i]['birth_year'] is not None]
        if not with_dob:
            return None, 'unmatched_source_birthdate_unavailable', []
        same_year = [i for i in with_dob if records[i]['birth_year'] == int(numeric_year)]
        if same_year:
            return None, 'unmatched_nationality_conflict', []
        return None, 'unmatched_birthyear_conflict', []
    return None, 'unmatched_name', []


def review_suggestions(row, nation, records, index, accepted_candidates):
    if pd.isna(row.Born) or not nation:
        return []
    candidates = set(accepted_candidates) or index['birth_country'].get((int(row.Born), nation), set())
    name_key = normalized(str(row.Player))
    ranked = []
    for player_id in candidates:
        score = max(SequenceMatcher(None, name_key, key).ratio() for key in records[player_id]['name_keys'])
        if score >= .65:
            ranked.append((score, player_id))
    return sorted(ranked, reverse=True)[:3]


def enrich(original, records, index, overrides, profile_retrieved_at):
    rows, suggestions, conflicts = [], [], []
    for source_row, row in original.iterrows():
        nation = original_country(row.Nation)
        player_id, status, candidates = choose_match(row.Player, row.Born, nation, row.Squad, records, index)
        result = {
            'source_row': source_row, 'Player': row.Player, 'Born': row.Born, 'Nation': row.Nation,
            'Squad': row.Squad, 'biometrics_match_status': status, 'biometrics_player_id': player_id,
            'height_cm': np.nan, 'weight_kg': np.nan, 'profile_listed_height_cm': np.nan,
            'profile_listed_weight_kg': np.nan, 'height_cm_source_url': None, 'weight_kg_source_url': None,
            'biometrics_source_snapshot_date': None, 'biometrics_source_dob': None,
            'biometrics_source_name': None, 'biometrics_profile_url': None,
            'biometrics_quality': 'not_matched', 'biometrics_profile_retrieved_at_utc': None,
            'biometrics_official_checked_on': None,
        }
        if player_id is not None:
            profile = records[player_id]
            result.update({
                'height_cm': profile['height_cm'], 'weight_kg': profile['weight_kg'],
                'profile_listed_height_cm': profile['height_cm'], 'profile_listed_weight_kg': profile['weight_kg'],
                'height_cm_source_url': PHYSICAL_URL, 'weight_kg_source_url': PHYSICAL_URL,
                'biometrics_source_snapshot_date': PHYSICAL_SNAPSHOT,
                'biometrics_source_dob': profile['dob'], 'biometrics_source_name': profile['name'],
                'biometrics_profile_url': profile['profile_url'], 'biometrics_quality': 'published_profile_listing',
                'biometrics_profile_retrieved_at_utc': profile_retrieved_at,
            })
            if player_id in overrides:
                checked = overrides[player_id]
                if checked['dob'] != profile['dob']:
                    raise ValueError(f'Official override DOB conflicts with profile ID {player_id}')
                for feature in ['height_cm', 'weight_kg']:
                    value = checked[feature]
                    if pd.isna(value):
                        continue
                    if pd.notna(result[feature]) and not np.isclose(result[feature], value):
                        conflicts.append({'source_row': source_row, 'Player': row.Player, 'field': feature,
                                          'profile_value': result[feature], 'official_value': value,
                                          'official_url': checked['source_url'], 'resolution': 'Prefer official club listing'})
                    result[feature] = value
                    result[f'{feature}_source_url'] = checked['source_url']
                result['biometrics_quality'] = 'official_club_listing_checked'
                result['biometrics_official_checked_on'] = checked['checked_on']
                # This is a listing retrieved now, not a known measurement date.
                result['biometrics_source_snapshot_date'] = None
        else:
            for rank, (score, candidate_id) in enumerate(review_suggestions(row, nation, records, index, candidates), 1):
                candidate = records[candidate_id]
                suggestions.append({'source_row': source_row, 'Player': row.Player, 'Born': row.Born,
                    'Nation': row.Nation, 'Squad': row.Squad, 'rank': rank, 'suggested_player_id': candidate_id,
                    'suggested_name': candidate['name'], 'suggested_birthdate': candidate['dob'],
                    'suggested_nationality': candidate['nationality'], 'suggested_club': candidate['club'],
                    'name_similarity_for_review_only': score, 'source_profile_url': candidate['profile_url'],
                    'action': 'Not merged; requires independent identity review'})
        rows.append(result)
    audit = pd.DataFrame(rows)
    audit['biometrics_player_id'] = audit.biometrics_player_id.astype('Int64')
    new_columns = [c for c in audit if c not in ['source_row', 'Player', 'Born', 'Nation', 'Squad']]
    enriched = original.copy(deep=True)
    for column in new_columns:
        enriched[column] = audit[column].to_numpy()
    return enriched, audit, pd.DataFrame(suggestions), pd.DataFrame(conflicts)


def validated_clean_views(original, enriched):
    """Prove row alignment before attaching metadata to identifier-free feature CSVs."""
    frame = original.loc[~original.duplicated()].copy()
    frame['source_row'] = frame.index
    frame['Target'] = frame.Pos.astype('string').str.split(',', n=1).str[0].str.strip().str.upper()
    frame['Min'] = pd.to_numeric(frame.Min.astype('string').str.replace(',', '', regex=False), errors='coerce')
    player_ok = frame.Player.notna() & frame.Player.astype('string').str.strip().ne('')
    frame = frame.loc[frame.Target.isin(['DF', 'MF', 'FW']) & frame.Min.ge(180) & player_ok].copy()
    for feature in ['Age']+COUNTS:
        text = frame[feature].astype('string').str.replace(',', '', regex=False)
        if feature == 'Age':
            text = text.str.extract(r'^([0-9]+(?:\.[0-9]+)?)', expand=False)
        frame[feature] = pd.to_numeric(text, errors='coerce').replace([np.inf, -np.inf], np.nan)
        frame[feature] = frame[feature].mask(frame[feature].le(0) if feature == 'Age' else frame[feature].lt(0))
    for feature in COUNTS:
        frame[f'{feature}_p90'] = 90*frame[feature]/frame.Min
    views = {}
    for suffix in ['cleaned', 'raw_counts']:
        existing_path = DATA/f'players_position_{suffix}_min180.csv'
        if not existing_path.is_file():
            continue
        existing = pd.read_csv(existing_path)
        expected = frame[existing.columns].reset_index(drop=True)
        for column in existing:
            if column in ['Comp', 'Target']:
                if not existing[column].astype('string').equals(expected[column].astype('string')):
                    raise ValueError(f'Cleaned CSV row alignment failed for {column}; do not attach biometrics by position.')
            else:
                np.testing.assert_allclose(existing[column], expected[column], equal_nan=True, rtol=1e-10, atol=1e-10)
        updated = existing.copy()
        for feature in ['height_cm', 'weight_kg']:
            updated[feature] = enriched.loc[frame.source_row, feature].to_numpy()
        views[suffix] = updated
    return views, frame


def table(frame):
    def render(value):
        return '—' if pd.isna(value) else str(value).replace('|', '\\|')
    return '\n'.join(['| '+' | '.join(frame.columns)+' |',
                      '| '+' | '.join(['---']*len(frame.columns))+' |',
                      *['| '+' | '.join(render(v) for v in row)+' |' for row in frame.itertuples(index=False, name=None)]])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--refresh-sources', action='store_true')
    args = parser.parse_args()
    REPORTS.mkdir(exist_ok=True)
    source = DATA/'players_data-2026_2027.csv'
    original_hash = sha256(source)
    original = pd.read_csv(source, encoding='utf-8-sig')
    physical, identity, source_metadata = download_sources(args.refresh_sources)
    records, index, profiles = load_profiles(physical, identity)
    override_file = REPORTS/'official_overrides.csv'
    overrides = {}
    if override_file.is_file():
        for row in pd.read_csv(override_file).to_dict('records'):
            overrides[int(row['player_id'])] = row
    enriched, audit, review, conflicts = enrich(original, records, index, overrides,
                                               source_metadata['physical']['retrieved_at_utc'])
    views, eligible = validated_clean_views(original, enriched)
    pd.testing.assert_frame_equal(enriched[original.columns], original)
    assert len(enriched) == len(original) and enriched.index.equals(original.index)
    accepted = audit.biometrics_match_status.str.startswith('matched_')
    assert enriched.loc[~accepted, ['height_cm', 'weight_kg']].isna().all().all()
    for i, row in audit.loc[accepted].iterrows():
        profile = records[int(row.biometrics_player_id)]
        assert int(original.loc[i, 'Born']) == profile['birth_year']
        assert original_country(original.loc[i, 'Nation']) in profile['country_codes']
        assert normalized(str(original.loc[i, 'Player'])) in profile['name_keys']
    full_output = DATA/'players_data-2026_2027_with_biometrics.csv'
    enriched.to_csv(full_output, index=False)
    outputs = {'full': full_output}
    for name, view in views.items():
        target = DATA/f'players_position_{name}_min180_with_biometrics.csv'
        view.to_csv(target, index=False)
        outputs[name] = target
    audit.to_csv(REPORTS/'player_match_audit.csv', index=False)
    review.to_csv(REPORTS/'review_only_candidates.csv', index=False)
    if conflicts.empty:
        conflicts = pd.DataFrame(columns=['source_row', 'Player', 'field', 'profile_value', 'official_value', 'official_url', 'resolution'])
    conflicts.to_csv(REPORTS/'source_conflicts.csv', index=False)
    # Coverage is reported for the full source and the exact primary-experiment cohort.
    coverage_rows = []
    for name, mask in [('All source records', pd.Series(True, index=original.index)),
                       ('Outfield, at least 180 minutes', original.index.isin(eligible.source_row))]:
        subset = enriched.loc[mask]
        complete = subset[['height_cm', 'weight_kg']].notna().all(axis=1)
        coverage_rows.append({'Sample': name, 'Rows': len(subset), 'Height and weight': int(complete.sum()),
                              'Coverage': f'{complete.mean():.1%}', 'Missing either': int((~complete).sum())})
    coverage = pd.DataFrame(coverage_rows)
    coverage.to_csv(REPORTS/'coverage_summary.csv', index=False)
    cohort = enriched.loc[eligible.source_row].copy()
    cohort['Target'] = cohort.Pos.astype('string').str.split(',', n=1).str[0].str.strip()
    cohort['biometrics_complete'] = cohort[['height_cm', 'weight_kg']].notna().all(axis=1)
    by_position = cohort.groupby('Target').biometrics_complete.agg(['count', 'sum', 'mean']).reset_index()
    by_position.columns = ['Position', 'Records', 'Both available', 'Fraction available']
    by_position.to_csv(REPORTS/'coverage_by_position.csv', index=False)
    by_league = cohort.groupby('Comp').biometrics_complete.agg(['count', 'sum', 'mean']).reset_index()
    by_league.to_csv(REPORTS/'coverage_by_league.csv', index=False)
    young = pd.to_numeric(cohort.Age, errors='coerce').lt(21)
    age_coverage = {name: float(cohort.loc[mask, 'biometrics_complete'].mean())
                    for name, mask in [('Under 21', young), ('Age 21+', ~young & cohort.Age.notna())]}
    status_table = audit.biometrics_match_status.value_counts().rename_axis('Status').reset_index(name='Records')
    status_table.to_csv(REPORTS/'match_status_counts.csv', index=False)
    manifest = {
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'original_file': str(source), 'original_sha256': original_hash,
        'external_sources': source_metadata, 'official_overrides_file': str(override_file),
        'original_rows': len(original), 'matched_rows': int(accepted.sum()),
        'match_status_counts': status_table.set_index('Status').Records.to_dict(),
        'candidate_profiles': len(profiles), 'candidate_profiles_with_DOB': int(profiles.birth_year.notna().sum()),
        'fuzzy_matches_automatically_accepted': 0,
        'match_rule': 'Unique normalized name alias + exact birth year + nationality; exact club used only to resolve ties',
        'ratings_imported': False, 'imputation_performed': False,
        'source_conflicts_resolved_with_official_listings': len(conflicts),
        'coverage_by_age': age_coverage,
        'versions': {p: importlib.metadata.version(p) for p in ['pandas', 'numpy', 'pycountry', 'requests', 'kagglehub']},
        'outputs': {name: {'path': str(p), 'sha256': sha256(p)} for name, p in outputs.items()},
    }
    (REPORTS/'enrichment_manifest.json').write_text(json.dumps(manifest, indent=2))
    report = f'''# Adding listed height and weight to the football dataset

The enriched CSVs are new copies; the original {len(original):,}-row dataset and previous
notebooks are unchanged. `height_cm` and `weight_kg` are **published profile listings**, not
newly measured, clinical, or match-day biometrics. No game ability ratings were imported.

## What was added and how much is available?

{table(coverage)}

The full export is `data/{full_output.name}`. The identifier-free primary feature export is
`data/players_position_cleaned_min180_with_biometrics.csv`, with the original 20 columns
plus height and weight. A corresponding raw-count feature export is also included. Their
rows, targets, and previous feature values are unchanged. Matching back to the raw records
was validated against every original feature, rather than assumed from row order.

## Sources and provenance

1. **Measurements:** [EAFC26 Player Database, version 2]({PHYSICAL_URL}), published by
   Aiden Flynn on Kaggle, snapshot updated **2 April 2026**. The men's file has {len(profiles):,}
   unique EA player IDs and explicit cm/kg values. The publisher labels the dataset GPL 3.
   Its update date identifies the snapshot; the original measurement dates are unknown.
2. **Identity bridge:** [EA FC 27 profile dataset]({IDENTITY_README}), publisher snapshot
   dated **12 September 2026**, pinned to commit `{IDENTITY_COMMIT}`. Its height and weight
   columns are empty, so they are **not** a source of body measurements. Shared EA player
   IDs attach its names, birth dates, and nationalities to {int(profiles.birth_year.notna().sum()):,}
   profiles in the physical source. Only biographical fields are used from this bridge.
3. **Checked official override:** [Real Madrid's Kylian Mbappé profile](https://www.realmadrid.com/en-US/football/squad/kylian-mbappe)
   lists 178 cm and 75 kg. The bulk profile lists 182 cm and 75 kg, so the official club value
   is preferred. This {len(conflicts)}-field conflict is recorded in `source_conflicts.csv`;
   it demonstrates that profile sources can disagree. The remainder of the bulk source is
   **not individually verified against clubs**. The club page's measurement date is also unknown.

Each accepted row in the full CSV and `player_match_audit.csv` records the matched EA ID,
profile name, DOB, source URLs, snapshot date when known, bulk retrieval time, and quality flag.
Official corrections also record the date the club listing was checked.
The original bulk values are retained separately for auditing checked overrides. Source
downloads and checksums are in `data/biometrics_sources/source_manifest.json`.

## Conservative identity matching

- Normalize accents, punctuation, whitespace, and case. Match only an **exact name alias,
  exact birth year, and matching nationality** in at least one of the published profiles.
- Use the stable EA ID to join the two external datasets; reject nonunique source IDs.
  Full DOB parsing is now necessary for those external profiles. The performance CSV still
  has no match-date column, and no match dates are invented.
- Current club is not a primary key because transfers and source club dates differ.
  When several candidates share name/year/nationality, accept a match only if exactly one
  also has an exact normalized club name in the source snapshots. Otherwise leave it ambiguous.
- Two distinct players called **Vitinha** share Portuguese nationality and birth year 2000;
  the Genoa and Paris SG records therefore require the club tie-break. Name + birth year
  alone would be insufficient even in this small dataset.
- Never use position, performance statistics, age-based birth-year estimates, or fuzzy
  similarity as automatic identity evidence. `review_only_candidates.csv` provides tentative
  suggestions for independent checking; **none are merged automatically**.
- Preserve unmatched records, leaving body values missing. Do not substitute class means,
  infer height from position, or fabricate body measurements. Plausibility checks only detect
  parsing/unit errors; they do not establish accuracy of a published listing.

{table(status_table)}

## Coverage and modeling implications

{table(by_position.assign(**{'Fraction available': by_position['Fraction available'].map(lambda x: f'{x:.1%}')}))}

Body values are available for {age_coverage['Under 21']:.1%} of under-21 cohort records and
{age_coverage['Age 21+']:.1%} of records aged 21+. Missingness may select against newer or less
covered players; complete-case deletion could change the experiment's population. Some
nationalities differ between sources because of dual citizenship or national-team choices.
No match should be interpreted as a guarantee that all possible namesakes were ruled out.

For a model, add height and weight as ordinary numerical variables with training-fold
imputation and scaling. **Do not convert them to per-90 values.** Keep IDs, source URLs,
match status, dates, external positions, and game ratings out of the predictor set. Source
metadata belongs in the audit. Weight may change over time, and these listings cannot be
presented as exact values at the date of each performance observation.

The current playing-style notebook remains performance-only. A future comparison can test
performance-only versus body-only versus combined features, using the same records and
reporting coverage. Adding body variables changes what similarity means; it should be an
explicit experiment, not a silent change to the existing styles.

## Reproduce

From the enclosing FootballAIProject directory:

```sh
.venv/bin/python -m pip install -r AI_Football_Project/requirements-biometrics.txt
.venv/bin/python AI_Football_Project/enrich_player_biometrics.py
```

Sources are pinned for reproducibility and cached after the first download. The option
`--refresh-sources` re-fetches those pinned snapshots; it does not claim current-day
measurements or automatically adopt a new snapshot. Updating sources requires validating
their schema and provenance. `official_overrides.csv` is an extensible, sourced correction
table. The original raw and cleaned data are never overwritten.
'''
    (REPORTS/'README.md').write_text(report)
    assert sha256(source) == original_hash
    print(coverage.to_string(index=False))
    print('\n'+status_table.to_string(index=False))
    print('\nFiles:')
    for name, path in outputs.items():
        print(name, path)
    print('Report:', REPORTS/'README.md')


if __name__ == '__main__':
    main()
