"""Synthetic fixtures test validation rules; these are not football observations."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import import_whoscored_stats as importer
import enrich_player_biometrics as biometrics


def metadata():
    return {
        'source': 'WhoScored', 'season': '2026/2027', 'competition': 'eng Premier League',
        'scope': 'domestic_league_only', 'cutoff_date': '2026-11-01',
        'retrieved_at_utc': '2026-11-19T12:00:00Z',
        'source_urls': ['https://www.whoscored.com/synthetic-test-only'],
        'verification_notes': 'Synthetic fixture; not a source verification.',
        'metrics': {
            'key_passes': {'column': 'KeyP', 'units': 'total', 'definition': 'Synthetic count'},
            'pass_completion_pct': {'column': 'PS%', 'units': 'percentage', 'definition': 'Synthetic percentage'},
        },
    }


def export():
    return pd.DataFrame([{'source_player_id': 999999, 'Player': 'Synthetic Player',
                          'Squad': 'Synthetic Club', 'Comp': 'eng Premier League',
                          'MP': 2, 'Min': 180, 'KeyP': 6, 'PS%': 80}])


def base():
    return pd.DataFrame([{'Player': 'Synthetic Player', 'Squad': 'Synthetic Club',
                          'Comp': 'eng Premier League', 'MP': 2, 'Min': 180, 'Born': 2000}])


class ImportValidationTests(unittest.TestCase):
    def test_totals_and_percentages_have_distinct_units(self):
        source, features = importer.prepare_export(export(), metadata())
        self.assertEqual(source.ws_key_passes_p90.iloc[0], 3)
        self.assertEqual(source.ws_key_passes_total.iloc[0], 6)
        self.assertEqual(source.ws_pass_completion_pct.iloc[0], 80)
        self.assertNotIn('source_player_id', features)

    def test_per90_is_retained_without_reverse_inventing_totals(self):
        meta = metadata()
        meta['metrics']['key_passes']['units'] = 'per90'
        data = export(); data['KeyP'] = 2.34
        source, features = importer.prepare_export(data, meta)
        self.assertEqual(source.ws_key_passes_p90.iloc[0], 2.34)
        self.assertNotIn('ws_key_passes_total', features)

    def test_rejects_wrong_season_competition_scope_and_units(self):
        for field, value in [('season', '2025/2026'), ('competition', 'Champions League'),
                             ('scope', 'all_competitions'), ('cutoff_date', '2026-11-19')]:
            with self.subTest(field=field):
                meta = metadata(); meta[field] = value
                with self.assertRaises(ValueError): importer.validate_metadata(meta)
        meta = metadata(); meta['metrics']['key_passes']['units'] = 'per_game'
        with self.assertRaises(ValueError): importer.validate_metadata(meta)

    def test_requires_usable_source_provenance(self):
        for field, value in [('source_urls', []), ('source_urls', ['https://example.com/stats']),
                             ('verification_notes', ''), ('retrieved_at_utc', '2026-11-19T12:00:00')]:
            with self.subTest(field=field):
                meta = metadata(); meta[field] = value
                with self.assertRaises(ValueError): importer.validate_metadata(meta)

    def test_rejects_row_metadata_conflict(self):
        data = export(); data['Comp'] = 'es La Liga'
        with self.assertRaises(ValueError): importer.prepare_export(data, metadata())
        data = export(); data['Season'] = '2025/26'
        with self.assertRaises(ValueError): importer.prepare_export(data, metadata())

    def test_rejects_bad_numeric_input_and_rounded_totals(self):
        for value in ['unavailable', -1, np.inf, 2.5]:
            with self.subTest(value=value):
                data = export(); data['KeyP'] = value
                with self.assertRaises(ValueError): importer.prepare_export(data, metadata())
        data = export(); data['PS%'] = 101
        with self.assertRaises(ValueError): importer.prepare_export(data, metadata())

    def test_missing_is_preserved_and_zero_is_observed(self):
        data = export(); data['KeyP'] = np.nan
        source, _ = importer.prepare_export(data, metadata())
        self.assertTrue(pd.isna(source.ws_key_passes_p90.iloc[0]))
        data['KeyP'] = 0
        source, _ = importer.prepare_export(data, metadata())
        self.assertEqual(source.ws_key_passes_p90.iloc[0], 0)

    def test_accepts_exact_matches_and_preserves_base(self):
        before = base()
        source, features = importer.prepare_export(export(), metadata())
        result, audit = importer.merge_export(before, source, features)
        pd.testing.assert_frame_equal(before, result[before.columns])
        self.assertEqual(audit.status.iloc[0], 'matched_exact_identity_and_exposure')
        self.assertEqual(result.ws_key_passes_p90.iloc[0], 3)

    def test_rejects_different_exposure_and_birth_year(self):
        for col, value in [('Min', 179), ('MP', 3), ('Born', 2001)]:
            with self.subTest(col=col):
                data = export(); data[col] = value
                source, features = importer.prepare_export(data, metadata())
                result, audit = importer.merge_export(base(), source, features)
                expected = 'birthyear_mismatch' if col == 'Born' else 'exposure_mismatch'
                self.assertEqual(audit.status.iloc[0], expected)
                self.assertTrue(result[features].isna().all().all())

    def test_does_not_fuzzily_match_names(self):
        data = export(); data['Player'] = 'Synthetic Players'
        source, features = importer.prepare_export(data, metadata())
        result, audit = importer.merge_export(base(), source, features)
        self.assertEqual(audit.status.iloc[0], 'unmatched')
        self.assertTrue(result[features].isna().all().all())

    def test_ambiguous_namesakes_are_not_imported(self):
        data = pd.concat([export(), export()], ignore_index=True)
        data.loc[1, 'source_player_id'] = 999998
        source, features = importer.prepare_export(data, metadata())
        result, audit = importer.merge_export(base(), source, features)
        self.assertEqual(audit.status.iloc[0], 'ambiguous_identity')
        self.assertTrue(result[features].isna().all().all())

    def test_requires_explicit_club_alias(self):
        data = export(); data['Squad'] = 'Synthetic Club FC'
        meta = metadata(); meta['club_aliases'] = {'Synthetic Club FC': 'Synthetic Club'}
        source, features = importer.prepare_export(data, meta)
        result, audit = importer.merge_export(base(), source, features)
        self.assertEqual(audit.status.iloc[0], 'matched_exact_identity_and_exposure')

    def test_end_to_end_output_preserves_existing_features(self):
        # Isolated temp output; never writes synthetic observations into project data.
        with tempfile.TemporaryDirectory() as name:
            root = Path(name); data = root / 'data'; data.mkdir()
            raw = pd.concat([base(), base()], ignore_index=True)
            raw.loc[1, 'Comp'] = 'es La Liga'
            raw['Pos'] = 'MF'; raw['Age'] = 26
            for metric in biometrics.COUNTS:
                raw[metric] = 0
            raw.to_csv(data / 'players_data-2026_2027.csv', index=False)
            bio = raw.copy(); bio['height_cm'] = 180; bio['weight_kg'] = 75
            bio.to_csv(data / 'players_data-2026_2027_with_biometrics.csv', index=False)
            for kind in ['cleaned', 'raw_counts']:
                pd.DataFrame({'Age': [26, 26], 'Min': [180, 180],
                              'Comp': ['eng Premier League', 'es La Liga'],
                              'Target': ['MF', 'MF']}).to_csv(data / f'players_position_{kind}_min180.csv', index=False)
            csv = root / 'fixture.csv'; export().to_csv(csv, index=False)
            meta = root / 'fixture.json'; meta.write_text(json.dumps(metadata()))
            spanish = export(); spanish['Comp'] = 'es La Liga'; spanish['KeyP'] = 2.34
            spanish_csv = root / 'spanish_fixture.csv'; spanish.to_csv(spanish_csv, index=False)
            spanish_meta = metadata(); spanish_meta['competition'] = 'es La Liga'
            spanish_meta['metrics']['key_passes']['units'] = 'per90'
            spanish_json = root / 'spanish_fixture.json'; spanish_json.write_text(json.dumps(spanish_meta))
            with patch.object(biometrics, 'DATA', data), patch.multiple(importer, DATA=data, RAW=data/'players_data-2026_2027.csv',
                                REPORTS=root/'reports', FROZEN_SHA256=importer.digest(data/'players_data-2026_2027.csv')):
                importer.run([csv, spanish_csv], [meta, spanish_json])
            result = pd.read_csv(data/'players_position_cleaned_min180_with_biometrics_whoscored.csv')
            self.assertEqual(len(result), 2)
            self.assertEqual(result.height_cm.iloc[0], 180)
            self.assertEqual(result.ws_key_passes_p90.iloc[0], 3)
            self.assertEqual(result.ws_key_passes_p90.iloc[1], 2.34)
            self.assertEqual(result.Target.iloc[0], 'MF')
            self.assertNotIn('ws_key_passes_total', result)
            self.assertNotIn('ws_player_id', result)

    def test_changed_base_is_rejected_before_writing(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            raw = root/'raw.csv'; raw.write_text('synthetic changed snapshot')
            with patch.multiple(importer, RAW=raw, REPORTS=root/'reports', DATA=root/'data'):
                with self.assertRaisesRegex(ValueError, 'base snapshot changed'):
                    importer.run([], [])
            self.assertFalse((root/'reports').exists())
            self.assertFalse((root/'data').exists())


if __name__ == '__main__':
    unittest.main()
