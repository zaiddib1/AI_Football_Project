"""Synthetic tables exercise the collector without making live WhoScored requests."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from urllib.parse import parse_qs, unquote, urlparse

import pandas as pd
from playwright.sync_api import sync_playwright

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import scrape_whoscored_stats as scraper
from import_whoscored_stats import prepare_export, validate_metadata

FIXTURE = '''<!doctype html><html><title>Synthetic league fixture</title><body>
<select id="seasons"><option selected>2026/2027</option><option>2025/2026</option></select>
<button id="summary" class="active" onclick="category='Summary';number=1;render()">Summary</button>
<button id="passing" onclick="category='Passing';number=1;render()">Passing</button>
<button id="offensive" onclick="category='Offensive';number=1;render()">Offensive</button>
<button id="defensive" onclick="category='Defensive';number=1;render()">Defensive</button>
<button id="total" class="active">Total</button>
<div id="stats"><table><thead></thead><tbody></tbody></table>
<button id="first" onclick="number=1;render()">First</button>
<button id="previous" onclick="number=1;render()">Previous</button>
<button id="next" onclick="number=2;render()">Next</button></div>
<script>
let number=1, category='Summary';
function render(){
document.querySelector('#summary').className=category==='Summary'?'active':'';
document.querySelector('#passing').className=category==='Passing'?'active':'';
document.querySelector('#offensive').className=category==='Offensive'?'active':'';
document.querySelector('#defensive').className=category==='Defensive'?'active':'';
const metric={Summary:'AerialsWon',Passing:'KeyP',Offensive:'Drb',Defensive:'Clr'}[category];
document.querySelector('thead').innerHTML='<tr><th>Player</th><th>Apps</th><th>Mins</th><th>'+metric+'</th><th>PS%</th></tr>';
document.querySelector('tbody').innerHTML='<tr><td><a href="/players/'+(900000+number)+'/show">Synthetic '+number+'</a><a href="/teams/990001/show">Synthetic Club</a></td><td>2(1)</td><td>180</td><td>'+ (category==='Summary'?'4':'6')+'</td><td>80</td></tr>';
document.querySelector('#next').disabled=number===2;
document.querySelector('#previous').disabled=number===1;
document.querySelector('#first').disabled=number===1;
}
render();</script></body></html>'''

DETAILED_URL = 'https://www.whoscored.com/regions/252/tournaments/2/seasons/11141/stages/25544/playerstatistics/synthetic'
DETAILED_FIXTURE = '''<!doctype html><html><title>Synthetic Detailed fixture</title><body>
<select id="seasons"><option selected>2026/2027</option></select>
<a href="#stage-top-player-stats-detailed">Detailed</a>
<select id="category"><option value="key-passes">Key passes</option><option value="passes">Passes</option>
<option value="dribbles">Dribbles</option><option value="aerial">Aerial</option>
<option value="clearances">Clearances</option><option value="blocks">Blocks</option>
<option value="possession-loss">Possession loss</option></select>
<select id="subcategory"><option value="length">Length</option></select>
<select id="statsAccumulationType"><option value="0">Per game</option><option value="1">Per 90 mins</option>
<option value="2">Total</option></select></body></html>'''


def fixture_feed(url):
    query = parse_qs(urlparse(url).query)
    category = query['category'][0]
    fields = {
        'key-passes': {'keyPassesTotal': 6},
        'passes': {'passTotal': 10, 'passLongBallAccurate': 2, 'passLongBallInaccurate': 1,
                   'shortPassAccurate': 6, 'shortPassInaccurate': 1},
        'dribbles': {'dribbleWon': 4}, 'aerial': {'duelAerialWon': 5},
        'clearances': {'clearanceTotal': 7}, 'blocks': {'outfielderBlock': 2},
        'possession-loss': {'turnover': 3, 'dispossessed': 1},
    }[category]
    if query['statsAccumulationType'][0] == '1':
        fields = {key: value / 2 for key, value in fields.items()}
    records = [{'playerId': 900001+i, 'name': f'Synthetic {i+1}', 'teamId': 990001,
                'teamName': 'Synthetic Club', 'seasonId': 11141, 'seasonName': '2026/2027',
                'tournamentId': 2, 'apps': 3, 'subOn': 1, 'minsPlayed': 180, **fields} for i in range(2)]
    page = int(query.get('page', ['1'])[0]); limit = int(query.get('numberOfPlayersToPick', ['500'])[0])
    selected = records[(page-1)*limit:page*limit]
    return {'playerTableStats': selected, 'statColumns': ['apps','subOn','minsPlayed',*fields],
            'paging': {'currentPage': page, 'totalResults': 2, 'totalPages': (2+limit-1)//limit, 'resultsPerPage': limit}}


def detailed_route(route):
    if 'getplayerstatistics' in route.request.url:
        route.fulfill(content_type='application/json', body=json.dumps(fixture_feed(route.request.url)))
    else:
        route.fulfill(content_type='text/html', body=DETAILED_FIXTURE)


def capture():
    return {
        'source': 'WhoScored', 'status': 'complete', 'observed_season': '2026/2027',
        'competition': 'eng Premier League',
        'source_urls': ['https://www.whoscored.com/regions/252/tournaments/2'],
        'finished_at_utc': '2026-11-21T12:00:00Z', 'reporting_cutoff_date': None,
        'pages': [{'category': 'Summary', 'page_number': 1, 'season': '2026/2027',
                   'competition': 'eng Premier League', 'units': 'total', 'units_labels': ['Total'],
                   'headers': [{'label': label, 'title': ''} for label in ['Player', 'Apps', 'Mins', 'KeyP', 'PS%']],
                   'rows': [{'player_id': '900001', 'player_name': 'Synthetic One',
                             'team_id': '990001', 'team_name': 'Synthetic Club',
                             'cells': ['Synthetic One', '2(1)', '180', '6', '80%']}]}],
    }


class NormalizationTests(unittest.TestCase):
    def test_feed_units_appearances_and_derived_accuracy(self):
        pages = []
        for category in ['key-passes','passes']:
            url = f'https://www.whoscored.com/statisticsfeed/1/getplayerstatistics?category={category}&statsAccumulationType=2&tournamentOptions=2&page=1'
            body = fixture_feed(url)
            pages.append({'category': category, 'page_number': 1, 'season': '2026/2027',
                          'competition': 'eng Premier League', 'units': 'total', 'units_labels': ['Total'],
                          'request_url': url, 'paging': body['paging'], 'stat_columns': body['statColumns'],
                          'records': body['playerTableStats']})
        data = {**capture(), 'format_version': 2, 'unit_option_mapping': [{'label': 'Total', 'value': '2'}], 'pages': pages}
        frame, meta = scraper.normalize_capture(data)
        self.assertEqual(frame.MP.tolist(), [3, 3])  # subOn must not be added to total apps.
        self.assertEqual(frame.pass_completion_pct.tolist(), [80, 80])
        self.assertEqual(frame.accurate_passes.tolist(), [8, 8])
        self.assertEqual(frame.long_balls.tolist(), [3, 3])
        data['pages'][0]['request_url'] = data['pages'][0]['request_url'].replace('statsAccumulationType=2','statsAccumulationType=0')
        with self.assertRaisesRegex(ValueError, 'unit parameter'): scraper.normalize_capture(data)

    def test_total_rate_and_unknown_cutoff_are_kept_distinct(self):
        frame, meta = scraper.normalize_capture(capture())
        self.assertEqual(frame.MP.iloc[0], 3)
        self.assertEqual(frame.key_passes.iloc[0], 6)
        self.assertEqual(frame.pass_completion_pct.iloc[0], 80)
        self.assertIsNone(meta['cutoff_date'])
        self.assertFalse(meta['import_ready'])
        with self.assertRaisesRegex(ValueError, 'reporting cutoff is unknown'):
            validate_metadata(meta)
        # An independently checked cutoff lets the existing importer compute rates.
        meta['cutoff_date'] = '2026-11-01'
        ready, _ = prepare_export(frame, meta)
        self.assertEqual(ready.ws_key_passes_p90.iloc[0], 3)

    def test_per_game_and_average_headers_are_excluded(self):
        self.assertIsNone(scraper.metric_for({'label': 'AvgP', 'title': ''}, 'total'))
        self.assertIsNone(scraper.metric_for({'label': 'KeyP', 'title': 'Key passes per game'}, 'total'))
        self.assertIsNone(scraper.metric_for({'label': 'Dribbles', 'title': 'Dribbles attempted'}, 'total'))

    def test_partial_wrong_season_and_misdeclared_units_are_rejected(self):
        for change in ['partial', 'season', 'labels']:
            with self.subTest(change=change):
                data = capture()
                if change == 'partial': data['status'] = 'incomplete'
                if change == 'season': data['observed_season'] = '2025/2026'
                if change == 'labels': data['pages'][0]['units_labels'] = ['Per game']
                with self.assertRaises(ValueError): scraper.normalize_capture(data)

    def test_duplicate_rows_and_category_conflicts_are_rejected(self):
        data = capture(); other = deepcopy(data['pages'][0]); other['page_number'] = 2
        data['pages'].append(other)
        with self.assertRaisesRegex(ValueError, 'repeats within a category'): scraper.normalize_capture(data)
        data['pages'][1]['category'] = 'Passing'; data['pages'][1]['rows'][0]['cells'][3] = '7'
        with self.assertRaisesRegex(ValueError, 'values disagree'): scraper.normalize_capture(data)

    def test_missing_values_and_valid_per90_values(self):
        data = capture(); data['pages'][0]['rows'][0]['cells'][3] = '—'
        frame, _ = scraper.normalize_capture(data)
        self.assertTrue(pd.isna(frame.key_passes.iloc[0]))
        data['pages'][0].update(units='per90', units_labels=['Per 90'])
        data['pages'][0]['rows'][0]['cells'][3] = '1.23'
        frame, meta = scraper.normalize_capture(data)
        self.assertEqual(frame.key_passes.iloc[0], 1.23)
        self.assertEqual(meta['metrics']['key_passes']['units'], 'per90')

    def test_blocked_capture_only_writes_evidence(self):
        with tempfile.TemporaryDirectory() as name:
            out = Path(name) / 'blocked'
            with self.assertRaises(ValueError):
                scraper.write_capture({'source': 'WhoScored', 'status': 'blocked', 'error': 'Synthetic block'}, out)
            self.assertEqual([p.name for p in out.iterdir()], ['raw_capture.json'])


class BrowserCollectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.playwright = sync_playwright().start()
        cls.browser = cls.playwright.chromium.launch(channel='chrome', headless=True)
        cls.script = '(async (options) => {\n' + scraper.COLLECTOR.read_text() + '\nreturn await collectWhoScored(options);\n})'
        cls.detailed_script = '(async (options) => {\n' + scraper.collector_source() + '\nreturn await collectWhoScoredDetailed(options);\n})'

    @classmethod
    def tearDownClass(cls):
        cls.browser.close(); cls.playwright.stop()

    def evaluate(self, html=FIXTURE, **options):
        page = self.browser.new_page()
        # Every request is fulfilled locally. These are not live-site validation results.
        page.route('**/*', lambda route: route.fulfill(content_type='text/html', body=html))
        page.goto('https://www.whoscored.com/regions/252/tournaments/2')
        try:
            return page.evaluate(self.script, {'league': 'epl', 'categories': ['Summary', 'Passing'],
                'units': 'total', 'delay_ms': 1, 'change_timeout_ms': 800, **options})
        finally:
            page.close()

    def test_pagination_and_category_join(self):
        result = self.evaluate()
        self.assertEqual(result['status'], 'complete', result.get('error'))
        self.assertEqual(len(result['pages']), 4)
        frame, meta = scraper.normalize_capture(result)
        self.assertEqual(len(frame), 2)
        self.assertEqual(frame.key_passes.tolist(), [6, 6])
        self.assertEqual(frame.aerial_duels_won.tolist(), [4, 4])
        self.assertIsNone(meta['cutoff_date'])

    def test_wrong_season_does_not_export_rows(self):
        result = self.evaluate(FIXTURE.replace('<option selected>2026/2027</option>', '<option selected>2025/2026</option>'))
        self.assertEqual(result['status'], 'incomplete')
        self.assertEqual(result['pages'], [])

    def test_block_page_is_reported(self):
        result = self.evaluate('<html><title>Attention Required!</title><body>Sorry, you have been blocked</body></html>')
        self.assertEqual(result['status'], 'blocked')
        self.assertEqual(result['pages'], [])

    def test_page_cap_is_not_a_complete_export(self):
        result = self.evaluate(max_pages=1)
        self.assertEqual(result['status'], 'incomplete')
        self.assertIn('Page limit', result['error'])

    def test_starting_on_page_two_rewinds_before_capture(self):
        html = FIXTURE.replace('render();</script>', 'number=2;render();</script>')
        result = self.evaluate(html)
        self.assertEqual(result['status'], 'complete', result.get('error'))
        self.assertEqual(result['pages'][0]['rows'][0]['player_id'], '900001')

    def test_generated_bookmarklet_downloads_a_complete_capture(self):
        scraper.build_bookmarklet()
        page = self.browser.new_page(accept_downloads=True)
        try:
            html = scraper.ROOT / 'whoscored_enrichment/WhoScored_exporter.html'
            page.goto(html.as_uri())
            href = page.get_by_role('link', name='Export WhoScored player tables').get_attribute('href')
            self.assertTrue(href.startswith('javascript:'))
            code = unquote(href[len('javascript:'):])
            page.route('https://www.whoscored.com/**', detailed_route)
            page.goto(DETAILED_URL)
            with page.expect_download(timeout=40000) as event:
                page.evaluate(code)
            downloaded = json.loads(Path(event.value.path()).read_text())
            self.assertEqual(downloaded['status'], 'complete', downloaded.get('error'))
            self.assertEqual(len(downloaded['pages']), 7)
            frame, meta = scraper.normalize_capture(downloaded)
            self.assertEqual(len(frame), 2)
            self.assertIn('successful_dribbles', meta['metrics'])
        finally:
            page.close()

    def test_detailed_feed_pagination(self):
        page = self.browser.new_page()
        try:
            page.route('**/*', detailed_route); page.goto(DETAILED_URL)
            result = page.evaluate(self.detailed_script, {'league': 'epl', 'categories': ['Key passes','Passes'],
                                                         'page_size': 1, 'delay_ms': 1})
            self.assertEqual(result['status'], 'complete', result.get('error'))
            self.assertEqual(len(result['pages']), 4)
            frame, _ = scraper.normalize_capture(result)
            self.assertEqual(len(frame), 2)
            self.assertEqual(frame.pass_completion_pct.iloc[0], 80)
        finally:
            page.close()


if __name__ == '__main__':
    unittest.main()
