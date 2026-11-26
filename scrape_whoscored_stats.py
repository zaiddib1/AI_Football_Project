"""Collect rendered WhoScored player tables with explicit season and units evidence.

Uses ordinary Chrome in a separate profile, or converts a capture from the browser
exporter. Source blocks are reported without an automatic bypass or fabricated CSV.
"""
from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import parse_qs, urlparse

import pandas as pd

from import_whoscored_stats import COMPETITIONS, COUNT_METRICS, numeric_column

ROOT = Path(__file__).resolve().parent
COLLECTOR = ROOT / 'whoscored_enrichment/browser_collector.js'
DETAILED_COLLECTOR = ROOT / 'whoscored_enrichment/detailed_collector.js'
DEFAULT_OUTPUT = ROOT / 'data/whoscored_sources'
LEAGUE_ROUTES = {
    'epl': ('252', '2'), 'laliga': ('206', '4'), 'bundesliga': ('81', '3'),
    'seriea': ('108', '5'), 'ligue1': ('74', '22'),
}
HEADER_METRICS = {
    'keyp': 'key_passes', 'keypasses': 'key_passes', 'keypass': 'key_passes',
    'passes': 'passes', 'totalpasses': 'passes', 'accuratepasses': 'accurate_passes',
    'ps': 'pass_completion_pct', 'passsuccess': 'pass_completion_pct',
    'passsuccessrate': 'pass_completion_pct', 'passaccuracy': 'pass_completion_pct',
    'passcompletion': 'pass_completion_pct',
    'drb': 'successful_dribbles', 'dribbles': 'successful_dribbles',
    'successfuldribbles': 'successful_dribbles',
    'aerialswon': 'aerial_duels_won', 'aerialduelswon': 'aerial_duels_won',
    'clr': 'clearances', 'clearances': 'clearances', 'blocks': 'blocks',
    'disp': 'dispossessions', 'dispossessed': 'dispossessions',
    'unschtch': 'poor_controls', 'badcontrol': 'poor_controls', 'poorcontrols': 'poor_controls',
    'longb': 'long_balls', 'longballs': 'long_balls', 'thrb': 'through_balls',
    'throughballs': 'through_balls', 'xg': 'xg', 'expectedgoals': 'xg',
}
FEED_FIELDS = {
    'keyPassesTotal': 'key_passes', 'passTotal': 'passes',
    'dribbleWon': 'successful_dribbles', 'duelAerialWon': 'aerial_duels_won',
    'clearanceTotal': 'clearances', 'outfielderBlock': 'blocks',
    'dispossessed': 'dispossessions', 'turnover': 'poor_controls',
}


def collector_source():
    return COLLECTOR.read_text() + '\n' + DETAILED_COLLECTOR.read_text()


def feed_as_tables(capture):
    """Convert source JSON values without reconstructing totals from rounded averages."""
    converted = dict(capture); converted['pages'] = []
    counts = {}
    for page in capture['pages']:
        info = counts.setdefault(page['category'], {'count': 0, 'pages': [], 'total': page['paging']['totalResults']})
        if info['total'] != page['paging']['totalResults']:
            raise ValueError('Feed totalResults changed between pages.')
        info['count'] += len(page['records']); info['pages'].append(page['page_number'])
        query = parse_qs(urlparse(page['request_url']).query)
        value = query.get('statsAccumulationType', [None])[0]
        expected_label = 'Total' if page['units'] == 'total' else 'Per 90 mins'
        mapping = capture.get('unit_option_mapping', [])
        if not any(option['value'] == value and option['label'] == expected_label for option in mapping):
            raise ValueError('Request unit parameter does not match the observed source option mapping.')
        fields = [(field, metric) for field, metric in FEED_FIELDS.items() if field in page['stat_columns']]
        pass_parts = ['passLongBallAccurate','passLongBallInaccurate','shortPassAccurate','shortPassInaccurate']
        derive_passes = all(field in page['stat_columns'] for field in ['passTotal', *pass_parts])
        headers = [{'label': label, 'title': ''} for label in ['Player','Apps','Mins']]
        headers += [{'label': metric, 'title': f'Source API field {field}'} for field, metric in fields]
        if derive_passes:
            headers += [{'label': name, 'title': definition} for name, definition in [
                ('accurate_passes','Source shortPassAccurate + passLongBallAccurate'),
                ('pass_completion_pct','100 * accurate passes / passTotal, after validating the four pass components sum to passTotal'),
                ('long_balls','Source passLongBallAccurate + passLongBallInaccurate')]]
        rows = []
        for record in page['records']:
            if record['seasonName'] != '2026/2027' or str(record['tournamentId']) != query.get('tournamentOptions', [None])[0]:
                raise ValueError('Feed row has an unexpected season or tournament.')
            cells = [record['name'], str(record['apps']), str(record['minsPlayed'])]
            cells += [record.get(field) for field, _ in fields]
            if derive_passes:
                parts = [record.get(field) for field in pass_parts]
                total = record.get('passTotal')
                if total is None or any(value is None for value in parts):
                    cells += [None, None, None]
                else:
                    if abs(sum(parts) - total) > 1e-7:
                        raise ValueError('Pass components do not partition passTotal; cannot derive accuracy.')
                    accurate = parts[0] + parts[2]
                    cells += [accurate, 100 * accurate / total if total else None, parts[0] + parts[1]]
            rows.append({'player_id': str(record['playerId']), 'player_name': record['name'],
                         'team_id': str(record['teamId']), 'team_name': record['teamName'], 'cells': cells})
        converted['pages'].append({**page, 'headers': headers, 'rows': rows})
    for info in counts.values():
        if info['count'] != info['total'] or info['pages'] != list(range(1, len(info['pages']) + 1)):
            raise ValueError('Feed capture does not contain all declared records/pages.')
    return converted


def key(value):
    return re.sub(r'[^a-z0-9]', '', str(value).lower())


def appearances(value):
    """WhoScored's starts(sub appearances) display, or an integer total."""
    text = re.sub(r'\s+', '', str(value))
    match = re.fullmatch(r'(\d+)(?:\((\d+)\))?', text)
    if not match:
        raise ValueError(f'Unrecognized appearances value: {value!r}')
    return int(match[1]) + int(match[2] or 0)


def metric_for(header, inherited_units):
    label, title = header.get('label', ''), header.get('title', '')
    description = f'{label} {title}'.strip()
    # Average passes / shots per game must not inherit a Total tab's units.
    if re.search(r'per\s*(?:game|match)|\baverage\b', description, re.I) or key(label) in {'avgp', 'spg'}:
        return None
    label_key = re.sub(r'(per90(?:minutes|mins|min)?|90)$', '', key(label))
    metric = HEADER_METRICS.get(label_key) or next((m for m in COUNT_METRICS | {'pass_completion_pct'} if key(m) == label_key), None)
    if not metric:
        return None
    if metric == 'successful_dribbles' and re.search(r'attempt|unsuccessful|failed', description, re.I):
        return None
    if metric == 'pass_completion_pct':
        units = 'percentage'
    elif re.search(r'per\s*90|/\s*90', description, re.I):
        units = 'per90'
    else:
        units = inherited_units
    return metric, {'column': metric, 'units': units,
                    'definition': f'WhoScored source heading: {label}; tooltip: {title or "not supplied"}.',
                    'source_heading': label, 'source_tooltip': title}


def normalize_capture(capture):
    if capture.get('status') != 'complete':
        raise ValueError(f'Capture is {capture.get("status", "unknown")}: {capture.get("error", "not complete")}')
    if capture.get('source') != 'WhoScored' or capture.get('observed_season') != '2026/2027':
        raise ValueError('Only a captured WhoScored 2026/2027 season is supported.')
    if capture.get('format_version') == 2:
        capture = feed_as_tables(capture)
    competition = capture.get('competition')
    if competition not in COMPETITIONS or not capture.get('pages'):
        raise ValueError('Missing supported competition or player pages.')
    for url in capture.get('source_urls', []):
        if urlparse(url).hostname not in {'whoscored.com', 'www.whoscored.com'} or not url.startswith('https://'):
            raise ValueError('Unexpected source URL.')
    if not capture.get('source_urls'):
        raise ValueError('Capture has no source URLs.')
    rows, metrics, page_keys, player_pages = {}, {}, set(), set()
    for page in capture['pages']:
        if page.get('season') != '2026/2027' or page.get('competition') != competition:
            raise ValueError('Season/competition changed between captured pages.')
        unit = page.get('units')
        if unit not in {'total', 'per90'} or not page.get('units_labels'):
            raise ValueError('A visible Total or Per 90 control is required for each page.')
        expected_label = lambda label: key(label) in ({'total', 'totals', 'overalltotal', 'overalltotals'}
            if unit == 'total' else {'per90', 'per90min', 'per90mins', 'per90minutes', '90min', '90mins'})
        if not all(expected_label(label) for label in page['units_labels']):
            raise ValueError('Captured unit-control labels disagree with the declared units.')
        page_key = (page['category'], page['page_number'])
        if page_key in page_keys:
            raise ValueError('Repeated category/page number in the capture.')
        page_keys.add(page_key)
        headers = page['headers']
        for item in page['rows']:
            if len(item['cells']) != len(headers):
                raise ValueError('Captured headings and cells are not aligned.')
            player_id = str(item.get('player_id', ''))
            if not player_id.isdigit() or not item.get('player_name'):
                raise ValueError('A stable player ID and player name are required.')
            team = str(item.get('team_name') or '').strip()
            values = {key(header['label']): value for header, value in zip(headers, item['cells'])}
            team = team or values.get('team', '') or values.get('club', '')
            if not team:
                raise ValueError('A player row has no club identity; cannot safely join categories.')
            row_key = (player_id, str(item.get('team_id') or key(team)))
            unique_row = (page['category'], *row_key)
            if unique_row in player_pages:
                raise ValueError('A player/club row repeats within a category.')
            player_pages.add(unique_row)
            record = rows.setdefault(row_key, {'source_player_id': player_id, 'Player': item['player_name'],
                                               'Squad': team, 'Comp': competition, 'Season': '2026/2027'})
            if record['Player'] != item['player_name'] or record['Squad'] != team:
                raise ValueError('Player/club labels disagree between categories for the same IDs.')
            for label, col in [('apps', 'MP'), ('appearances', 'MP'), ('mins', 'Min'), ('minutes', 'Min')]:
                if label not in values:
                    continue
                value = appearances(values[label]) if col == 'MP' else int(str(values[label]).replace(',', '').strip())
                if col in record and record[col] != value:
                    raise ValueError('Appearances/minutes changed between categories.')
                record[col] = value
            for header, value in zip(headers, item['cells']):
                match = metric_for(header, unit)
                if match is None:
                    continue
                metric, specification = match
                if metric in metrics and metrics[metric]['units'] != specification['units']:
                    raise ValueError(f'{metric}: mixed units between tables.')
                metrics.setdefault(metric, specification)
                text = '' if value is None else str(value).strip()
                if specification['units'] == 'percentage':
                    text = text.removesuffix('%').strip()
                parsed = numeric_column(pd.Series([text]), metric,
                    integer=metric in COUNT_METRICS and specification['units'] == 'total').iloc[0]
                if metric == 'pass_completion_pct' and pd.notna(parsed) and parsed > 100:
                    raise ValueError('Pass completion percentage is outside 0–100.')
                if metric in record and pd.notna(record[metric]) and pd.notna(parsed) and record[metric] != parsed:
                    raise ValueError(f'{metric}: values disagree between captured categories.')
                if metric not in record or pd.notna(parsed):
                    record[metric] = parsed
    if not metrics or not rows:
        raise ValueError('No supported new statistics were captured.')
    for record in rows.values():
        if not {'MP', 'Min'} <= record.keys() or record['MP'] <= 0 or record['Min'] <= 0:
            raise ValueError('Missing positive appearances/minutes; capture the Summary category too.')
        if record['Min'] > 90 * record['MP']:
            raise ValueError('Minutes exceed 90 per appearance; verify the source exposure convention.')
    frame = pd.DataFrame(rows.values())
    metadata = {
        'source': 'WhoScored', 'season': '2026/2027', 'competition': competition,
        'scope': 'domestic_league_only', 'cutoff_date': capture.get('reporting_cutoff_date'),
        'retrieved_at_utc': capture['finished_at_utc'], 'source_urls': capture['source_urls'],
        'verification_notes': 'Visible season selector and tournament URL checked. Unit options matched to feed request parameters '
                              'for Detailed captures, or active unit controls recorded for DOM captures. '
                              'Categories joined by source player/club IDs. Exact reporting cutoff still requires verification.',
        'metrics': metrics, 'appearances_convention': 'API apps is total appearances; subOn is not added.'
            if capture.get('format_version') == 2 else 'Integer total or starts(substitute appearances), summed.',
        'source_filters': capture.get('filters') or [page.get('selected_controls', []) for page in capture['pages']],
        'pagination_evidence': capture.get('pagination_evidence'),
        'import_ready': False, 'cutoff_verification': 'not verified by scraper',
        'note': 'Collection time is not a reporting cutoff. This export has not been merged into the experiment.',
    }
    return frame, metadata


def write_capture(capture, directory):
    directory.mkdir(parents=True, exist_ok=False)
    raw = directory / 'raw_capture.json'
    raw.write_text(json.dumps(capture, indent=2, ensure_ascii=False))
    print(f'Capture status: {capture.get("status")}; evidence: {raw}', flush=True)
    frame, metadata = normalize_capture(capture)
    metadata['raw_capture_sha256'] = hashlib.sha256(raw.read_bytes()).hexdigest()
    frame.to_csv(directory / 'players.csv', index=False)
    rates = frame[['source_player_id','Player','Squad','Comp','Season','MP','Min']].copy()
    for metric, spec in metadata['metrics'].items():
        if spec['units'] == 'percentage':
            rates[metric] = frame[metric]
        else:
            rates[f'{metric}_p90'] = 90 * frame[metric] / frame.Min if spec['units'] == 'total' else frame[metric]
    rates.to_csv(directory / 'players_per90.csv', index=False)
    (directory / 'metadata.json').write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
    print(f'Collected {len(frame)} player/club records and {len(metadata["metrics"])} supported metrics.', flush=True)
    print(f'CSV: {directory / "players.csv"}', flush=True)
    print('Not imported: verify the source reporting cutoff against the frozen experiment first.', flush=True)


async def browser_capture(args):
    from playwright.async_api import async_playwright
    region, tournament = LEAGUE_ROUTES[args.league]
    url = args.url or f'https://www.whoscored.com/regions/{region}/tournaments/{tournament}'
    if urlparse(url).hostname not in {'www.whoscored.com', 'whoscored.com'} or not url.startswith('https://'):
        raise ValueError('--url must be an HTTPS WhoScored page.')
    async with async_playwright() as playwright:
        profile = Path.home() / '.cache/footballai/whoscored-browser'
        context = await playwright.chromium.launch_persistent_context(
            str(profile), channel='chrome', headless=args.headless)
        page = context.pages[0] if context.pages else await context.new_page()
        await page.expose_function('footballCollectorProgress', lambda message: print(message, flush=True))
        try:
            response = await page.goto(url, wait_until='domcontentloaded', timeout=30000)
            if response and response.status in {401, 403, 429}:
                return {'source': 'WhoScored', 'status': 'blocked', 'http_status': response.status,
                        'source_urls': [page.url], 'page_title': await page.title(),
                        'error': f'WhoScored returned HTTP {response.status}; normal browser access was not granted.',
                        'finished_at_utc': datetime.now(timezone.utc).isoformat()}
            # These season selectors are documented by soccerdata's WhoScored reader.
            season = page.locator('select[id*="seasons"]').filter(visible=True)
            if await season.count() == 1:
                choices = await season.locator('option').evaluate_all(
                    'nodes => nodes.map(node => ({label:node.textContent.trim(),value:node.value}))')
                wanted = next((choice for choice in choices if re.sub(r'\s', '', choice['label']).replace('-', '/') in {'2026/2027', '2026/27'}), None)
                if wanted is None:
                    raise ValueError('2026/2027 is not offered by the visible season selector.')
                if await season.input_value() != wanted['value']:
                    await season.select_option(value=wanted['value'])
                    await page.wait_for_timeout(3500)
            links = page.get_by_role('link', name=re.compile(r'^Player\s+(?:Statistics|Stats)$', re.I))
            if await links.count() == 1 and await links.is_visible():
                await links.click()
                await page.wait_for_timeout(3500)
            if args.interactive:
                print('In Chrome, select the player table, season, filters and units. Press Enter here when ready.', flush=True)
                await asyncio.to_thread(input)
            function = 'collectWhoScoredDetailed' if args.mode == 'detailed' else 'collectWhoScored'
            script = '(async (options) => {\n' + collector_source() + f'\nreturn await {function}(options);\n}})'
            categories = args.categories.split(',') if args.categories else None
            return await page.evaluate(script, {'league': args.league, 'season': '2026/2027',
                'units': args.units, **({'categories': categories} if categories else {}), 'max_pages': args.max_pages,
                'subcategories': getattr(args, 'subcategories', {})})
        finally:
            await context.close()


def build_bookmarklet():
    script = collector_source() + '''
void (async () => {
  if (window.footballWhoScoredCollectorRunning) { alert('A WhoScored export is already running.'); return; }
  window.footballWhoScoredCollectorRunning = true;
  const banner = document.createElement('div');
  banner.style.cssText = 'position:fixed;bottom:12px;left:12px;z-index:2147483647;background:#18232f;color:white;padding:14px;font:14px sans-serif;max-width:550px;box-shadow:0 2px 12px #0006';
  banner.textContent = 'WhoScored exporter is starting…'; document.body.append(banner);
  window.footballCollectorProgress = message => { banner.textContent = message; };
  try {
    const capture = await collectWhoScoredDetailed({units:'total'});
    const file = new Blob([JSON.stringify(capture,null,2)],{type:'application/json'});
    const url = URL.createObjectURL(file), link = document.createElement('a');
    link.href = url; link.download = 'whoscored_capture_' + new Date().toISOString().replace(/[:.]/g,'-') + '.json';
    link.click(); setTimeout(() => URL.revokeObjectURL(url),10000);
    banner.textContent = capture.status === 'complete' ? 'Capture downloaded. Convert it with scrape_whoscored_stats.py --from-capture. Reporting cutoff still needs verification.' : 'Capture stopped: ' + capture.error;
  } catch(error) { banner.textContent = 'Export failed: ' + error.message; }
  finally { window.footballWhoScoredCollectorRunning = false; }
})();
'''
    from urllib.parse import quote
    href = 'javascript:' + quote('(()=>{' + script + '})()', safe='')
    html = '''<!doctype html><html lang="en"><meta charset="utf-8"><title>WhoScored table exporter</title>
<style>body{font:17px/1.55 system-ui;max-width:760px;margin:60px auto;padding:0 24px;color:#18232f}a{color:#075c92}.bookmark{display:inline-block;padding:12px 20px;background:#075c92;color:white;border-radius:7px;text-decoration:none}code{background:#eef2f5;padding:3px 6px}</style>
<h1>WhoScored table exporter</h1>
<p>Drag this button to your browser's bookmarks bar:</p>
<p><a class="bookmark" href="''' + href + '''">Export WhoScored player tables</a></p>
<ol><li>Open an accessible WhoScored domestic-league player-statistics page.</li>
<li>Select <strong>2026/2027</strong> and open the league's Player Statistics page.</li>
<li>Click the saved bookmark. It uses the Detailed view's Total option and downloads the source data with verified pagination.</li>
<li>Convert the downloaded capture using <code>scrape_whoscored_stats.py --from-capture /path/to/capture.json</code>.</li></ol>
<p>The exporter reads the current page and clicks its table controls. It sends no data to another service.
It stops on blocked access, unknown units, ambiguous tables or incomplete pagination. The exact reporting
cutoff must be verified before values can be attached to your November 2 dataset.</p>
<p>If the site blocks the bookmark or the layout has changed, retain its error and do not treat a partial capture as a complete export.</p></html>'''
    path = ROOT / 'whoscored_enrichment/WhoScored_exporter.html'
    path.write_text(html)
    print(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--league', choices=[*LEAGUE_ROUTES, 'all'], default='epl')
    parser.add_argument('--url', help='Optional specific WhoScored league player-statistics page')
    parser.add_argument('--units', choices=['total', 'per90'], default='total')
    parser.add_argument('--mode', choices=['detailed','dom'], default='detailed', help='Detailed source feed (default) or rendered-table fallback')
    parser.add_argument('--categories', help='Comma-separated source category labels; default collects seven Detailed categories')
    parser.add_argument('--max-pages', type=int, default=100)
    parser.add_argument('--headless', action='store_true', help='Run unattended; default opens ordinary Chrome')
    parser.add_argument('--interactive', action='store_true', help='Pause for manual page/filter selection')
    parser.add_argument('--from-capture', type=Path, help='Convert JSON downloaded by the browser exporter')
    parser.add_argument('--build-bookmarklet', action='store_true')
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.headless and args.interactive:
        parser.error('--interactive requires a visible browser.')
    if not 1 <= args.max_pages <= 200:
        parser.error('--max-pages must be between 1 and 200.')
    if args.league == 'all' and args.url:
        parser.error('Use --url with a single league; all-league runs use each league route.')
    try:
        if args.build_bookmarklet:
            build_bookmarklet(); return
        directories = []
        for league in (list(LEAGUE_ROUTES) if args.league == 'all' and not args.from_capture else [args.league]):
            run_args = argparse.Namespace(**{**vars(args), 'league': league})
            if args.from_capture:
                capture = json.loads(args.from_capture.read_text())
            else:
                capture = asyncio.run(browser_capture(run_args))
            stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
            directory = args.output_dir / f'{capture.get("league", league)}_{stamp}'
            write_capture(capture, directory); directories.append(directory)
        if args.league == 'all' and not args.from_capture:
            combined_outputs = {}
            for filename, suffix in [('players.csv','totals' if args.units == 'total' else 'source_per90'), ('players_per90.csv','per90')]:
                combined = pd.concat([pd.read_csv(directory/filename) for directory in directories], ignore_index=True)
                if combined.duplicated(['source_player_id','Comp','Squad']).any():
                    raise ValueError('Repeated player/club/competition records in combined export.')
                path = ROOT/'data'/f'players_advanced_whoscored_2026_2027_{suffix}.csv'
                combined.to_csv(path, index=False); print('Combined:', path, flush=True)
                combined_outputs[suffix] = {'file': str(path), 'rows': len(combined),
                    'columns': combined.columns.tolist(), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
            (ROOT/'whoscored_enrichment/collection_manifest.json').write_text(json.dumps({
                'status':'collected_not_imported', 'collected_at_utc':datetime.now(timezone.utc).isoformat(),
                'season':'2026/2027', 'directories':[str(d) for d in directories],
                'reporting_cutoff_date':None, 'original_experiment_modified':False,
                'rows_by_league': combined.groupby('Comp').size().to_dict(),
                'outputs': combined_outputs,
                'note':'Current source snapshot; not merged into the November 2 experiment.'},indent=2))
    except Exception as error:
        parser.exit(2, f'Collection stopped: {error}\n')


if __name__ == '__main__':
    main()
