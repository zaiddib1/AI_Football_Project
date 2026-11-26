"""Cache current WhoScored baseline counts for the append-only position experiments."""
import argparse
import asyncio
import json
from pathlib import Path

from scrape_whoscored_stats import browser_capture, LEAGUE_ROUTES

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'data/position_comparison_sources'
CATEGORIES = ['Shots','Goals','Assists','Tackles','Interception','Fouls','Offsides','Cards','Passes']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--league', choices=[*LEAGUE_ROUTES, 'all'], default='all')
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    for league in list(LEAGUE_ROUTES) if args.league == 'all' else [args.league]:
        path = OUT/f'{league}_baseline.json'
        if path.exists():
            print('Cached:', path, flush=True)
            continue
        options = argparse.Namespace(league=league, url=None, headless=False, interactive=False,
            mode='detailed', categories=','.join(CATEGORIES), units='total', max_pages=100,
            subcategories={'Shots':'accuracy', 'Passes':'type'})
        capture = asyncio.run(browser_capture(options))
        if capture['status'] != 'complete':
            (OUT/f'{league}_failed.json').write_text(json.dumps(capture, indent=2))
            raise RuntimeError(capture.get('error', capture['status']))
        path.write_text(json.dumps(capture, indent=2, ensure_ascii=False))
        print('Saved:', path, flush=True)
        print('Fields:', {p['category']:p['stat_columns'] for p in capture['pages']}, flush=True)


if __name__ == '__main__':
    main()
