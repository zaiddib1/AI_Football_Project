/* Shared collector for Playwright and the bookmarklet. Reads rendered public tables. */
async function collectWhoScored(options = {}) {
  const leagues = {
    epl: { id: '2', competition: 'eng Premier League' },
    laliga: { id: '4', competition: 'es La Liga' },
    bundesliga: { id: '3', competition: 'de Bundesliga' },
    seriea: { id: '5', competition: 'it Serie A' },
    ligue1: { id: '22', competition: 'fr Ligue 1' }
  };
  const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));
  const clean = value => String(value || '').replace(/\s+/g, ' ').trim();
  const visible = node => Boolean(node.getClientRects().length) && getComputedStyle(node).visibility !== 'hidden';
  const unitOf = label => {
    const value = clean(label).toLowerCase().replace(/[-_]/g, ' ');
    if (/^(totals?|overall totals?)$/.test(value)) return 'total';
    if (/^(per\s*90(?:\s*(?:min|mins|minutes))?|90\s*mins?)$/.test(value)) return 'per90';
    if (/^(per\s*(?:game|match)|average|averages)$/.test(value)) return 'per_game';
    return null;
  };
  const canonicalSeason = text => clean(text).replace(/2026\s*[-/]\s*(?:2027|27)\b/, '2026/2027');
  const active = node => node.matches('[aria-selected="true"], [aria-pressed="true"], :checked, .active, .selected, .current') ||
    Boolean(node.parentElement && node.parentElement.matches('li.active, li.selected, li.current'));
  const capture = { format_version: 1, source: 'WhoScored', status: 'started',
    requested_season: options.season || '2026/2027', requested_units: options.units || 'total',
    started_at_utc: new Date().toISOString(), source_urls: [], pages: [], warnings: [] };
  const report = message => { if (typeof window.footballCollectorProgress === 'function') window.footballCollectorProgress(message); };
  function checkAccess() {
    if (!['whoscored.com', 'www.whoscored.com'].includes(location.hostname)) throw new Error('Open an HTTPS WhoScored page first.');
    if (location.protocol !== 'https:') throw new Error('HTTPS is required.');
    const text = document.body ? document.body.innerText : '';
    if (/sorry, you have been blocked|you are unable to access|verify you are human|checking your browser|just a moment/i.test(text + document.title)) {
      const error = new Error('WhoScored access is blocked or requires a person to complete its access check.');
      error.blocked = true; throw error;
    }
  }
  function controls() {
    const selections = [...document.querySelectorAll('select')].filter(visible).map(select => ({
      id: select.id, name: select.name,
      selected: [...select.selectedOptions].map(option => ({ text: clean(option.textContent), value: option.value }))
    }));
    const checks = [...document.querySelectorAll('input[type="checkbox"],input[type="radio"]')].filter(visible).map(input => ({
      id: input.id, name: input.name, type: input.type, checked: input.checked,
      label: input.id ? clean(document.querySelector(`label[for="${CSS.escape(input.id)}"]`)?.textContent) : ''
    }));
    return [...selections, ...checks];
  }
  function observedSeason() {
    const found = controls().filter(control => /season/i.test(control.id + control.name))
      .flatMap(control => (control.selected || []).map(option => canonicalSeason(option.text)));
    const seasons = [...new Set(found.filter(text => /^20\d\d\/20\d\d$/.test(text)))];
    if (seasons.length !== 1) throw new Error('Cannot verify one selected season from the visible season selector.');
    return seasons[0];
  }
  function observedLeague() {
    const match = location.pathname.match(/\/tournaments\/(\d+)(?:\/|$)/i);
    if (!match) throw new Error('Open a single domestic-league statistics page, with its tournament ID in the URL.');
    const found = Object.entries(leagues).find(([, value]) => value.id === match[1]);
    if (!found) throw new Error('This is not one of the five configured domestic leagues.');
    if (options.league && options.league !== found[0]) throw new Error('The page tournament differs from the requested league.');
    return { key: found[0], ...found[1] };
  }
  function observedUnits() {
    const labels = [];
    for (const select of document.querySelectorAll('select')) {
      if (!visible(select) || ![...select.options].some(option => unitOf(option.textContent))) continue;
      for (const option of select.selectedOptions) if (unitOf(option.textContent)) labels.push(clean(option.textContent));
    }
    for (const node of document.querySelectorAll('a,button,[role="tab"],input[type="radio"]')) {
      if (!visible(node) || !active(node)) continue;
      let label = clean(node.textContent);
      if (node.matches('input') && node.id) label = clean(document.querySelector(`label[for="${CSS.escape(node.id)}"]`)?.textContent);
      if (unitOf(label)) labels.push(label);
    }
    const units = [...new Set(labels.map(unitOf))];
    if (units.length !== 1) throw new Error('Cannot verify one active Total/Per 90/Per game control. Select Total or Per 90 explicitly.');
    return { units: units[0], labels };
  }
  function tableSnapshot() {
    const tables = [...document.querySelectorAll('table')].filter(visible).filter(table => {
      const labels = [...table.querySelectorAll('thead th')].map(node => clean(node.textContent).toLowerCase());
      return labels.includes('apps') && labels.includes('mins') &&
        [...table.querySelectorAll('tbody a[href]')].some(link => /\/players\/\d+(?:\/|$)/i.test(link.getAttribute('href')));
    });
    if (tables.length !== 1) throw new Error(`Expected one visible player table; found ${tables.length}. Select a single player-statistics tab.`);
    const table = tables[0];
    const headerRows = [...table.querySelectorAll('thead tr')];
    const headerRow = headerRows[headerRows.length - 1];
    if (!headerRow) throw new Error('The player table has no readable column headings.');
    const headers = [...headerRow.cells].map(cell => ({ label: clean(cell.innerText),
      title: clean(cell.getAttribute('title') || cell.querySelector('[title]')?.getAttribute('title')) }));
    const rows = [...table.querySelectorAll('tbody tr')].map(row => {
      const links = [...row.querySelectorAll('a[href]')];
      const player = links.find(link => /\/players\/\d+(?:\/|$)/i.test(link.getAttribute('href')));
      if (!player) return null;
      const team = links.find(link => /\/teams\/\d+(?:\/|$)/i.test(link.getAttribute('href')));
      const cells = [...row.cells].map(cell => clean(cell.innerText));
      if (cells.length !== headers.length) throw new Error('Column counts differ between player rows and headings; refusing to guess alignment.');
      const name = player.cloneNode(true); name.querySelectorAll('.table-ranking').forEach(node => node.remove());
      return { player_id: player.getAttribute('href').match(/\/players\/(\d+)/i)[1],
        player_name: clean(name.textContent), team_id: team?.getAttribute('href').match(/\/teams\/(\d+)/i)?.[1] || null,
        team_name: clean(team?.textContent).replace(/,\s*$/, ''), cells };
    }).filter(Boolean);
    if (!rows.length) throw new Error('The selected player table is empty.');
    return { table, headers, rows };
  }
  function fingerprint(table) { return JSON.stringify([table.headers, table.rows]); }
  async function changed(before) {
    const deadline = Date.now() + (options.change_timeout_ms || 15000);
    do {
      await sleep(200); checkAccess();
      try { const now = tableSnapshot(); if (fingerprint(now) !== before) return now; }
      catch (error) { if (!/visible player table|empty/.test(error.message)) throw error; }
    } while (Date.now() < deadline);
    throw new Error('The table did not change after clicking. Pagination/category capture stopped.');
  }
  const clickables = () => [...document.querySelectorAll('a,button,[role="tab"]')].filter(visible);
  async function selectUnits() {
    const requested = capture.requested_units;
    if (!['total', 'per90'].includes(requested)) throw new Error('Only total or per90 capture is supported.');
    try { if (observedUnits().units === requested) return; } catch (_) { /* inspect available controls next */ }
    const selectors = [...document.querySelectorAll('select')].filter(visible).filter(select =>
      [...select.options].some(option => unitOf(option.textContent) === requested));
    if (selectors.length === 1) {
      const select = selectors[0];
      select.value = [...select.options].find(option => unitOf(option.textContent) === requested).value;
      select.dispatchEvent(new Event('change', { bubbles: true }));
    } else {
      const buttons = clickables().filter(node => unitOf(node.textContent) === requested);
      if (buttons.length !== 1) throw new Error('Cannot uniquely identify the requested accumulation control. Select it manually and rerun.');
      buttons[0].click();
    }
    await sleep(options.delay_ms || 3500);
    if (observedUnits().units !== requested) throw new Error('The page did not confirm the requested units.');
  }
  function paginationControl(table, direction) {
    const patterns = { next: /^(?:next(?:\s*page)?|[›»>])$/i,
      first: /^(?:first(?:\s*page)?|[«]|<<)$/i, previous: /^(?:prev(?:ious)?(?:\s*page)?|[‹<])$/i };
    let scope = table.parentElement;
    for (let level = 0; level < 5 && scope; level++, scope = scope.parentElement) {
      const all = [...scope.querySelectorAll('a,button,[role="button"]')].filter(visible).filter(node =>
        patterns[direction].test(clean(node.textContent)) ||
        patterns[direction].test(clean(node.getAttribute('aria-label') || node.title)));
      if (!all.length) continue;
      const enabled = all.filter(node => !node.disabled && node.getAttribute('aria-disabled') !== 'true' &&
        !node.matches('.disabled') && !node.closest('li.disabled'));
      if (enabled.length > 1) throw new Error(`More than one enabled ${direction} control; cannot safely select pagination.`);
      return enabled[0] || null;
    }
    return null;
  }
  try {
    checkAccess();
    const league = observedLeague(); capture.league = league.key; capture.competition = league.competition;
    capture.page_title = document.title; capture.observed_season = observedSeason();
    if (capture.observed_season !== capture.requested_season) throw new Error('Selected season does not match the requested season.');
    const categories = options.categories || ['Summary', 'Passing', 'Offensive', 'Defensive'];
    if (!categories.length) throw new Error('Select at least one statistics category.');
    for (const category of categories) {
      const before = fingerprint(tableSnapshot());
      let nodes = clickables().filter(node => clean(node.textContent).toLowerCase() === category.toLowerCase());
      const tabs = nodes.filter(node => (node.getAttribute('href') || '').startsWith('#'));
      if (tabs.length === 1) nodes = tabs;
      if (category !== 'Current') {
        if (nodes.length !== 1) throw new Error(`Cannot uniquely locate the ${category} tab. Open it manually and use --categories Current.`);
        if (!active(nodes[0])) { nodes[0].click(); await changed(before); }
      }
      await selectUnits();
      const initial = tableSnapshot(), first = paginationControl(initial.table, 'first');
      if (first) { first.click(); await changed(fingerprint(initial)); }
      else if (paginationControl(initial.table, 'previous')) throw new Error('Start on page 1; an enabled Previous control was found without a First control.');
      const unit = observedUnits();
      if (unit.units !== capture.requested_units) throw new Error('Units changed unexpectedly.');
      const seenPages = new Set(), seenPlayers = new Set();
      for (let number = 1; number <= (options.max_pages || 100); number++) {
        checkAccess();
        if (observedSeason() !== capture.requested_season || observedLeague().key !== league.key) throw new Error('Season or league changed during capture.');
        if (observedUnits().units !== unit.units) throw new Error('Units changed during pagination.');
        const snapshot = tableSnapshot(), signature = fingerprint(snapshot);
        if (seenPages.has(signature)) throw new Error('A repeated page was detected.');
        seenPages.add(signature);
        for (const row of snapshot.rows) {
          const key = `${row.player_id}|${row.team_id || row.team_name}`;
          if (seenPlayers.has(key)) throw new Error('A player/club row repeats across pages; refusing partial/overlapping pagination.');
          seenPlayers.add(key);
        }
        if (!capture.source_urls.includes(location.href)) capture.source_urls.push(location.href);
        capture.pages.push({ category, page_number: number, source_url: location.href,
          season: capture.observed_season, competition: league.competition, units: unit.units,
          units_labels: unit.labels, selected_controls: controls(),
          headers: snapshot.headers, rows: snapshot.rows, captured_at_utc: new Date().toISOString() });
        report(`${category}: page ${number}, ${seenPlayers.size} player/club rows`);
        const next = paginationControl(snapshot.table, 'next');
        if (!next) break;
        if (number === (options.max_pages || 100)) throw new Error('Page limit reached before the end of the table.');
        await sleep(options.delay_ms || 3500);
        next.click(); await changed(signature);
      }
    }
    capture.status = 'complete'; capture.pagination_evidence = 'No enabled Next control remained in each captured category; source filters are preserved.';
    capture.reporting_cutoff_date = null;
    capture.warnings.push('The reporting cutoff is not inferred from retrieval time. Verify it separately before importing.');
  } catch (error) {
    capture.status = error.blocked ? 'blocked' : 'incomplete'; capture.error = error.message;
  }
  capture.finished_at_utc = new Date().toISOString();
  return capture;
}
