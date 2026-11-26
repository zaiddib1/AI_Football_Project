/* Same-origin feed used by the Detailed view. Option values and the endpoint
 * were observed from ordinary WhoScored browser requests on 21 November 2026. */
async function collectWhoScoredDetailed(options = {}) {
  const capture = { format_version: 2, source: 'WhoScored', status: 'started', pages: [],
    source_urls: [], started_at_utc: new Date().toISOString(), reporting_cutoff_date: null };
  const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));
  const routes = { '2': ['epl','eng Premier League'], '4': ['laliga','es La Liga'],
    '3': ['bundesliga','de Bundesliga'], '5': ['seriea','it Serie A'], '22': ['ligue1','fr Ligue 1'] };
  const text = node => (node?.textContent || '').replace(/\s+/g, ' ').trim();
  const report = message => { if (typeof window.footballCollectorProgress === 'function') window.footballCollectorProgress(message); };
  try {
    if (location.protocol !== 'https:' || !['whoscored.com','www.whoscored.com'].includes(location.hostname)) throw new Error('Open the WhoScored league player-statistics page.');
    if (/sorry, you have been blocked|verify you are human|just a moment/i.test(document.title + document.body.innerText)) {
      capture.status = 'blocked'; throw new Error('Normal WhoScored access was not granted.');
    }
    const tournament = location.pathname.match(/\/tournaments\/(\d+)/i)?.[1];
    const stage = location.pathname.match(/\/stages\/(\d+)/i)?.[1];
    const seasonId = location.pathname.match(/\/seasons\/(\d+)/i)?.[1];
    if (!routes[tournament] || !stage || !seasonId) throw new Error('A league season/stage player-statistics URL is required.');
    [capture.league, capture.competition] = routes[tournament];
    if (options.league && options.league !== capture.league) throw new Error('The selected tournament differs from the requested league.');
    const season = text(document.querySelector('select[id*="seasons"]')?.selectedOptions[0]);
    if (!['2026/2027','2026/27','2026-2027'].includes(season)) throw new Error('The visible season selector is not 2026/2027.');
    capture.observed_season = '2026/2027'; capture.page_title = document.title;
    capture.source_urls.push(location.href);
    const detailed = document.querySelector('a[href="#stage-top-player-stats-detailed"]');
    if (!detailed) throw new Error('The Detailed player-statistics tab is unavailable.');
    detailed.click();
    const deadline = Date.now() + 15000;
    while ((!document.querySelector('#category') || !document.querySelector('#statsAccumulationType')) && Date.now() < deadline) await sleep(100);
    const categorySelect = document.querySelector('#category'), unitSelect = document.querySelector('#statsAccumulationType');
    if (!categorySelect || !unitSelect) throw new Error('Detailed controls did not load.');
    const units = options.units || 'total';
    if (!['total','per90'].includes(units)) throw new Error('Only totals and per90 values are supported.');
    const unitOption = [...unitSelect.options].find(option => units === 'total' ? /^total$/i.test(text(option)) : /^per\s*90\s*mins?$/i.test(text(option)));
    if (!unitOption) throw new Error('The source does not offer the requested unit option.');
    capture.unit_option_mapping = [...unitSelect.options].map(option => ({label:text(option),value:option.value}));
    const categories = options.categories || ['Key passes','Passes','Dribbles','Aerial','Clearances','Blocks','Possession loss'];
    for (const category of categories) {
      const currentSelect = document.querySelector('#category');
      const choice = [...currentSelect.options].find(option => text(option).toLowerCase() === category.toLowerCase());
      if (!choice) throw new Error(`Detailed category ${category} is unavailable.`);
      currentSelect.value = choice.value; currentSelect.dispatchEvent(new Event('change', {bubbles:true}));
      await sleep(options.delay_ms || 3500);
      const sub = document.querySelector('#subcategory');
      const requestedSubcategory = options.subcategories?.[category];
      if (requestedSubcategory && ![...(sub?.options || [])].some(option => option.value === requestedSubcategory)) throw new Error(`Subcategory ${requestedSubcategory} is unavailable for ${category}.`);
      const subcategory = requestedSubcategory || sub?.options[0]?.value || 'all';
      const seen = new Set(); let expected = null;
      for (let number = 1; number <= (options.max_pages || 100); number++) {
        const url = new URL('/statisticsfeed/1/getplayerstatistics', location.origin);
        const query = { category:choice.value, subcategory, statsAccumulationType:unitOption.value,
          isCurrent:'true',playerId:'',teamIds:'',matchId:'',stageId:stage,tournamentOptions:tournament,
          sortBy:'Rating',sortAscending:'',age:'',ageComparisonType:'0',appearances:'',appearancesComparisonType:'0',
          field:'',nationality:'',positionOptions:"'FW','AML','AMC','AMR','ML','MC','MR','DMC','DL','DC','DR','GK','Sub'",
          timeOfTheGameEnd:'5',timeOfTheGameStart:'0',isMinApp:'false',page:String(number),
          includeZeroValues:'true',numberOfPlayersToPick:String(options.page_size || 500),incPens:'' };
        Object.entries(query).forEach(([name,value]) => url.searchParams.set(name,value));
        const response = await fetch(url, {headers:{Accept:'application/json','X-Requested-With':'XMLHttpRequest'}});
        if ([401,403,429].includes(response.status)) { capture.status = 'blocked'; throw new Error(`Statistics feed returned HTTP ${response.status}.`); }
        if (!response.ok) throw new Error(`Statistics feed returned HTTP ${response.status}.`);
        const body = await response.json(), records = body.playerTableStats, paging = body.paging;
        if (!Array.isArray(records) || !paging || !Array.isArray(body.statColumns)) throw new Error('Unexpected statistics feed schema.');
        if (paging.currentPage !== number || !Number.isInteger(paging.totalResults)) throw new Error('Unexpected pagination metadata.');
        if (expected !== null && expected !== paging.totalResults) throw new Error('The row count changed during pagination.');
        expected = paging.totalResults;
        for (const row of records) {
          if (String(row.seasonId) !== seasonId || row.seasonName !== '2026/2027' || String(row.tournamentId) !== tournament) throw new Error('A feed record belongs to a different season or tournament.');
          const id = `${row.playerId}|${row.teamId}`;
          if (seen.has(id)) throw new Error('Duplicate player/club IDs during feed pagination.');
          seen.add(id);
        }
        capture.pages.push({category, page_number:number, source_url:location.href, request_url:url.href,
          season:'2026/2027',competition:capture.competition,units,units_labels:[text(unitOption)],
          unit_option_value:unitOption.value, paging,stat_columns:body.statColumns,records,
          captured_at_utc:new Date().toISOString()});
        report(`${category}: ${seen.size}/${expected} player/club records (${units})`);
        if (number >= paging.totalPages) {
          if (seen.size !== expected) throw new Error('Feed pagination ended before every declared record was captured.');
          break;
        }
        if (number === (options.max_pages || 100)) throw new Error('Page cap reached before all source records were captured.');
        await sleep(options.delay_ms || 3500);
      }
    }
    capture.status = 'complete';
    capture.pagination_evidence = 'Unique player/club records equal the feed totalResults for every requested category.';
    capture.filters = {scope:'single domestic league stage',appearances:'all',positions:'all',location:'overall',game_minutes:'0–90'};
    capture.warnings = ['The reporting cutoff is not inferred from retrieval time. Verify it before importing into the frozen experiment.'];
  } catch(error) { if (capture.status !== 'blocked') capture.status = 'incomplete'; capture.error = error.message; }
  capture.finished_at_utc = new Date().toISOString();
  return capture;
}
