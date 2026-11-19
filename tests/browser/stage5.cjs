// Run against the actual Reflex frontend and serve_stage1 with PRACTICE_GRAPH_FIXTURE.
const assert = require('node:assert/strict');
const {mkdir, writeFile} = require('node:fs/promises');
const {join} = require('node:path');
const {chromium} = require(process.env.PLAYWRIGHT_MODULE || 'playwright');

const url = process.argv[2] || 'http://127.0.0.1:8301/';
const evidence = process.env.PRACTICE_EVIDENCE;

async function main() {
  const browser = await chromium.launch({args: ['--no-sandbox', '--renderer-process-limit=2']});
  try {
    const context = await browser.newContext({viewport: {width: 1440, height: 900}});
    const page = await context.newPage();
    page.setDefaultTimeout(15000);
    await page.addInitScript(() => {
      const observed = new Set();
      const BrowserResizeObserver = window.ResizeObserver;
      window.graphObserverCount = () => observed.size;
      window.ResizeObserver = class extends BrowserResizeObserver {
        observe(element, ...options) {
          if (element.matches('.balanced-graph .graph-canvas')) observed.add(this);
          return super.observe(element, ...options);
        }
        disconnect() {
          observed.delete(this);
          return super.disconnect();
        }
      };
    });
    const failures = [], events = [];
    page.on('pageerror', error => failures.push(error.message));
    page.on('console', message => {
      if (message.type() === 'error' && /hydrat|reflex|balanced/i.test(message.text())) failures.push(message.text());
    });
    page.on('response', response => {
      if (response.status() >= 400 && /knowledge_graph|graph_(layout|camera)|practice\.css|\.woff2/.test(response.url())) failures.push(response.url());
    });
    page.on('websocket', socket => socket.on('framesent', ({payload}) => {
      if (String(payload).includes('app_state')) events.push(String(payload));
    }));
    const graph = page.locator('.balanced-graph');
    const inspector = graph.locator('.concept-inspector');
    const workshop = page.getByRole('button', {name: 'Graph', exact: true});
    const study = page.getByRole('button', {name: 'Study', exact: true});
    const node = id => graph.locator(`[data-node="${id}"]`);
    const picker = page.getByRole('combobox', {name: 'Session map'});
    const turn = (page, prompt) => page.locator('.study-turn').filter({
      has: page.locator('.user-message').filter({hasText: prompt}),
    });
    const turnMap = (page, prompt) => turn(page, prompt).getByRole('button', {name: /^Open map:/});
    const assertQueryTooltip = async (page, trigger, queryText, keyboard = false) => {
      if (keyboard) {
        await trigger.scrollIntoViewIfNeeded();
        await trigger.focus();
      } else {
        await trigger.hover();
      }
      const tooltip = page.getByRole('tooltip');
      await tooltip.waitFor({timeout: 2500});
      assert.equal(await tooltip.textContent(), queryText);
      const panel = page.locator('.map-request-tooltip');
      const {left, right, themedText} = await panel.evaluate(el => {
        const {left, right} = el.getBoundingClientRect();
        return {left, right, themedText: getComputedStyle(el.querySelector('p')).color === getComputedStyle(el).color};
      });
      assert.equal(themedText, true);
      assert.ok(left >= 0 && right <= page.viewportSize().width);
      await trigger.press('Escape');
      await tooltip.waitFor({state: 'hidden'});
    };
    const ask = async prompt => {
      const input = page.locator('#chat-input');
      await page.waitForFunction(() => !document.querySelector('#chat-input')?.readOnly);
      await input.fill(prompt);
      await input.press('Control+Enter');
    };
    const restore = async () => {
      await page.evaluate(() => sessionStorage.removeItem('token'));
      await page.reload();
      await page.getByRole('textbox', {name: 'Ask a follow-up'}).waitFor();
    };
    const capture = async name => {
      if (evidence) await page.screenshot({path: join(evidence, `${name}.png`), animations: 'disabled'});
    };
    const checkMapTime = async result => {
      const time = result.locator('time');
      assert.match(await time.textContent(), /^\d{2}:\d{2}$/);
      assert.equal(await time.evaluate(el => getComputedStyle(el, '::before').width), '3px');
      const timestamp = await time.boundingBox();
      const link = await result.getByRole('button', {name: /^Open map:/}).boundingBox();
      const row = await result.boundingBox();
      const gap = timestamp.x - (link.x + link.width);
      assert.ok(gap > 0 && gap <= 16);
      assert.ok(timestamp.x + timestamp.width <= row.x + row.width);
    };
    const checkRelationsHeader = async () => {
      const table = graph.getByRole('table', {name: 'Relations in this map'});
      await table.getByRole('columnheader', {name: 'Source concept'}).waitFor();
      const heading = graph.locator('.canvas-caption').getByRole('heading');
      assert.equal(await heading.textContent(), 'Relations in this map');
      assert.equal(await graph.locator('.map-title').count(), 0);
      assert.equal(await table.locator('caption').count(), 0);
      assert.equal(await graph.locator('.graph-center').evaluate(el => {
        const center = getComputedStyle(el);
        const relations = getComputedStyle(el.querySelector('.relations-view'));
        return center.backgroundImage === 'none' && center.backgroundColor === relations.backgroundColor;
      }), true);
    };
    if (evidence) await mkdir(evidence, {recursive: true});

    await page.goto(url);
    await page.getByRole('heading', {name: 'Study your AI research notes.'}).waitFor();
    await page.locator('.welcome-diagram[data-draw]').waitFor();
    await ask('Visualize my knowledge about BERT');
    const firstMapResult = turn(page, 'Visualize my knowledge about BERT').locator('.map-result');
    await firstMapResult.waitFor({timeout: 3000});
    await checkMapTime(firstMapResult);
    await capture('study-map-with-time-desktop-dark');
    const cacheQueries = 'KV cache & grouped-query attention & PagedAttention block allocation';
    await assertQueryTooltip(page, turnMap(page, 'Visualize my knowledge about BERT'), cacheQueries);
    await turnMap(page, 'Visualize my knowledge about BERT').click();
    await node('kv').waitFor();
    assert.equal(await picker.count(), 0);
    const controls = graph.getByRole('complementary', {name: 'Map controls'});
    const types = controls.getByRole('heading', {name: 'Concept types'});
    const connected = inspector.getByRole('heading', {name: 'Most connected'});
    const typography = element => element.evaluate(el => {
      const style = getComputedStyle(el);
      return [style.fontSize, style.fontWeight];
    });
    assert.deepEqual(await typography(types), await typography(connected));
    const singleTypes = await types.boundingBox();
    const singleFind = await controls.getByRole('search').boundingBox();
    assert.ok(Math.abs(singleTypes.y - (await connected.boundingBox()).y) <= 3);
    const typeRow = controls.locator('.family-filters button').first();
    const connectedRow = inspector.locator('.connected-list button').first();
    assert.deepEqual(await typography(typeRow), await typography(connectedRow));
    const typeRowBox = await typeRow.boundingBox();
    const connectedRowBox = await connectedRow.boundingBox();
    assert.equal(typeRowBox.height, connectedRowBox.height);
    assert.ok(Math.abs(typeRowBox.y - connectedRowBox.y) <= 3);
    await capture('desktop-dark-single-map-overview');
    const mapTitle = graph.locator('.map-title');
    await mapTitle.filter({hasText: /^KV caching$/}).waitFor();
    await assertQueryTooltip(page, mapTitle, cacheQueries);
    assert.equal(await graph.locator('details.map-title').count(), 0);
    assert.equal(await inspector.locator('.overview-count').count(), 0);
    assert.equal(await page.locator('.practice-history').count(), 0);
    assert.equal(await page.getByRole('button', {name: 'Open current-session history'}).count(), 0);
    assert.equal(await graph.locator('.map-panel').getByText('Most connected').count(), 0);
    await page.keyboard.press('Tab');
    const edge = graph.locator('[data-edge-hit]').first();
    await edge.focus();
    await graph.locator('.graph-edge.is-hovered').waitFor({state: 'attached'});
    assert.deepEqual(await edge.evaluate(el => {
      const style = getComputedStyle(el);
      return [style.outlineStyle, style.boxShadow, el.matches(':focus-visible')];
    }), ['none', 'none', true]);
    await page.locator('.balanced-graph').evaluate(element => {element.dataset.testInstance = 'retained';});
    const mapAction = graph.getByRole('button', {name: 'Map', exact: true});
    const relations = graph.getByRole('button', {name: 'Relations', exact: true});
    await node('kv').focus();
    await page.keyboard.press('Enter');
    await inspector.getByRole('heading', {name: 'KV cache', exact: true}).waitFor();
    const initialLocalEvents = events.length;
    await graph.getByRole('button', {name: 'Zoom in'}).click();
    const zoom = graph.getByLabel('Zoom level');
    const zoomed = await zoom.textContent();
    const card = await node('kv').boundingBox();
    const position = await node('kv').getAttribute('style');
    await page.mouse.move(card.x + card.width / 2, card.y + card.height / 2);
    await page.mouse.down();
    await page.mouse.move(card.x + card.width / 2 + 38, card.y + card.height / 2 + 15, {steps: 12});
    await page.mouse.up();
    const dragged = await node('kv').getAttribute('style');
    assert.notEqual(dragged, position);
    assert.equal(events.length, initialLocalEvents);
    await study.click();
    await page.waitForFunction(() => window.graphObserverCount() === 0);
    await workshop.click();
    await graph.waitFor();
    assert.equal(await graph.getAttribute('data-test-instance'), 'retained');
    assert.equal(await zoom.textContent(), zoomed);
    assert.equal(await node('kv').getAttribute('style'), dragged);
    assert.equal(await node('kv').getAttribute('aria-pressed'), 'true');
    const fit = graph.getByRole('button', {name: 'Fit graph'});
    assert.deepEqual(await fit.locator('path').evaluate(path => {
      const {x, y, width, height} = path.getBBox();
      return [x, y, width, height];
    }), [2, 2, 12, 12]);
    const mapStage = graph.locator('.graph-stage');
    const beforeFit = await mapStage.evaluate(stage => stage.style.transform);
    await fit.click();
    assert.notEqual(await mapStage.evaluate(stage => stage.style.transform), beforeFit);
    await inspector.getByRole('button', {name: 'Focus on map'}).click();
    await capture('desktop-dark-selected');
    await page.getByRole('button', {name: 'Switch to light theme'}).click();
    const clearSelection = graph.getByRole('button', {name: 'Clear selection', exact: true});
    const beforeRelations = await clearSelection.boundingBox();
    const workspaceHeading = graph.locator('.canvas-caption h2');
    const mapHeadingType = await typography(workspaceHeading);
    const mapHeadingBox = await workspaceHeading.boundingBox();
    await relations.click();
    await checkRelationsHeader();
    assert.deepEqual(await typography(workspaceHeading), mapHeadingType);
    const relationHeadingBox = await workspaceHeading.boundingBox();
    assert.deepEqual([relationHeadingBox.x, relationHeadingBox.y], [mapHeadingBox.x, mapHeadingBox.y]);
    assert.deepEqual(await clearSelection.boundingBox(), beforeRelations);
    await capture('desktop-light-relations');
    await clearSelection.click();
    await inspector.getByRole('heading', {name: 'Map overview'}).waitFor();
    await capture('desktop-light-overview');
    await mapAction.click();
    await graph.getByRole('heading', {name: 'Knowledge graph', exact: true}).waitFor();
    const find = graph.getByRole('searchbox', {name: 'Find a concept'});
    await find.fill('KV cache');
    await find.press('Enter');
    await inspector.getByRole('heading', {name: 'KV cache', exact: true}).waitFor();
    assert.equal(await graph.locator('.graph-canvas').evaluate(el => getComputedStyle(el).outlineStyle), 'none');
    assert.equal(await find.evaluate(el => el === document.activeElement), true);
    await graph.getByRole('button', {name: /^Techniques/}).click();
    assert.equal(await node('kv').count(), 0);
    await graph.getByRole('button', {name: /^Techniques/}).click();
    await find.press('Enter');
    await inspector.getByRole('button', {name: 'Explain this'}).click();
    const composer = page.getByRole('textbox', {name: 'Ask a follow-up'});
    await page.waitForFunction(() => document.activeElement?.value === 'Explain KV cache.');
    assert.equal(await composer.inputValue(), 'Explain KV cache.');
    assert.equal(await page.locator('.user-message').count(), 1);

    await ask('Study local notes');
    await page.getByText('Local explanation one', {exact: true}).waitFor();
    await ask('Map the cache');
    const cacheMap = turnMap(page, 'Map the cache');
    await cacheMap.waitFor();
    assert.equal(await cacheMap.getAttribute('data-state'), null);
    await page.locator('.assistant-message').filter({hasText: 'Local explanation one'})
      .getByRole('button', {name: 'Sources & concepts'}).click();
    const panel = page.locator('.practice-sources');
    await panel.getByText('Decoder attention notes', {exact: true}).waitFor();
    assert.equal(await panel.getByRole('heading', {name: 'Related concepts'}).count(), 0);
    const studySourceType = await typography(panel.getByRole('heading', {name: 'Notes used as context'}));
    await cacheMap.click();
    await node('concept-cache').click();
    await inspector.getByRole('heading', {name: 'KV caching', exact: true}).waitFor();
    assert.deepEqual(await typography(inspector.getByRole('heading', {name: 'Knowledge graph sources'})),
      studySourceType);
    await controls.getByRole('heading', {name: 'Map controls'}).waitFor();
    assert.equal(await controls.getByRole('button', {name: 'Clear session'}).count(), 1);
    assert.equal(await controls.locator('select option:checked').textContent(), 'Cache map');
    assert.equal(await controls.getByText('Session map', {exact: true}).count(), 0);
    assert.equal(await controls.getByRole('combobox', {name: 'Session map'}).count(), 1);
    const multipleFind = await controls.getByRole('search').boundingBox();
    const multipleTypes = await types.boundingBox();
    assert.ok(multipleFind.y > singleFind.y);
    assert.ok(Math.abs((multipleTypes.y - multipleFind.y) - (singleTypes.y - singleFind.y)) < 1);
    await mapTitle.filter({hasText: /^Cache map$/}).waitFor();
    assert.equal(await mapTitle.getAttribute('tabindex'), null);
    assert.equal(await controls.getByText(/concepts · .* relations/).count(), 0);
    const untitled = inspector.locator('.source-note').filter({hasText: 'Sliding-window decoding'});
    await untitled.locator('.source-preview').waitFor();
    assert.equal(await untitled.locator('summary').getByText(/Note \d|Decoding \/ Sliding windows/).count(), 0);
    await capture('desktop-light-graph-sources');
    const graphNote = inspector.locator('.source-note').filter({hasText: 'Graph cache note'});
    await graphNote.locator('summary').click();
    await graphNote.locator('summary').getByText('Summary', {exact: true}).waitFor();
    await graphNote.getByText('Describes how key and value projections are cached.').waitFor();
    await graphNote.evaluate(el => Promise.allSettled(el.getAnimations({subtree: true}).map(animation => animation.finished)));
    await capture('desktop-light-graph-source-expanded');
    assert.equal(await graphNote.getByRole('button', {name: 'View in Sources & concepts'}).count(), 0);
    await untitled.locator('summary').click();
    assert.equal(await untitled.getByText('Decoding / Sliding windows', {exact: true}).count(), 0);
    const toggle = inspector.getByRole('button', {name: 'Show all (8)'});
    assert.equal(await inspector.locator('.source-list > li').count(), 5);
    await toggle.focus();
    await toggle.press('Enter');
    await inspector.getByText('Supplemental cache note 8', {exact: true}).waitFor();
    assert.equal(await inspector.locator('.source-list > li').count(), 8);
    const collapse = inspector.getByRole('button', {name: 'Show fewer'});
    assert.equal(await collapse.getAttribute('aria-expanded'), 'true');
    await capture('desktop-light-graph-sources-expanded');
    await collapse.press('Enter');
    assert.equal(await inspector.locator('.source-list > li').count(), 5);
    await picker.focus();
    await picker.selectOption({index: 0});
    await node('kv').waitFor();
    await page.waitForFunction(() => document.activeElement.matches('.map-panel select'));
    assert.equal(await node('kv').getAttribute('aria-pressed'), 'false');
    assert.equal(await inspector.getByRole('heading', {name: 'Most connected'}).count(), 1);
    await capture('desktop-light-map-selector');

    await study.click();
    await ask('Map at the rendering ceiling');
    const progress = page.getByRole('list', {name: 'Agent progress'});
    await progress.locator('li[data-outcome="running"]').getByText('Visualizer', {exact: true}).waitFor();
    assert.equal(await progress.locator('li[data-outcome="inactive"]').count(), 5);
    await turnMap(page, 'Map at the rendering ceiling').waitFor();
    const showExecution = turn(page, 'Map at the rendering ceiling').getByRole('button', {name: 'Execution details'});
    await showExecution.click();
    const graphExecution = page.getByRole('region', {name: 'Execution details'});
    await graphExecution.getByText('Map ready', {exact: true}).waitFor();
    assert.deepEqual(await graphExecution.locator('.stage-label strong').allTextContents(), ['Visualizer']);
    await capture('study-graph-result-light');
    await showExecution.click();
    assert.equal(await composer.isVisible(), true);
    await workshop.click();
    await node('kv').waitFor();
    await picker.selectOption({index: 1});
    await node('concept-cache').waitFor();
    const started = Date.now();
    await picker.selectOption({index: 2});
    await node('n47').waitFor({state: 'attached'});
    const ceilingPaintMs = Date.now() - started;
    assert.equal(await graph.locator('[data-node]').count(), 48);
    await study.click();
    const longRequest = 'Map above the rendering ceiling\nInclude the architecture, training objectives, and all the connections to related models in my notes.';
    await ask(longRequest);
    await turnMap(page, 'Map above the rendering ceiling').click();
    await graph.getByText(/too large for Map/).waitFor();
    assert.equal(await mapAction.isDisabled(), true);
    assert.equal(await graph.locator('tbody tr').count(), 97);
    await restore();
    await page.setViewportSize({width: 390, height: 844});
    const longTitle = 'Attention architectures and inference techniques across a collection of research notes on efficient transformer decoding';
    const longQueries = `${longTitle} & grouped query attention & transformer training objectives`;
    await checkMapTime(turn(page, 'Map above the rendering ceiling').locator('.map-result'));
    await assertQueryTooltip(page, turnMap(page, 'Map above the rendering ceiling'), longQueries, true);
    await workshop.click();
    await graph.getByText(/too large for Map/).waitFor();
    assert.equal(await graph.locator('tbody tr').count(), 97);
    await checkRelationsHeader();
    const openFilters = graph.getByRole('button', {name: 'Open graph filters'});
    await openFilters.press('Enter');
    const tools = graph.getByRole('dialog', {name: 'Graph controls'});
    await tools.waitFor();
    await page.keyboard.press('Escape');
    await tools.waitFor({state: 'hidden'});
    assert.equal(await openFilters.evaluate(el => document.activeElement === el), true);
    await page.setViewportSize({width: 1440, height: 900});

    const replay = await page.context().newPage();
    await replay.goto(url);
    await replay.locator('#chat-input').fill('Visualize my knowledge about BERT');
    await replay.locator('#chat-input').press('Control+Enter');
    await assertQueryTooltip(replay, turnMap(replay, 'Visualize my knowledge about BERT'), cacheQueries);
    await turnMap(replay, 'Visualize my knowledge about BERT').click();
    await replay.locator('[data-node="kv"]').waitFor();
    await replay.getByRole('button', {name: 'Study', exact: true}).click();
    await replay.getByRole('button', {name: 'Execution details'}).click();
    await replay.locator('.execution-origin').filter({hasText: 'Cached'}).waitFor();
    await replay.close();

    // Real mobile-size behavior, keyboard dismissal, and reduced motion.
    for (const width of [1280, 1279, 960, 959]) {
      await page.setViewportSize({width, height: 900});
      assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true);
    }
    await page.setViewportSize({width: 390, height: 844});
    await page.emulateMedia({reducedMotion: 'reduce'});
    await picker.focus();
    await picker.selectOption({index: 0});
    await node('kv').waitFor();
    await page.waitForFunction(() => document.activeElement.matches('.map-picker-mobile select'));
    await graph.getByRole('heading', {name: 'Knowledge graph', exact: true}).waitFor();
    assert.equal(await page.getByRole('button', {name: 'Open current-session history'}).count(), 0);
    await capture('mobile-light-map');
    await assertQueryTooltip(page, mapTitle, cacheQueries, true);
    const touch = await context.newCDPSession(page);
    await touch.send('Emulation.setTouchEmulationEnabled', {enabled: true});
    await node('kv').scrollIntoViewIfNeeded();
    const mobileCard = await node('kv').boundingBox();
    await touch.send('Input.dispatchTouchEvent', {type: 'touchStart', touchPoints: [{x: mobileCard.x + mobileCard.width / 2, y: mobileCard.y + mobileCard.height / 2}]});
    await touch.send('Input.dispatchTouchEvent', {type: 'touchEnd', touchPoints: []});
    await graph.locator('.inspector-peek').click();
    const drawer = graph.locator('.inspector-drawer');
    await drawer.waitFor();
    await capture('mobile-light-inspector');
    await page.keyboard.press('Escape');
    await drawer.waitFor({state: 'hidden'});
    assert.equal(await graph.locator('.inspector-peek').evaluate(el => document.activeElement === el), true);
    await graph.getByRole('button', {name: 'Clear selection', exact: true}).click();
    await graph.getByRole('button', {name: 'Map overview'}).click();
    const overview = graph.getByRole('dialog', {name: 'Map overview'});
    await overview.getByRole('heading', {name: 'Most connected'}).waitFor();
    await overview.getByRole('button', {name: 'KV cache', exact: true}).click();
    await page.waitForFunction(() => document.activeElement.matches('.inspector-drawer h2'));
    await page.keyboard.press('Escape');
    await graph.getByRole('button', {name: 'Open graph filters'}).click();
    await graph.locator('dialog[open]').waitFor();
    await page.keyboard.press('Escape');
    await graph.locator('dialog[open]').waitFor({state: 'hidden'});
    await page.getByRole('button', {name: 'Switch to dark theme'}).click();
    await relations.click();
    await checkRelationsHeader();
    assert.equal(await picker.isVisible(), true);
    await capture('mobile-dark-relations');
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true);
    await picker.selectOption({index: 1});
    await node('concept-cache').waitFor();
    await node('concept-cache').click();
    await graph.locator('.inspector-peek').click();
    const mobileSources = drawer.getByRole('button', {name: 'Show all (8)'});
    await mobileSources.click();
    await drawer.getByText('Supplemental cache note 8', {exact: true}).scrollIntoViewIfNeeded();
    assert.equal(await drawer.locator('.source-list').evaluate(list => list.scrollHeight <= list.clientHeight), true);
    await capture('mobile-dark-graph-sources-expanded');
    await page.keyboard.press('Escape');
    await drawer.waitFor({state: 'hidden'});
    await graph.getByRole('button', {name: 'Open graph filters'}).click();
    const clearSession = page.getByRole('button', {name: 'Clear session', exact: true});
    await clearSession.click();
    const confirm = page.getByRole('dialog', {name: 'Clear this session?'});
    await confirm.getByRole('button', {name: 'Keep studying'}).click();
    assert.equal(await clearSession.evaluate(el => document.activeElement === el), true);
    assert.equal(await graph.isVisible(), true);
    await clearSession.click();
    await page.setViewportSize({width: 960, height: 844});
    await confirm.getByRole('button', {name: 'Keep studying'}).click();
    const desktopClear = graph.locator('.map-panel .clear-session');
    await page.waitForFunction(() => document.activeElement.matches('.map-panel .clear-session'));
    await desktopClear.click();
    await confirm.getByRole('button', {name: 'Clear session', exact: true}).click();
    await page.getByRole('heading', {name: 'Study your AI research notes.'}).waitFor();

    const empty = await context.newPage();
    await empty.setViewportSize({width: 1440, height: 900});
    await empty.goto(url);
    await empty.locator('#chat-input').fill('Study without retrieval');
    await empty.locator('#chat-input').press('Control+Enter');
    await empty.getByRole('heading', {name: 'Session history'}).waitFor();
    await empty.getByRole('button', {name: 'Graph', exact: true}).click();
    await empty.getByRole('heading', {name: 'No knowledge graph yet'}).waitFor();
    assert.equal(await empty.getByRole('button', {name: 'Clear session', exact: true}).count(), 0);
    const emptyHeading = empty.getByRole('heading', {name: 'No knowledge graph yet'});
    for (const theme of ['dark', 'light']) {
      const nextTheme = empty.getByRole('button', {name: `Switch to ${theme} theme`});
      if (await nextTheme.isVisible()) await nextTheme.click();
      for (const [size, width, height] of [['desktop', 1440, 900], ['mobile', 390, 844]]) {
        await empty.setViewportSize({width, height});
        await empty.mouse.move(0, 65);
        await empty.waitForFunction(() => Math.abs(document.querySelector('.practice-app').clientHeight - innerHeight) < 1);
        const workspaceBox = await empty.getByRole('region', {name: 'Graph workspace'}).boundingBox();
        const headingBox = await emptyHeading.boundingBox();
        assert.ok(Math.abs(workspaceBox.x + workspaceBox.width / 2 - headingBox.x - headingBox.width / 2) < 2);
        assert.ok(Math.abs(workspaceBox.y + workspaceBox.height / 2 - headingBox.y - headingBox.height / 2) < 24);
        if (evidence) await empty.screenshot({path: join(evidence, `${size}-${theme}-empty-graph.png`), animations: 'disabled'});
      }
    }
    await empty.setViewportSize({width: 1440, height: 900});
    await empty.getByRole('button', {name: 'Study', exact: true}).click();
    await empty.getByRole('button', {name: 'Clear session', exact: true}).waitFor();
    await empty.close();
    assert.deepEqual(failures, []);
    const result = {ceilingPaintMs, ceiling: {nodes: 48, edges: 96}, fallbackRows: 97, localDragAndZoomEvents: 0};
    console.log(JSON.stringify(result));
    if (evidence) await writeFile(join(evidence, 'metrics.json'), JSON.stringify(result, null, 2) + '\n');
  } finally {
    await browser.close();
  }
}

main().catch(error => {console.error(error); process.exitCode = 1;});
