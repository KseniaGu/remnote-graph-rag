const assert = require('node:assert/strict');
const {mkdirSync, writeFileSync} = require('node:fs');
const {join} = require('node:path');
const {chromium} = require(process.env.PLAYWRIGHT_MODULE || 'playwright');

const base = process.argv[2] || 'http://127.0.0.1:8381/';
const destination = process.env.STAGE4_EVIDENCE;
const reference = process.env.STAGE4_REFERENCE;
const graph = page => page.locator('.balanced-graph');
const canvas = page => graph(page).locator('.graph-canvas');
const node = (page, id) => graph(page).locator(`[data-node="${id}"]`);
const choice = (page, name) => page.getByRole('group', {name: 'Graph presentation'}).getByRole('button', {name, exact: true});
const inspector = page => graph(page).locator('.concept-inspector');
const transform = page => page.locator('.graph-stage').evaluate(el => el.style.transform);
const positions = page => page.locator('[data-node]').evaluateAll(nodes =>
  Object.fromEntries(nodes.map(el => [el.dataset.node, [el.style.left, el.style.top, el.offsetWidth, el.offsetHeight]])));
const zoom = page => graph(page).getByLabel('Zoom level', {exact: true});
const settle = page => page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
const poll = (page, expression, arg) => page.waitForFunction(expression, arg);
const expectText = (locator, text) => locator.filter({hasText: text}).waitFor();
const choose = (page, name) => page.locator('#fixture').selectOption(name);
const close = (a, b, epsilon = 1) => assert.ok(Math.abs(a - b) < epsilon, `${a} is not near ${b}`);

async function navigate(page) {
  page.setDefaultTimeout(8000);
  await page.goto(base, {waitUntil: 'domcontentloaded'});
  await choice(page, 'Map').waitFor();
}

async function checkDesktop(browser, errors) {
  const page = await browser.newPage({viewport: {width: 1440, height: 900}});
  page.on('pageerror', err => errors.push(String(err)));
  page.on('console', message => {
    if (message.type() === 'error' && !message.text().includes('Permissions policy violation: unload is not allowed')) {
      errors.push(message.text());
    }
  });
  let outbound = 0;
  page.on('websocket', ws => ws.on('framesent', ({payload}) => {
    if (String(payload).startsWith('42')) outbound++;
  }));
  let releaseFont;
  const fontGate = new Promise(resolve => { releaseFont = resolve; });
  await page.route('**/fonts/archivo.woff2', async route => {
    await fontGate;
    await route.continue();
  });
  await page.addInitScript(() => {
    const Observer = window.ResizeObserver;
    window.graphObservers = new Set();
    window.graphWheels = new Set();
    window.graphFrames = new Set();
    const {addEventListener, removeEventListener} = EventTarget.prototype;
    EventTarget.prototype.addEventListener = function(type, listener, options) {
      if (type === 'wheel' && this.classList?.contains('graph-canvas')) window.graphWheels.add(listener);
      return addEventListener.call(this, type, listener, options);
    };
    EventTarget.prototype.removeEventListener = function(type, listener, options) {
      if (type === 'wheel' && this.classList?.contains('graph-canvas')) window.graphWheels.delete(listener);
      return removeEventListener.call(this, type, listener, options);
    };
    const request = window.requestAnimationFrame.bind(window);
    const cancel = window.cancelAnimationFrame.bind(window);
    window.requestAnimationFrame = callback => {
      const id = request(time => {window.graphFrames.delete(id); callback(time);});
      if (window.traceFrames) window.graphFrames.add(id);
      return id;
    };
    window.cancelAnimationFrame = id => {
      window.graphFrames.delete(id);
      cancel(id);
    };
    window.ResizeObserver = class extends Observer {
      observe(element, ...options) {
        if (element.matches('.graph-canvas')) window.graphObservers.add(this);
        return super.observe(element, ...options);
      }
      disconnect() {
        window.graphObservers.delete(this);
        return super.disconnect();
      }
    };
  });
  await navigate(page);
  await page.locator('[data-measure-node]').first().waitFor({state: 'attached'});
  assert.equal(await page.locator('[data-node]').count(), 0);
  await page.locator('#visibility').click();
  await graph(page).waitFor({state: 'hidden'});
  releaseFont();
  await page.evaluate(() => document.fonts.ready);
  assert.equal(await page.locator('[data-node]').count(), 0);
  await page.locator('#visibility').click();
  await node(page, 'kv').waitFor();

  const steady = await positions(page);
  const fitted = await transform(page);
  assert.equal(Object.keys(steady).length, 18);
  assert.equal(steady.kv[2], 168);
  assert.match(await page.locator('[data-edge]').first().getAttribute('marker-end'), /^url\(#.+-normal\)$/);

  await node(page, 'kv').focus();
  await page.keyboard.press('Enter');
  await expectText(inspector(page).locator('h2'), 'KV cache');
  assert.match(await node(page, 'kv').getAttribute('class'), /is-selected/);
  assert.ok(await page.locator('.is-neighbor').count() > 0);
  assert.ok(await page.locator('.is-muted').count() > 0);
  assert.ok(await page.locator('[data-edge-label]').count() >= 4);
  assert.deepEqual(await positions(page), steady);
  assert.equal(await transform(page), fitted);

  const predicate = page.getByRole('button', {name: 'Highlight relationship: KV cache stores keys and values for Autoregressive decoding'});
  await predicate.focus();
  await graph(page).locator('.edge-label.is-hovered').filter({hasText: 'stores keys and values for'}).waitFor();
  assert.equal(await page.locator('.is-edge-endpoint').count(), 2);
  await page.keyboard.press('Escape');
  assert.equal(await page.locator('.is-edge-endpoint').count(), 0);
  assert.equal(await node(page, 'kv').getAttribute('aria-pressed'), 'true');
  await page.keyboard.press('ArrowRight');
  assert.ok(await page.locator('[data-node]:focus').count());
  await page.keyboard.press('Escape');
  assert.equal(await graph(page).locator('[data-node][aria-pressed=true]').count(), 0);

  const find = page.getByRole('searchbox', {name: 'Find a concept'}).first();
  await find.fill('cache');
  await find.press('Enter');
  await expectText(inspector(page).locator('h2'), 'KV cache');
  await choice(page, 'Relations').click();
  const table = graph(page).getByRole('table');
  assert.equal(await table.locator('tbody tr').count(), 23);
  assert.ok(await table.locator('tbody tr.is-selected').count() > 0);
  const oriented = table.getByRole('row').filter({hasText: 'stores keys and values for'}).first();
  assert.match(await oriented.locator('td').first().innerText(), /KV cache/);
  assert.match(await oriented.locator('td').last().innerText(), /Autoregressive decoding/);
  await inspector(page).getByRole('button', {name: 'Focus on map'}).click();
  await node(page, 'kv').waitFor();
  await poll(page, () => Number(document.querySelector('.graph-stage').style.transform.match(/scale\(([\d.]+)/)?.[1]) >= .95);
  await page.waitForTimeout(120);
  const focused = await transform(page);
  const bounds = await canvas(page).boundingBox(), card = await node(page, 'kv').boundingBox();
  close(card.x + card.width / 2, bounds.x + bounds.width / 2);
  close(card.y + card.height / 2, bounds.y + bounds.height / 2);
  await inspector(page).getByRole('button', {name: 'Focus on map'}).click();
  await settle(page);
  assert.equal(await transform(page), focused);
  assert.deepEqual(await positions(page), steady);

  const sendBefore = outbound;
  const edgeD = await page.locator('.graph-edge.is-active').first().getAttribute('d');
  const center = [card.x + card.width / 2, card.y + card.height / 2];
  await page.mouse.move(...center);
  await page.mouse.down();
  await page.mouse.move(center[0] + 4, center[1]);
  await page.mouse.up();
  assert.deepEqual(await positions(page), steady);
  await page.mouse.move(...center);
  await page.mouse.down();
  await page.mouse.move(center[0] + 32, center[1] + 22, {steps: 8});
  await page.mouse.up();
  await poll(page, ([x, y]) => {
    const card = document.querySelector('[data-node=kv]');
    return card.style.left !== x && card.style.top !== y;
  }, steady.kv);
  const dragged = await positions(page);
  assert.notEqual(await page.locator('.graph-edge.is-active').first().getAttribute('d'), edgeD);
  assert.equal(await node(page, 'kv').getAttribute('aria-pressed'), 'true');
  assert.deepEqual(dragged.ar, steady.ar);

  const background = [bounds.x + 8, bounds.y + 8];
  await page.mouse.move(...background);
  await page.mouse.down();
  await page.mouse.move(background[0] + 37, background[1] + 25);
  await page.mouse.up();
  assert.notEqual(await transform(page), focused);
  const matrix = () => page.locator('.graph-stage').evaluate(el => {
    const {a, e, f} = new DOMMatrix(el.style.transform);
    return [a, e, f];
  });
  const [a, e, f] = await matrix();
  await page.mouse.move(...background);
  await page.mouse.wheel(0, -100);
  await poll(page, prior => new DOMMatrix(document.querySelector('.graph-stage').style.transform).a > prior, a);
  const [newA, newE, newF] = await matrix();
  close((8 - e) / a, (8 - newE) / newA, .01);
  close((8 - f) / a, (8 - newF) / newA, .01);
  assert.equal(outbound, sendBefore, 'graph interactions emitted Reflex events');

  const beforeUpdate = await transform(page);
  await page.locator('#stream').click();
  await expectText(page.locator('#tick'), '12');
  await page.locator('#theme').click();
  await graph(page).locator('xpath=self::*[@data-graph-theme="light"]').waitFor();
  await page.locator('#visibility').click();
  await graph(page).waitFor({state: 'hidden'});
  await page.locator('#visibility').click();
  await graph(page).waitFor();
  assert.equal(await transform(page), beforeUpdate);
  assert.deepEqual(await positions(page), dragged);
  assert.equal(await node(page, 'kv').getAttribute('aria-pressed'), 'true');
  const filters = graph(page).getByRole('group', {name: 'Visible concept types'}).first();
  await filters.getByRole('button', {name: /Techniques/}).click();
  assert.equal(await node(page, 'kv').count(), 0);
  await expectText(graph(page).locator('.map-counts'), '7 concepts');
  assert.equal(await graph(page).locator('.graph-node.is-muted').count(), 0);
  await filters.getByRole('button', {name: /Techniques/}).click();
  assert.deepEqual(await positions(page), dragged);
  assert.equal(await transform(page), beforeUpdate);
  await node(page, 'kv').focus();
  await page.keyboard.press('Enter');
  await expectText(inspector(page).locator('h2'), 'KV cache');
  await graph(page).getByRole('button', {name: 'Fit graph'}).click();
  assert.notEqual(await transform(page), beforeUpdate);
  await inspector(page).getByRole('button', {name: 'Explain this'}).click();
  await expectText(page.locator('#result'), 'explain kv');
  const cacheSource = inspector(page).locator('.source-note').first();
  assert.equal(await cacheSource.count(), 0);
  assert.equal(await inspector(page).getByText('AI Research / Efficient inference', {exact: true}).count(), 0);
  await page.locator('#reject').click();
  await expectText(page.locator('#result'), 'rejected');

  await inspector(page).getByRole('button', {name: 'Focus on map'}).click();
  await page.waitForTimeout(65);
  await page.mouse.move(...background);
  await page.mouse.wheel(0, 110);
  await settle(page);
  const interrupted = await transform(page);
  await page.waitForTimeout(500);
  assert.equal(await transform(page), interrupted);
  await graph(page).getByRole('button', {name: 'Fit graph'}).click();
  const beforeResizeFocus = await transform(page);
  await inspector(page).getByRole('button', {name: 'Focus on map'}).click();
  await settle(page);
  assert.notEqual(await transform(page), beforeResizeFocus);
  await page.setViewportSize({width: 1180, height: 760});
  await settle(page);
  const resized = await transform(page);
  await page.waitForTimeout(500);
  assert.equal(await transform(page), resized);
  assert.deepEqual(await positions(page), dragged);

  await graph(page).getByRole('button', {name: 'Fit graph'}).click();
  await inspector(page).getByRole('button', {name: 'Focus on map'}).click();
  await page.emulateMedia({reducedMotion: 'reduce'});
  await settle(page);
  const reduced = await transform(page);
  await page.waitForTimeout(100);
  assert.equal(await transform(page), reduced);
  assert.ok(parseFloat(await zoom(page).innerText()) >= 95);

  await graph(page).getByRole('button', {name: 'Fit graph'}).click();
  await page.emulateMedia({reducedMotion: 'no-preference'});
  assert.equal(await page.evaluate(() => window.graphWheels.size), 1);
  await page.evaluate(() => {window.traceFrames = true;});
  await inspector(page).getByRole('button', {name: 'Focus on map'}).click();
  await page.locator('#mount').evaluate(element => element.click());
  await graph(page).waitFor({state: 'detached'});
  await settle(page);
  assert.deepEqual(await page.evaluate(() => [window.graphObservers.size, window.graphWheels.size, window.graphFrames.size]), [0, 0, 0]);
  await page.evaluate(() => {window.traceFrames = false;});
  await page.locator('#mount').click();
  await node(page, 'kv').waitFor();
  assert.equal(await node(page, 'kv').getAttribute('aria-pressed'), 'false');
  assert.equal(await page.evaluate(() => window.graphObservers.size), 1);
  await page.locator('#command').click();
  await expectText(inspector(page).locator('h2'), 'KV cache');
  await inspector(page).getByRole('button', {name: 'Autoregressive decoding'}).click();
  assert.equal(await page.locator('.concept-inspector h2:focus').innerText(), 'Autoregressive decoding');

  await choose(page, 'unusual');
  await node(page, 'first').waitFor();
  assert.equal(await page.locator('[data-node]').count(), 3);
  assert.equal(await page.locator('[data-edge]').count(), 4);
  assert.equal(new Set(await page.locator('[data-edge]').evaluateAll(edges => edges.map(edge => edge.getAttribute('d')))).size, 4);
  assert.equal(await graph(page).locator('img, script, .concept-description strong').count(), 0);
  await node(page, '__proto__').focus();
  await page.keyboard.press('Enter');
  await expectText(inspector(page).locator('.concept-description'), '<strong>Stored markup</strong>');
  assert.ok(await node(page, '__proto__').evaluate(element => element.offsetHeight < 145));
  await node(page, 'first').focus();
  await page.keyboard.press('Enter');
  await inspector(page).getByText('Summary unavailable', {exact: true}).waitFor();
  assert.equal(await inspector(page).getByText('Untitled source path', {exact: true}).count(), 0);
  await inspector(page).locator('summary').getByText('External reference', {exact: true}).click();
  const external = inspector(page).getByRole('link', {name: 'Open External reference'});
  assert.equal(await external.getAttribute('href'), 'https://example.org/reference');
  assert.equal(await external.getAttribute('rel'), 'noopener noreferrer');
  await choice(page, 'Relations').click();
  assert.equal(await table.locator('tbody tr').count(), 4);
  await expectText(table, 'Synthetic');
  await choose(page, 'empty');
  await expectText(graph(page).locator('.graph-unavailable'), 'There are no concepts');
  assert.equal(await choice(page, 'Map').isDisabled(), true);
  await page.close();
  console.log('Desktop font/visibility, relationships, navigation, drag, wheel, camera, callbacks and lifecycle passed');
}

async function checkMobile(browser, errors) {
  const page = await browser.newPage({viewport: {width: 390, height: 844}, isMobile: true, hasTouch: true, deviceScaleFactor: 1});
  page.on('pageerror', error => errors.push(String(error)));
  await navigate(page);
  await node(page, 'kv').waitFor();
  assert.equal(await zoom(page).innerText(), '90%');
  await node(page, 'kv').tap();
  const peek = graph(page).locator('.inspector-peek');
  await peek.tap();
  const details = page.getByRole('dialog', {name: 'Selected concept'});
  await expectText(details.locator('h2'), 'KV cache');
  await page.keyboard.press('Escape');
  await details.waitFor({state: 'hidden'});
  assert.equal(await peek.evaluate(el => el === document.activeElement), true);
  await choice(page, 'Relations').tap();
  await peek.tap();
  await details.getByRole('button', {name: 'Focus on map'}).tap();
  await details.waitFor({state: 'hidden'});
  await poll(page, () => Number(document.querySelector('.graph-stage').style.transform.match(/scale\(([\d.]+)/)?.[1]) >= .95);
  const bounds = await canvas(page).boundingBox(), card = await node(page, 'kv').boundingBox();
  close(card.x + card.width / 2, bounds.x + bounds.width / 2);
  await graph(page).getByRole('button', {name: 'Fit graph'}).tap();
  await node(page, 'ar').tap();
  await expectText(peek, 'Autoregressive decoding');
  const ar = await node(page, 'ar').boundingBox(), fittedBounds = await canvas(page).boundingBox();
  close(ar.x + ar.width / 2, fittedBounds.x + fittedBounds.width / 2);
  close(ar.y + ar.height / 2, fittedBounds.y + fittedBounds.height / 2);
  await graph(page).getByRole('button', {name: 'Fit graph'}).tap();
  const hit = await page.locator('[data-edge-hit]').evaluateAll(paths => {
    for (const path of paths) {
      const point = path.getPointAtLength(path.getTotalLength() / 2).matrixTransform(path.getScreenCTM());
      if (document.elementFromPoint(point.x, point.y) === path) return {x: point.x, y: point.y};
    }
    return null;
  });
  assert.ok(hit, 'no visible touch relationship target');
  await page.touchscreen.tap(hit.x, hit.y);
  await graph(page).locator('.graph-edge.is-hovered').waitFor({state: 'attached'});
  await graph(page).locator('.canvas-caption').tap();
  assert.equal(await graph(page).locator('.graph-edge.is-hovered').count(), 0);
  const filters = graph(page).getByRole('button', {name: 'Open graph filters'});
  await filters.tap();
  const tools = page.getByRole('dialog', {name: 'Graph controls'});
  await tools.getByRole('searchbox').fill('PagedAttention');
  await tools.getByRole('searchbox').press('Enter');
  await tools.waitFor({state: 'hidden'});
  await expectText(peek, 'PagedAttention');
  await filters.tap();
  await tools.waitFor();
  await page.touchscreen.tap(9, 270);
  await tools.waitFor({state: 'hidden'});
  assert.equal(await filters.evaluate(el => el === document.activeElement), true);
  await page.emulateMedia({reducedMotion: 'reduce'});
  await peek.tap();
  await details.getByRole('button', {name: 'Focus on map'}).tap();
  assert.equal(await graph(page).locator('.graph-node').first().evaluate(el => getComputedStyle(el).transitionDuration), '0s');
  assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true);
  await choose(page, 'unusual');
  await node(page, '__proto__').waitFor();
  await node(page, '__proto__').focus();
  await page.keyboard.press('Enter');
  await expectText(peek, '超長いラベル');
  assert.ok((await canvas(page).boundingBox()).height > 450, 'long label crowded out the map');
  await page.close();
  console.log('Mobile touch selection, drawer focus, Relations-to-Map, Find and reduced motion passed');
}

async function performanceCheck(browser, errors) {
  const page = await browser.newPage({viewport: {width: 1440, height: 900}});
  page.on('pageerror', error => errors.push(String(error)));
  await navigate(page);
  await node(page, 'kv').waitFor();
  const observations = {};
  for (const [name, count, edgeCount] of [['normal', 25, 35], ['combined', 33, 38], ['ceiling', 48, 96]]) {
    await page.evaluate(() => {
      window.afterFixture = new Promise(resolve => {
        const observer = new MutationObserver(() => {
          if (!document.querySelector('[data-node]')) return;
          if (document.querySelector('[data-measure-node]')) return;
          observer.disconnect();
          requestAnimationFrame(() => resolve(performance.now()));
        });
        observer.observe(document.querySelector('#stage4-host'), {childList: true, subtree: true});
      });
      window.beforeFixture = performance.now();
    });
    await choose(page, name);
    await poll(page, expected => document.querySelectorAll('[data-node]').length === expected, count);
    const elapsed = await page.evaluate(async () => await window.afterFixture - window.beforeFixture);
    const edges = await page.locator('[data-edge]').count();
    assert.equal(edges, edgeCount);
    const moving = node(page, name === 'combined' ? 'kv' : 'n11');
    const leftBefore = await moving.evaluate(el => el.style.left);
    const target = await moving.boundingBox();
    const x = target.x + target.width / 2, y = target.y + target.height / 2;
    await page.mouse.move(x, y);
    await page.mouse.down();
    const frames = page.evaluate(() => new Promise(resolve => {
      const times = [];
      function next(now) {
        times.push(now);
        if (times.length < 25) requestAnimationFrame(next);
        else resolve(times.slice(1).map((t, i) => t - times[i]));
      }
      requestAnimationFrame(next);
    }));
    await page.mouse.move(x + 60, y + 35, {steps: 20});
    await page.mouse.up();
    assert.notEqual(await moving.evaluate(el => el.style.left), leftBefore, `${name} drag did not move the node`);
    const gaps = (await frames).sort((a, b) => a - b);
    observations[name] = {nodes: count, edges, fixtureToPaintMs: +elapsed.toFixed(1),
      dragFrameP95Ms: +gaps[Math.ceil(.95 * gaps.length) - 1].toFixed(1), dragFrameMaxMs: +gaps.at(-1).toFixed(1)};
  }
  for (const [name, count] of [['oversized', 97], ['max', 2048]]) {
    await choose(page, name);
    await poll(page, expected => document.querySelectorAll('.relations-view tbody tr').length === expected, count);
    assert.equal(await page.locator('[data-node], [data-measure-node]').count(), 0);
    assert.equal(await choice(page, 'Map').isDisabled(), true);
  }
  await page.close();
  return observations;
}

async function images(browser, errors) {
  if (!destination || !reference) return;
  for (const [mode, viewport] of [['desktop', {width: 1440, height: 900}], ['mobile', {width: 390, height: 844}]]) {
    for (const theme of ['dark', 'light']) {
      const mobile = mode === 'mobile';
      const context = await browser.newContext({viewport, hasTouch: mobile, isMobile: mobile, deviceScaleFactor: 1, reducedMotion: 'reduce'});
      const page = await context.newPage();
      page.on('pageerror', error => errors.push(String(error)));
      await navigate(page);
      await node(page, 'kv').waitFor();
      if (theme === 'light') {
        await page.locator('#theme').click();
        await page.locator('[data-graph-theme=light]').waitFor();
      }
      const ref = await context.newPage();
      await ref.goto(`${reference}?theme=${theme}#graph`);
      await ref.waitForFunction(() => document.querySelector('.graph-stage').style.transform);
      await ref.evaluate(() => document.fonts.ready);
      const paired = async state => {
        const drawer = mobile && state === 'selected';
        await (drawer ? page : graph(page)).screenshot({path: join(destination, `app-${mode}-${theme}-${state}.png`)});
        await (drawer ? ref : ref.locator('.graph-workspace')).screenshot({path: join(destination, `reference-${mode}-${theme}-${state}.png`)});
      };
      await paired('map');
      await node(page, 'kv').click();
      await ref.locator('[data-node=kv]').click();
      if (mobile) {
        await page.locator('.inspector-peek').tap();
        await ref.locator('[data-open-inspector]').tap();
      }
      await paired('selected');
      if (mobile) {
        await page.keyboard.press('Escape');
        await ref.keyboard.press('Escape');
      }
      await choice(page, 'Relations').click();
      await ref.locator('[data-graph-view=relations]').click();
      await paired('relations');
      await context.close();
    }
  }
}

(async () => {
  const browser = await chromium.launch({headless: true});
  const errors = [];
  try {
    if (destination) mkdirSync(destination, {recursive: true});
    await checkDesktop(browser, errors);
    await checkMobile(browser, errors);
    const timings = await performanceCheck(browser, errors);
    const report = {chromium: browser.version(), node: process.version, timings};
    console.log(JSON.stringify(report));
    if (destination) writeFileSync(join(destination, 'performance.json'), JSON.stringify(report, null, 2) + '\n');
    await images(browser, errors);
    assert.deepEqual(errors, []);
  } finally {
    await browser.close();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
