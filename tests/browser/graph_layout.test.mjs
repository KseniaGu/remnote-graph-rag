import assert from 'node:assert/strict';
import test from 'node:test';
import {normalizeGraph, placeGraph, routeEdges, placeLabels, safeSourceURL} from '../../app/ui/graph_layout.mjs';

const nodes = [
  {id: 'a', label: 'β cache', width: 192, height: 102},
  {id: 'b', label: 'β cache', width: 192, height: 150},
  {id: 'c', label: 'Attention', width: 192, height: 102},
  {id: '__proto__', label: 'Isolated', width: 192, height: 102},
];
const edge = (id, source, target, predicate, synthetic = false) => ({id, source, target, predicate, synthetic});
const edges = [
  edge('ab', 'a', 'b', 'uses'),
  edge('ab2', 'a', 'b', 'contains', true),
  edge('ba', 'b', 'a', 'feeds'),
  edge('bc', 'b', 'c', 'queries'),
  edge('aa', 'a', 'a', 'reuses'),
];

test('Balanced reproduces approved shortest-path geometry with measured sizes', () => {
  const actual = placeGraph(nodes, edges);
  // Expected geometry captured from the approved Balanced layout for this fixture.
  assert.deepEqual({...actual.positions}, {
    ['__proto__']: [661.4889509822356, 239.1638527029099],
    a: [152.3785023167281, 385.29155810872976],
    b: [138, 239.2],
    c: [152.3785023167281, 93],
  });
  assert.equal(actual.width, 799.4889509822356);
  assert.equal(actual.height, 478.29155810872976);
  assert.equal(actual.root, 'b');
  assert.deepEqual(placeGraph([...nodes].reverse(), [...edges].reverse()), actual);
  for (const [index, a] of nodes.entries()) {
    for (const b of nodes.slice(index + 1)) {
      const [ax, ay] = actual.positions[a.id], [bx, by] = actual.positions[b.id];
      assert.ok(Math.abs(ax - bx) >= (a.width + b.width) / 2 || Math.abs(ay - by) >= (a.height + b.height) / 2);
    }
  }
});

test('payload preserves IDs, direction, synthetic edges and incomplete metadata', () => {
  const graph = normalizeGraph({
    schema_version: 1, artifact_id: 'map', nodes,
    edges: [...edges, edge('broken', 'a', 'missing', 'unknown')],
  });
  assert.equal(graph.nodes.length, 4);
  assert.deepEqual(graph.edges.map(({id, source, target, synthetic}) => ({id, source, target, synthetic})),
    edges.map(({id, source, target, synthetic}) => ({id, source, target, synthetic})));
  assert.equal(graph.nodes[0].family, 'unknown');
  assert.equal(graph.incomplete, true);
  assert.equal(graph.title, 'Untitled map');
  assert.equal(normalizeGraph({schema_version: 1, artifact_id: 'map', title: ' Stored title ', nodes, edges}).title, 'Stored title');
  assert.equal(normalizeGraph({schema_version: 77, nodes, edges}).nodes.length, 0);
  assert.equal(normalizeGraph(null).nodes.length, 0);
});

test('parallel, reverse and self-loop paths are directed and distinguishable', () => {
  const rects = new Map([['a', {x: 200, y: 220, w: 168, h: 52}], ['b', {x: 600, y: 230, w: 168, h: 72}]]);
  const routes = routeEdges(edges.filter(e => rects.has(e.target)), rects);
  assert.equal(new Set(Array.from(routes.values(), path => path.d)).size, 4);
  assert.notDeepEqual(routes.get('ab').at(.5), routes.get('ab2').at(.5));
  assert.ok(routes.get('ab').at(0).x < routes.get('ab').at(1).x);
  assert.ok(routes.get('ba').at(0).x > routes.get('ba').at(1).x);
  assert.notDeepEqual(routes.get('aa').at(0), routes.get('aa').at(1));
  const active = ['ab2', 'ab', 'ba', 'aa'];
  const sizes = new Map(active.map(id => [id, {w: 70, h: 24}]));
  const labels = placeLabels(active, 'ab2', sizes, routes, rects, {width: 900, height: 600});
  assert.ok(labels.has('ab2'));
  for (const [index, [id, a]] of [...labels].entries()) {
    assert.ok(a.x >= 39 && a.x <= 861 && a.y >= 16 && a.y <= 584, id);
    for (const [, b] of [...labels].slice(index + 1)) {
      assert.ok(Math.abs(a.x - b.x) >= 70 || Math.abs(a.y - b.y) >= 24);
    }
  }
});

test('stored source URLs accept only ordinary HTTP(S) links', () => {
  assert.equal(safeSourceURL('https://example.org/notes'), 'https://example.org/notes');
  for (const url of ['javascript:alert(1)', 'data:text/html,hello', '//example.org',
    'https://user:pass@example.org', 'https://@example.org', 'https://example.org:0', 'https:\n//example.org']) {
    assert.equal(safeSourceURL(url), null, url);
  }
});
