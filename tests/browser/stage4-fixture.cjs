const content = require('./fixtures/graph-topology.json');

const type = {idea: 'CONCEPT', technique: 'METHOD', measure: 'METRIC', artifact: 'TOOL'};
const associations = {kv: [0, 1, 3], gqa: [1, 2], mqa: [1, 2], mha: [0, 1], sdpa: [1], flash: [3],
  gqa_paper: [2], llama2: [2], ar: [0, 3], prefill: [0], decode: [0, 3], seqlen: [0], tpot: [3],
  membw: [3], kvquant: [0, 3], batch: [3], paged: [0, 3], vllm: [3]};
const descriptions = {kv: 'Keys and values from earlier tokens, stored once and reused during decoding.',
  gqa: 'Groups of query heads share key and value heads, reducing the cache footprint.',
  paged: 'Fixed-size cache blocks allocated on demand to reduce memory fragmentation.',
  decode: 'The token-by-token phase that reads the cached prefix.'};
const uuid = n => `00000000-0000-4000-8000-${String(n).padStart(12, '0')}`;
const edge = (n, source, predicate, target, synthetic = false) => ({id: uuid(10000 + n), source, target, predicate, synthetic});

function graph(index, title, nodes, edges) {
  return {artifact_id: uuid(index), schema_version: 1, title, nodes, edges};
}
const node = n => ({id: n.id, label: n.label, type: type[n.family], description: descriptions[n.id] ?? null,
  source_ids: (associations[n.id] ?? []).map(index => `note:${index}`)});
const kv = graph(1, 'KV caching', content.kvGraph.nodes.map(node),
  content.kvGraph.edges.map(({source, label, target}, i) => edge(i, source, label, target)));
const combined = graph(2, 'Connected research',
  [...new Map([...content.kvGraph.nodes, ...content.overviewGraph.nodes].map(n => [n.id, node(n)])).values()],
  [...content.kvGraph.edges, ...content.overviewGraph.edges].map(({source, label, target}, i) => edge(i, source, label, target)));
const unusual = graph(3, 'Edge cases', [
  {id: 'first', label: 'β Cache', type: 'COMPONENT', source_ids: ['web:valid', 'note:untitled']},
  {id: 'second', label: 'β Cache', type: 'CONCEPT', description: 'Distinct identity'},
  {id: '__proto__', label: '超長いラベル'.repeat(24) + ' <img src=x onerror=alert(1)>', description: '<strong>Stored markup</strong>'},
], [edge(0, 'first', 'uses', 'second'), edge(1, 'first', 'shares with', 'second', true),
  edge(2, 'second', 'reduces', 'first'), edge(3, 'first', 'loops', 'first')]);
function sized(index, count, edgeCount) {
  const nodes = Array.from({length: count}, (_, i) => ({id: `n${i}`, label: `Concept ${i}`, type: 'COMPONENT'}));
  const edges = Array.from({length: edgeCount}, (_, i) => {
    const source = i < count - 1 ? i : (i * 7) % count;
    const target = i < count - 1 ? i + 1 : (source + 3 + Math.floor(i / count)) % count;
    return edge(i, nodes[source].id, `connects ${i}`, nodes[target].id);
  });
  return graph(index, `${count} concept map`, nodes, edges);
}
const sources = [...content.sources.map((source, index) => ({
  kind: 'note', source_id: `note:${index}`, title: source.title, path: source.path.split(' / '),
})), {kind: 'note', source_id: 'note:untitled', title: null, path: ['Untitled source path']}, {kind: 'web', source_id: 'web:valid', title: 'External reference', url: 'https://example.org/reference'}];

process.stdout.write(JSON.stringify({graphs: {kv, combined, unusual, normal: sized(4, 25, 35),
  ceiling: sized(5, 48, 96), oversized: sized(6, 49, 97), max: sized(7, 512, 2048)}, sources}));
