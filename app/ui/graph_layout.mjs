// Balanced shortest-path stress and measured-card separation from the approved graph-layouts prototype.
function topology(nodes, edges) {
  const adjacency = new Map(nodes.map(n => [n.id, new Set()]));
  for (const e of edges) if (adjacency.has(e.source) && adjacency.has(e.target) && e.source !== e.target) {
    adjacency.get(e.source).add(e.target); adjacency.get(e.target).add(e.source);
  }
  const root = [...nodes].sort((a, b) =>
    adjacency.get(b.id).size - adjacency.get(a.id).size || a.id.localeCompare(b.id))[0]?.id;
  const distances = nodes.map(n => {
    const found = new Map([[n.id, 0]]), queue = [n.id];
    for (let i = 0; i < queue.length; i++) for (const id of [...adjacency.get(queue[i])].sort()) {
      if (!found.has(id)) { found.set(id, found.get(queue[i]) + 1); queue.push(id); }
    }
    return nodes.map(other => found.get(other.id) ?? Math.max(4, Math.sqrt(nodes.length) * 2));
  });
  return {root, distances};
}

function separate(points, nodes, gap = 20) {
  for (let pass = 0; pass < 220; pass++) {
    let moved = false;
    for (let i = 0; i < points.length; i++) for (let j = i + 1; j < points.length; j++) {
      const a = points[i], b = points[j], dx = b[0] - a[0], dy = b[1] - a[1];
      const overlapX = (nodes[i].width + nodes[j].width) / 2 + gap - Math.abs(dx);
      const overlapY = (nodes[i].height + nodes[j].height) / 2 + gap - Math.abs(dy);
      if (overlapX <= 0 || overlapY <= 0) continue;
      moved = true;
      const axis = overlapX < overlapY * 1.5 ? 0 : 1;
      const delta = (axis ? overlapY : overlapX) / 2 + .1;
      const sign = (axis ? dy : dx) >= 0 ? 1 : -1;
      a[axis] -= sign * delta; b[axis] += sign * delta;
    }
    if (!moved) break;
  }
}

function balanced(nodes, distances) {
  const count = nodes.length;
  if (count < 2) return nodes.map(() => [0, 0]);
  const squared = distances.map(row => row.map(d => d * d));
  const means = squared.map(row => row.reduce((a, b) => a + b, 0) / count);
  const mean = means.reduce((a, b) => a + b, 0) / count;
  const matrix = squared.map((row, i) => row.map((d, j) => -.5 * (d - means[i] - means[j] + mean)));
  const vectors = [];
  for (let axis = 0; axis < 2; axis++) {
    let vector = nodes.map((_, i) => Math.sin((i + 1) * (axis + 1) * 1.31));
    for (let k = 0; k < 180; k++) {
      let next = matrix.map(row => row.reduce((sum, v, j) => sum + v * vector[j], 0));
      if (axis) {
        const dot = next.reduce((s, v, i) => s + v * vectors[0][i], 0);
        next = next.map((v, i) => v - dot * vectors[0][i]);
      }
      const length = Math.hypot(...next) || 1;
      vector = next.map(v => v / length);
    }
    vectors.push(vector);
  }
  const points = nodes.map((_, i) => [vectors[0][i] * count * 65, vectors[1][i] * count * 65]);
  for (let tick = 0; tick < 450; tick++) {
    const delta = nodes.map(() => [0, 0]);
    for (let i = 0; i < count; i++) for (let j = i + 1; j < count; j++) {
      const dx = points[j][0] - points[i][0], dy = points[j][1] - points[i][1];
      const d = Math.max(.1, Math.hypot(dx, dy)), target = distances[i][j] * 125;
      const f = (d - target) / (d * distances[i][j] ** 2) * .07;
      delta[i][0] += dx * f; delta[i][1] += dy * f;
      delta[j][0] -= dx * f; delta[j][1] -= dy * f;
    }
    points.forEach((p, i) => { p[0] += delta[i][0]; p[1] += delta[i][1]; });
  }
  return points.map(([x, y]) => [x * 1.05, y * 1.1]);
}

export function placeGraph(inputNodes, edges) {
  const nodes = inputNodes.map(n => ({...n, width: n.width || 156, height: n.height || 60})).sort((a, b) => a.id.localeCompare(b.id));
  const {root, distances} = topology(nodes, edges);
  const points = balanced(nodes, distances);
  separate(points, nodes);
  const left = Math.min(0, ...points.map((p, i) => p[0] - nodes[i].width / 2)) - 42;
  const top = Math.min(0, ...points.map((p, i) => p[1] - nodes[i].height / 2)) - 42;
  const width = Math.max(100, ...points.map((p, i) => p[0] + nodes[i].width / 2 - left)) + 42;
  const height = Math.max(100, ...points.map((p, i) => p[1] + nodes[i].height / 2 - top)) + 42;
  const positions = Object.fromEntries(nodes.map((n, i) => [n.id, [points[i][0] - left, points[i][1] - top]]));
  return {positions, width, height, root};
}

export const families = {
  idea: 'Ideas', technique: 'Techniques', measure: 'Measures',
  artifact: 'Artifacts', unknown: 'Unknown type',
};
const typeFamilies = new Map([
  ['CONCEPT', 'idea'], ['TASK', 'idea'], ['PROBLEM', 'idea'],
  ['METHOD', 'technique'], ['MODEL', 'technique'], ['COMPONENT', 'technique'],
  ['FORMULA', 'measure'], ['PARAMETER', 'measure'], ['METRIC', 'measure'],
  ['DATASET', 'artifact'], ['TOOL', 'artifact'], ['PAPER', 'artifact'],
]);

// The Map ceiling bounds synchronous layout and obstacle routing. Relations supports
// the full Stage 2b payload, including graphs too large to lay out interactively.
export const mapLimits = {nodes: 48, edges: 96};
const payloadLimits = {nodes: 512, edges: 2048};
const isText = value => typeof value === 'string' && Boolean(value.trim());
const refs = value => Array.isArray(value) ? [...new Set(value.filter(isText))] : [];

export function normalizeGraph(payload) {
  const empty = {id: '', title: 'Untitled map', nodes: [], edges: [], incomplete: false};
  if (payload?.schema_version !== 1 || !isText(payload.artifact_id) ||
      !Array.isArray(payload.nodes) || !Array.isArray(payload.edges)) return empty;
  if (payload.nodes.length > payloadLimits.nodes || payload.edges.length > payloadLimits.edges) {
    return {...empty, status: 'This graph exceeds the supported data size.'};
  }

  const nodes = new Map();
  let incomplete = false;
  for (const node of payload.nodes) {
    if (!isText(node?.id) || !isText(node.label) || nodes.has(node.id)) {
      incomplete = true;
      continue;
    }
    const type = isText(node.type) ? node.type : null;
    nodes.set(node.id, {
      id: node.id, label: node.label, type, family: typeFamilies.get(type?.trim().toUpperCase()) ?? 'unknown',
      description: isText(node.description) ? node.description : null, sourceIds: refs(node.source_ids),
    });
  }

  const edges = new Map();
  for (const edge of payload.edges) {
    if (!isText(edge?.id) || edges.has(edge.id) || !nodes.has(edge.source) || !nodes.has(edge.target)) {
      incomplete = true;
      continue;
    }
    if (!isText(edge.predicate)) incomplete = true;
    edges.set(edge.id, {
      id: edge.id, source: edge.source, target: edge.target,
      label: isText(edge.predicate) ? edge.predicate : 'Relationship unavailable',
      synthetic: edge.synthetic === true, relationId: isText(edge.relation_id) ? edge.relation_id : null,
      evidenceSourceIds: refs(edge.evidence_source_ids),
    });
  }
  return {
    id: payload.artifact_id, title: isText(payload.title) ? payload.title.trim() : empty.title,
    nodes: [...nodes.values()], edges: [...edges.values()], incomplete,
  };
}

export function safeSourceURL(value) {
  if (typeof value !== 'string' || value.length > 2048 || !/^https?:\/\/[^/@?#]+(?:[/?#]|$)/i.test(value) ||
    /[\s\u0000-\u001f<>"\\]/u.test(value)) return null;
  try {
    const url = new URL(value);
    return url.port === '0' ? null : url.href;
  } catch {
    return null;
  }
}

function port(card, point) {
  const dx = point.x - card.x, dy = point.y - card.y;
  const scale = Math.min((card.w / 2 + 3) / (Math.abs(dx) || .00001),
    (card.h / 2 + 3) / (Math.abs(dy) || .00001));
  return {x: card.x + dx * scale, y: card.y + dy * scale};
}

function quadratic(start, control, end) {
  return {
    d: `M${start.x} ${start.y}Q${control.x} ${control.y} ${end.x} ${end.y}`,
    at: t => ({
      x: (1 - t) ** 2 * start.x + 2 * (1 - t) * t * control.x + t * t * end.x,
      y: (1 - t) ** 2 * start.y + 2 * (1 - t) * t * control.y + t * t * end.y,
    }),
  };
}

function routeEdge(a, b, obstacles, lane) {
  const dx = b.x - a.x, dy = b.y - a.y, length = Math.hypot(dx, dy);
  if (!length) return null;
  let bestCost = Infinity, best;
  for (const bend of [0, 35, -35, 75, -75, 140, -140]) {
    const offset = lane + bend;
    const control = {
      x: (a.x + b.x) / 2 - dy / length * offset,
      y: (a.y + b.y) / 2 + dx / length * offset,
    };
    const start = port(a, control), end = port(b, control);
    const route = quadratic(start, control, end);
    const left = Math.min(start.x, control.x, end.x), right = Math.max(start.x, control.x, end.x);
    const top = Math.min(start.y, control.y, end.y), bottom = Math.max(start.y, control.y, end.y);
    const nearby = obstacles.filter(n =>
      n.x + n.w / 2 + 7 > left && n.x - n.w / 2 - 7 < right &&
      n.y + n.h / 2 + 7 > top && n.y - n.h / 2 - 7 < bottom);
    let cost = Math.abs(bend) * .02;
    for (let t = .08; nearby.length && t < 1 && cost < bestCost; t += .06) {
      const p = route.at(t);
      for (const n of nearby) {
        if (Math.abs(p.x - n.x) < n.w / 2 + 7 && Math.abs(p.y - n.y) < n.h / 2 + 7) cost += 100;
      }
    }
    if (cost < bestCost) {
      best = route;
      bestCost = cost;
    }
  }
  return best;
}

export function routeEdges(edges, rects) {
  const groups = new Map();
  for (const edge of edges) {
    if (!rects.has(edge.source) || !rects.has(edge.target)) continue;
    const key = JSON.stringify([edge.source, edge.target].sort());
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(edge);
  }
  const routes = new Map();
  for (const group of groups.values()) {
    group.sort((a, b) => a.id.localeCompare(b.id));
    group.forEach((edge, index) => {
      const a = rects.get(edge.source), b = rects.get(edge.target);
      let route;
      if (edge.source === edge.target) {
        const y = a.y - a.h / 2 - 3;
        route = quadratic({x: a.x - 22, y}, {x: a.x, y: y - 55 - index * 20}, {x: a.x + 22, y});
      } else {
        const obstacles = Array.from(rects.values()).filter(rect => rect !== a && rect !== b);
        const lane = (index - (group.length - 1) / 2) * 24 * (edge.source < edge.target ? 1 : -1);
        route = routeEdge(a, b, obstacles, lane);
      }
      if (route) routes.set(edge.id, route);
    });
  }
  return routes;
}

function box(x, y, w, h, margin = 0) {
  return {left: x - w / 2 - margin, right: x + w / 2 + margin,
    top: y - h / 2 - margin, bottom: y + h / 2 + margin};
}

export function placeLabels(ids, highlighted, sizes, routes, rects, world) {
  const used = Array.from(rects.values(), r => box(r.x, r.y, r.w, r.h, 7));
  const result = new Map();
  for (const id of ids) {
    const route = routes.get(id), size = sizes.get(id);
    if (!route || !size) continue;
    const active = id === highlighted;
    const times = active ? [.5, .35, .65, .22, .78, .12, .88] : [.5, .35, .65, .22, .78];
    const offsets = [[0, -12], [0, 14], [-24, -28], [24, 28], [-45, -12], [45, 14], [0, -42], [0, 42]];
    if (active) offsets.push([0, -58], [0, 58], [-48, -52], [48, 52], [-48, 52], [48, -52],
      [-72, -20], [72, 20], [-72, 20], [72, -20], [0, -78], [0, 78]);
    const candidates = times.flatMap(t => {
      const p = route.at(t);
      return offsets.map(([dx, dy]) => ({x: p.x + dx, y: p.y + dy}));
    });
    const match = candidates.find(({x, y}) => {
      const bounds = box(x, y, size.w, size.h);
      if (bounds.left < 4 || bounds.right > world.width - 4 ||
          bounds.top < 4 || bounds.bottom > world.height - 4) return false;
      if (used.some(r => bounds.left < r.right && bounds.right > r.left &&
          bounds.top < r.bottom && bounds.bottom > r.top)) return false;
      used.push(bounds);
      return true;
    });
    if (match) result.set(id, match);
  }
  return result;
}
