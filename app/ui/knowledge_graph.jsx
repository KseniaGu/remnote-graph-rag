import React, {useCallback, useEffect, useId, useLayoutEffect, useMemo, useRef, useState} from 'react';
import {Tooltip} from '@radix-ui/themes';
import {families, mapLimits, normalizeGraph, placeGraph, placeLabels, routeEdges, safeSourceURL} from './graph_layout.mjs';
import {fittedCamera, mobileQuery, useGraphCamera} from './graph_camera.mjs';

const icons = {
  search: <><circle cx="7" cy="7" r="4.5"/><path d="m10.5 10.5 3 3"/></>,
  filter: <><path d="M2 4h2m4 0h6M2 12h6m4 0h2"/><circle cx="6" cy="4" r="2"/><circle cx="10" cy="12" r="2"/></>,
  chevron: <path d="m6 4 4 4-4 4"/>,
  arrow: <path d="M3 8h10M9 4l4 4-4 4"/>,
  close: <path d="m4 4 8 8m0-8-8 8"/>,
  fit: <path d="M6 2H2v4M10 2h4v4M2 10v4h4M10 14h4v-4"/>,
  map: <><path d="m4.5 5 6-1M5 6l2.5 5M11 6l-2.5 5"/><circle cx="3.5" cy="4.5" r="2"/><path d="m12 2 2.5 2.5L12 7 9.5 4.5Z"/><rect x="6" y="11" width="4" height="4" rx=".5"/></>,
  note: <path d="M4 2h5l3 3v9H4zM6 8h4m-4 3h4"/>,
  globe: <><circle cx="8" cy="8" r="6"/><ellipse cx="8" cy="8" rx="2.5" ry="6"/><path d="M2 8h12"/></>,
  trash: <path d="M2 4h12M6 4V2h4v2M4 4l.8 10h6.4L12 4M6.5 7v4m3-4v4"/>,
};
function Icon({name, className}) {
  return <svg viewBox="0 0 16 16" className={className} aria-hidden="true">{icons[name]}</svg>;
}

function Family({name}) {
  return <span className={`family-shape ${name}`} aria-hidden="true"/>;
}

function Drawer({dialog, label, className, children}) {
  return <dialog ref={dialog} className={`graph-drawer ${className}`} aria-label={label}
    onClick={event => {
      if (event.target !== event.currentTarget) return;
      const {left, right, top, bottom} = event.currentTarget.getBoundingClientRect();
      if (event.clientX < left || event.clientX > right || event.clientY < top || event.clientY > bottom) {
        event.currentTarget.close();
      }
    }}>{children}</dialog>;
}

function GraphSource({source}) {
  const note = source.kind === 'note';
  const summary = source.summary?.trim();
  const title = source.title?.trim();
  const label = title || (note ? summary || 'Summary unavailable' : 'Untitled web source');
  const url = safeSourceURL(source.url);
  const icon = <Icon name={note ? 'note' : 'globe'}/>;
  if (!(note ? summary : source.excerpt) && !url) return <li>
    <div className="source-static">{icon}<span className="source-label">{label}</span></div>
  </li>;
  return <li>
    <details className={note ? 'source-note' : undefined}>
      <summary>
        {icon}
        <span className="source-label"><span className={note && !title && summary ? 'source-preview' : undefined}>{label}</span></span>
        {note && <span className="source-summary-label">Summary</span>}
        <Icon name="chevron" className="disclosure-chevron"/>
      </summary>
      <div className="source-body">
        {note ? <p>{summary || 'Summary unavailable.'}</p> : source.excerpt && <p>{source.excerpt}</p>}
        {url && <a href={url} target="_blank" rel="noopener noreferrer" aria-label={title ? `Open ${title}` : undefined}>Open source</a>}
      </div>
    </details>
  </li>;
}

export function SourceList({title, children, className}) {
  const [expanded, setExpanded] = useState(false);
  const listId = useId();
  const rows = React.Children.toArray(children);
  const previewLimit = 5;
  return <section className={className}>
    <div className="source-section-heading">
      <h3>{title}</h3>
      {rows.length > previewLimit && <button type="button" className="source-list-toggle"
        aria-expanded={expanded} aria-controls={listId} onClick={() => setExpanded(!expanded)}>
        {expanded ? 'Show fewer' : `Show all (${rows.length})`}
      </button>}
    </div>
    {rows.length > 0 && <ul id={listId} className="source-list" role="list">
      {expanded ? rows : rows.slice(0, previewLimit)}
    </ul>}
  </section>;
}

function Inspector({node, nodes, edges, mostConnected, select, clear, focusMap, focusAvailable,
  sources, prepare, close, requestHeading}) {
  const heading = useRef(null);
  useLayoutEffect(() => {
    if (requestHeading.current && heading.current?.getClientRects().length) {
      heading.current.focus({preventScroll: true});
      requestHeading.current = false;
    }
  }, [node, requestHeading]);

  const follow = id => {
    requestHeading.current = id !== node?.id;
    select(id, true);
  };
  const closeButton = close && <button className="icon-button mobile-only" type="button" onClick={close}
    aria-label={node ? 'Close concept details' : 'Close map overview'}>
    <Icon name={node ? "chevron" : "close"}/>
  </button>;

  if (!node) return <div className="inspector-content">
    <div className="inspector-heading overview-heading"><h2 ref={heading} tabIndex={-1}>Map overview</h2>{closeButton}</div>
    <p className="concept-description">Select a concept to see available sources and explore its relationships.</p>
    {mostConnected.length > 0 && <><h3>Most connected</h3><div className="connected-list">
      {mostConnected.map(concept => <button type="button" key={concept.id} onClick={() => follow(concept.id)}>
        <Family name={concept.family}/>{concept.label}<Icon name="arrow"/>
      </button>)}
    </div></>}
  </div>;

  const records = node.sourceIds.map(id => sources.get(id)).filter(Boolean);
  return <div className="inspector-content">
    <div className="inspector-heading">
      <span><Family name={node.family}/>{families[node.family]}{node.type && ` · ${node.type}`}</span>
      <button className="icon-button" type="button" onClick={clear} aria-label="Clear concept selection"><Icon name="close"/></button>
      {closeButton}
    </div>
    <h2 ref={heading} tabIndex={-1}>{node.label}</h2>
    {node.description && <p className="concept-description">{node.description}</p>}
    {(prepare || focusAvailable) && <div className="concept-actions">
      {prepare && <button className="primary" type="button" onClick={() => {close?.(); prepare(node.id);}}>Explain this <Icon name="arrow"/></button>}
      {focusAvailable && <button className="secondary focus-map" type="button" onClick={() => focusMap(node.id)}>Focus on map <Icon name="map"/></button>}
    </div>}
    <SourceList key={node.id} className="concept-notes"
      title={records.length ? 'Knowledge graph sources' : 'Source details unavailable'}>
      {records.map(source => <GraphSource key={source.source_id} source={source}/>)}
    </SourceList>
    <section className="concept-relations">
      <h3>Relationships <span className="count">{edges.length}</span></h3>
      {edges.length > 0 && <ul>{edges.map(edge => {
        const outgoing = edge.source === node.id;
        const other = nodes.get(outgoing ? edge.target : edge.source);
        const relation = <span className="relationship-label">
          {edge.label}{!outgoing && ` ${node.label}`}{edge.synthetic && ' · Synthetic'}
        </span>;
        return <li key={edge.id}>
          {outgoing && relation}
          <button type="button" onClick={() => follow(other.id)}>
            <Family name={other.family}/>{other.label}<Icon name="arrow"/>
          </button>
          {!outgoing && relation}
        </li>;
      })}</ul>}
    </section>
  </div>;
}

export function KnowledgeGraph({graph, maps = [], onSelectMap, ...props}) {
  const data = useMemo(() => normalizeGraph(graph), [graph]);
  const desktopPicker = useRef(null), mobilePicker = useRef(null), previousPicker = useRef(null);
  useLayoutEffect(() => {
    const previous = previousPicker.current;
    previousPicker.current = null;
    if (previous && !previous.isConnected && document.activeElement === document.body) {
      [desktopPicker.current, mobilePicker.current].find(picker => picker?.getClientRects().length)
        ?.focus({preventScroll: true});
    }
  }, [data.id]);

  const pickMap = ref => <select ref={ref} aria-label="Session map" title={data.title} value={data.id} onChange={event => {
    if (document.activeElement === event.currentTarget) previousPicker.current = event.currentTarget;
    onSelectMap(event.currentTarget.value);
  }}>
    {maps.map(item => <option key={item.artifact_id} value={item.artifact_id}>{item.title}</option>)}
  </select>;
  const canSwitchMaps = onSelectMap && maps.length > 1;
  const retrievalText = Array.isArray(graph?.retrieval_queries) ? graph.retrieval_queries.join(' & ') : '';
  return <Workspace key={data.id} data={data} {...props} sources={props.sources ?? graph?.sources}
    retrievalText={retrievalText} mapPicker={canSwitchMaps && pickMap(desktopPicker)}
    mobileMapPicker={canSwitchMaps && pickMap(mobilePicker)}/>;
}

function Workspace({data, sources, theme = 'dark', active = true, selectionId, selectionNonce,
  onPrepare, canClearSession, mapPicker, mobileMapPicker, retrievalText}) {
  const {id, title, nodes, edges} = data;
  const byId = useMemo(() => new Map(nodes.map(node => [node.id, node])), [nodes]);
  const suppliedSources = useMemo(() => new Map((Array.isArray(sources) ? sources : [])
    .filter(source => typeof source?.source_id === 'string' && (source.title == null || typeof source.title === 'string'))
    .map(source => [source.source_id, source])), [sources]);
  const available = nodes.length > 0 && nodes.length <= mapLimits.nodes && edges.length <= mapLimits.edges;
  const [selected, setSelected] = useState(null);
  const [view, setView] = useState('map');
  const [hiddenFamilies, setHiddenFamilies] = useState(() => new Set());
  const [searchResult, setSearchResult] = useState('');
  const [layout, setLayout] = useState(null);
  const [moved, setMoved] = useState(() => new Map());
  const [highlighted, setHighlighted] = useState(null);
  const root = useRef(null), canvas = useRef(null), stage = useRef(null), zoomLabel = useRef(null);
  const cards = useRef(null), labelLayer = useRef(null), help = useRef(null);
  const details = useRef(null), tools = useRef(null);
  const size = useRef(null), initialized = useRef(false);
  const pointer = useRef(null), dragFrame = useRef(null), suppressClick = useRef(false);
  const hoverEdge = useRef(null), command = useRef(null), pendingCenter = useRef(null);
  const headingFocus = useRef(false);
  const {current: camera, paint, stop, center, zoomAt} = useGraphCamera(canvas, stage, zoomLabel);
  const marker = useId();
  const headingId = useId();
  const map = view === 'map' && available;
  const visibleNodes = useMemo(() => nodes.filter(node => !hiddenFamilies.has(node.family)), [nodes, hiddenFamilies]);
  const visibleEdges = useMemo(() => edges.filter(edge =>
    !hiddenFamilies.has(byId.get(edge.source).family) && !hiddenFamilies.has(byId.get(edge.target).family)),
  [edges, byId, hiddenFamilies]);
  const adjacent = useMemo(() => {
    const index = new Map(nodes.map(node => [node.id, []]));
    for (const edge of edges) {
      index.get(edge.source).push(edge);
      if (edge.target !== edge.source) index.get(edge.target).push(edge);
    }
    return index;
  }, [nodes, edges]);
  const mostConnected = useMemo(() => [...nodes].sort((a, b) =>
    adjacent.get(b.id).length - adjacent.get(a.id).length || a.id.localeCompare(b.id)).slice(0, 4), [nodes, adjacent]);
  const selection = byId.get(selected);
  const relations = adjacent.get(selected) ?? [];
  const neighbors = new Set(relations.flatMap(edge => [edge.source, edge.target]));
  const selectedEdges = new Set(relations.map(edge => edge.id));
  const activeEdge = visibleEdges.find(edge => edge.id === highlighted);
  const rects = useMemo(() => new Map(layout ? visibleNodes.map(node => {
    const [x, y] = moved.get(node.id) ?? layout.positions[node.id];
    return [node.id, {x, y, ...layout.sizes.get(node.id)}];
  }) : []), [visibleNodes, moved, layout]);
  const routes = useMemo(() => map && layout ? routeEdges(visibleEdges, rects) : new Map(),
    [map, layout, visibleEdges, rects]);
  const labels = visibleEdges.filter(edge => routes.has(edge.id) && (selectedEdges.has(edge.id) || edge === activeEdge))
    .sort((a, b) => Number(b === activeEdge) - Number(a === activeEdge));

  const cancelCamera = useCallback(() => {
    pendingCenter.current = null;
    stop();
  }, [stop]);

  function closeDrawers() {
    details.current?.close();
    tools.current?.close();
  }

  function select(id, jump = false) {
    cancelCamera();
    hoverEdge.current = null;
    setHighlighted(null);
    setSelected(id);
    const node = byId.get(id);
    if (!node) return;
    setHiddenFamilies(previous => {
      if (!previous.has(node.family)) return previous;
      const next = new Set(previous);
      next.delete(node.family);
      return next;
    });
    if (jump && available) {
      if (id === selected && map && layout && canvas.current.clientWidth) center(moved.get(id) ?? layout.positions[id]);
      else pendingCenter.current = {id, animate: false};
    }
  }

  function clear() {
    if (details.current?.open) details.current.close();
    select(null);
    if (map) canvas.current.focus({preventScroll: true});
  }

  function focusMap(id) {
    cancelCamera();
    closeDrawers();
    pendingCenter.current = {id, animate: true};
    if (map && layout && size.current) {
      pendingCenter.current = null;
      canvas.current.focus({preventScroll: true});
      center(moved.get(id) ?? layout.positions[id], true, Math.max(.95, camera.current.zoom));
    } else {
      setView('map');
    }
  }

  function changeView(next) {
    cancelCamera();
    hoverEdge.current = null;
    setHighlighted(null);
    setView(next);
  }

  // An artifact is an immutable payload. A new revision must receive a new
  // artifact identity or a different React key from its host.
  useEffect(() => {
    if (!active || !available || !map || layout) return;
    const viewport = canvas.current;
    let fontsReady = false, disposed = false, frame;
    const measure = () => {
      if (disposed || !fontsReady || !viewport.clientWidth || !viewport.clientHeight || frame != null) return;
      frame = requestAnimationFrame(() => {
        frame = null;
        if (disposed || !cards.current || !viewport.clientWidth || !viewport.clientHeight) return;
        disposed = true;
        observer.disconnect();
        const elements = [...cards.current.children];
        const sizes = new Map(elements.map(element =>
          [element.dataset.measureNode, {w: element.offsetWidth, h: element.offsetHeight}]));
        // Reserve room for selected text without moving or resizing cards later.
        elements.forEach(element => element.classList.add('is-selected'));
        elements.forEach(element => {
          const size = sizes.get(element.dataset.measureNode);
          size.h = Math.max(size.h, element.offsetHeight);
        });
        const measured = nodes.map(node => {
          const {w, h} = sizes.get(node.id);
          return {id: node.id, width: w + 24, height: h + 50};
        });
        setLayout({...placeGraph(measured, edges), sizes});
      });
    };
    const observer = new ResizeObserver(measure);
    observer.observe(viewport);
    const fonts = document.fonts;
    const text = nodes.map(node => node.label.slice(0, 200)).join(' ');
    Promise.allSettled([fonts.load('500 14px Archivo', text), fonts.load('600 14px Archivo', text)])
      .then(() => {
        if (disposed) return;
        fontsReady = true;
        measure();
      });
    return () => {
      disposed = true;
      observer.disconnect();
      if (frame != null) cancelAnimationFrame(frame);
    };
  }, [active, available, map, layout, nodes, edges]);

  useLayoutEffect(() => {
    if (!active || !map || !layout || (initialized.current && !pendingCenter.current)) return;
    const viewport = canvas.current;
    const width = viewport.clientWidth, height = viewport.clientHeight;
    if (!width || !height) return;
    const previous = size.current;
    size.current = {width, height};
    if (previous && (previous.width !== width || previous.height !== height)) stop();
    if (!initialized.current) {
      initialized.current = true;
      if (window.matchMedia(mobileQuery).matches) {
        center(layout.positions[selected ?? layout.root], false, .9);
      } else {
        paint(fittedCamera(viewport, rects));
      }
    }
    const request = pendingCenter.current;
    if (request && layout.positions[request.id]) {
      pendingCenter.current = null;
      if (request.animate) viewport.focus({preventScroll: true});
      center(moved.get(request.id) ?? layout.positions[request.id], request.animate,
        request.animate ? Math.max(.95, camera.current.zoom) : camera.current.zoom);
    }
  });

  useEffect(() => {
    if (!active || !map || !layout) return;
    const viewport = canvas.current;
    const observer = new ResizeObserver(() => {
      const width = viewport.clientWidth, height = viewport.clientHeight;
      if (!width || !height) return;
      const previous = size.current;
      if (previous && (previous.width !== width || previous.height !== height)) cancelCamera();
      size.current = {width, height};
    });
    observer.observe(viewport);
    return () => observer.disconnect();
  }, [active, map, layout, cancelCamera]);

  useLayoutEffect(() => {
    if (!map || !layout || !active) return;
    const elements = [...labelLayer.current.children];
    const measurements = new Map(elements.map(element =>
      [element.dataset.edgeLabel, {w: element.offsetWidth, h: element.offsetHeight}]));
    const placements = placeLabels(elements.map(element => element.dataset.edgeLabel), highlighted,
      measurements, routes, rects, layout);
    for (const element of elements) {
      const point = placements.get(element.dataset.edgeLabel);
      element.style.visibility = point ? 'visible' : 'hidden';
      if (point) {
        element.style.left = `${point.x}px`;
        element.style.top = `${point.y}px`;
      }
    }
  }, [active, map, layout, labels, highlighted, rects, routes]);

  useEffect(() => {
    if (selectionNonce == null || command.current === selectionNonce || !byId.has(selectionId)) return;
    command.current = selectionNonce;
    select(selectionId, true);
  }, [selectionId, selectionNonce, byId]);

  const stopPointer = useCallback(() => {
    if (dragFrame.current != null) cancelAnimationFrame(dragFrame.current);
    dragFrame.current = null;
    const drag = pointer.current;
    pointer.current = null;
    if (drag?.capture.hasPointerCapture(drag.id)) drag.capture.releasePointerCapture(drag.id);
    canvas.current?.classList.remove('is-dragging');
  }, [canvas]);

  useEffect(() => {
    if (!active || !map) {
      cancelCamera();
      stopPointer();
      if (!active) closeDrawers();
    }
    return stopPointer;
  }, [active, map, cancelCamera, stopPointer]);

  useEffect(() => {
    const breakpoint = window.matchMedia(mobileQuery);
    const dismiss = () => {
      const wasOpen = details.current?.open || tools.current?.open;
      closeDrawers();
      if (wasOpen) root.current?.querySelector('[aria-label="Graph presentation"] button[aria-pressed=true]')
        ?.focus({preventScroll: true});
    };
    breakpoint.addEventListener('change', dismiss);
    return () => breakpoint.removeEventListener('change', dismiss);
  }, []);

  useEffect(() => {
    const viewport = canvas.current;
    if (!active || !map || !layout) return;
    function wheel(event) {
      event.preventDefault();
      cancelCamera();
      hoverEdge.current = null;
      setHighlighted(null);
      const {left, top} = viewport.getBoundingClientRect();
      zoomAt(Math.exp(-event.deltaY * .0015), [event.clientX - left, event.clientY - top]);
    }
    viewport.addEventListener('wheel', wheel, {passive: false});
    return () => viewport.removeEventListener('wheel', wheel);
  }, [active, map, layout, cancelCamera, zoomAt]);

  useEffect(() => {
    if (!active || !map || !layout) return;
    function outsidePointer(event) {
      if (root.current.contains(event.target)) return;
      if (help.current?.open) help.current.open = false;
      cancelCamera();
      if (canvas.current.contains(document.activeElement) && document.activeElement?.dataset.edgeHit) {
        canvas.current.focus({preventScroll: true});
      }
      hoverEdge.current = null;
      setHighlighted(null);
    }
    document.addEventListener('pointerdown', outsidePointer);
    return () => document.removeEventListener('pointerdown', outsidePointer);
  }, [active, map, layout, cancelCamera]);

  function beginPointer(event) {
    if (event.button !== 0 || !event.isPrimary || !layout || pointer.current ||
      event.target.closest('[data-edge-hit]')) return;
    const button = event.target.closest('[data-node]');
    const node = button?.dataset.node;
    const capture = button ?? canvas.current;
    capture.setPointerCapture(event.pointerId);
    pointer.current = {id: event.pointerId, capture, node, card: rects.get(node),
      startX: event.clientX, startY: event.clientY, camera: camera.current, moved: false};
    suppressClick.current = false;
  }

  function movePointer(event) {
    const drag = pointer.current;
    if (!drag) {
      if (event.pointerType === 'touch') return;
      const edge = event.target.closest('[data-edge-hit]')?.dataset.edgeHit;
      hoverEdge.current = edge ?? null;
      setHighlighted(edge ?? document.activeElement?.dataset.edgeHit ?? null);
      return;
    }
    if (drag.id !== event.pointerId) return;
    const dx = event.clientX - drag.startX, dy = event.clientY - drag.startY;
    if (!drag.moved && Math.hypot(dx, dy) < 5) return;
    drag.moved = true;
    canvas.current.classList.add('is-dragging');
    if (!drag.card) {
      paint({...drag.camera, x: drag.camera.x + dx, y: drag.camera.y + dy});
      return;
    }
    const {x, y, w, h} = drag.card, zoom = drag.camera.zoom;
    drag.next = [
      Math.max(w / 2 + 8, Math.min(layout.width - w / 2 - 8, x + dx / zoom)),
      Math.max(h / 2 + 34, Math.min(layout.height - h / 2 - 8, y + dy / zoom)),
    ];
    if (dragFrame.current != null) return;
    dragFrame.current = requestAnimationFrame(() => {
      dragFrame.current = null;
      setMoved(old => new Map(old).set(drag.node, drag.next));
    });
  }

  function finishPointer(event) {
    const drag = pointer.current;
    if (drag?.id !== event.pointerId) return;
    if (drag.next) setMoved(old => new Map(old).set(drag.node, drag.next));
    suppressClick.current = drag.moved;
    stopPointer();
    if (event.type === 'pointerup' && !drag.moved && !drag.node) select(null);
  }

  function find(event) {
    event.preventDefault();
    const query = new FormData(event.currentTarget).get('query').trim().toLocaleLowerCase();
    const result = query && nodes.find(node => node.label.toLocaleLowerCase().includes(query));
    setSearchResult(result ? `Selected ${result.label}.` : query ? 'No matching concept in this map.' : '');
    if (!result) return;
    closeDrawers();
    if (available) setView('map');
    select(result.id, true);
  }

  function restoreMap() {
    cancelCamera();
    closeDrawers();
    select(null);
    setHiddenFamilies(new Set());
    setMoved(new Map());
    setSearchResult('');
    root.current.querySelectorAll('input[type=search]').forEach(input => { input.value = ''; });
    if (available) {
      initialized.current = false;
      setView('map');
    }
  }

  function toggleFamily(family) {
    cancelCamera();
    setHighlighted(null);
    if (selection?.family === family && !hiddenFamilies.has(family)) setSelected(null);
    setHiddenFamilies(previous => {
      const next = new Set(previous);
      if (next.has(family)) next.delete(family);
      else next.add(family);
      return next;
    });
  }

  function keys(event) {
    if (event.target instanceof HTMLSelectElement) return;
    cancelCamera();
    if (event.key === 'Escape') {
      if (event.defaultPrevented || details.current?.open || tools.current?.open) return;
      if (highlighted) {
        event.preventDefault();
        hoverEdge.current = null;
        setHighlighted(null);
        canvas.current.focus({preventScroll: true});
      } else if (help.current?.open) {
        event.preventDefault();
        help.current.open = false;
        help.current.querySelector('summary').focus();
      } else if (selection) {
        event.preventDefault();
        clear();
      }
      return;
    }
    if (!map || !layout || !canvas.current.contains(event.target)) return;
    if (['ArrowRight', 'ArrowLeft', 'ArrowUp', 'ArrowDown'].includes(event.key)) {
      const buttons = [...canvas.current.querySelectorAll('[data-node]')];
      if (!buttons.length) return;
      event.preventDefault();
      const index = buttons.indexOf(event.target);
      const backward = ['ArrowLeft', 'ArrowUp'].includes(event.key);
      const next = buttons[index < 0 ? (backward ? buttons.length - 1 : 0)
        : (index + (backward ? -1 : 1) + buttons.length) % buttons.length];
      next.focus({preventScroll: true});
      const rect = rects.get(next.dataset.node), viewport = canvas.current;
      const bounds = next.getBoundingClientRect(), frame = viewport.getBoundingClientRect();
      if (window.matchMedia(mobileQuery).matches ||
        bounds.left < frame.left || bounds.right > frame.right || bounds.top < frame.top || bounds.bottom > frame.bottom) {
        center([rect.x, rect.y]);
      }
    }
    if (['+', '=', '-'].includes(event.key)) {
      event.preventDefault();
      zoomAt(event.key === '-' ? 1 / 1.15 : 1.15);
    }
  }

  const familyKeys = Object.keys(families).filter(key => key !== 'unknown' || nodes.some(node => node.family === key));
  const filters = <div className="family-filters" role="group" aria-label="Visible concept types">
    {familyKeys.map(key => <button type="button" key={key} aria-pressed={!hiddenFamilies.has(key)} onClick={() => toggleFamily(key)}>
      <Family name={key}/><span>{families[key]}</span><span>{nodes.filter(node => node.family === key).length}</span>
    </button>)}
  </div>;
  const findForm = <><form className="find-form" role="search" onSubmit={find}>
    <label><Icon name="search"/><span className="graph-sr-only">Find a concept</span>
      <input name="query" type="search" placeholder="Find a concept…"/></label>
  </form><p className="find-status" role="status">{searchResult}</p></>;
  const inspector = {node: selection, nodes: byId, edges: relations, mostConnected, select, clear, focusMap,
    focusAvailable: available, sources: suppliedSources, requestHeading: headingFocus,
    prepare: onPrepare && (concept => onPrepare(id, concept))};
  const clearSession = canClearSession && <div className="history-foot">
    <button className="clear-session" type="button" aria-haspopup="dialog"
      aria-controls="clear-dialog" data-practice-action="clear"><Icon name="trash"/>Clear session</button>
  </div>;
  const visibleRelationView = !map && nodes.length > 0;
  const visibleEndpoints = visibleRelationView ? new Set(visibleEdges.flatMap(edge => [edge.source, edge.target])) : null;
  const isolated = visibleEndpoints ? visibleNodes.filter(node => !visibleEndpoints.has(node.id)) : [];
  const conceptCount = `${visibleNodes.length} ${visibleNodes.length === 1 ? 'concept' : 'concepts'}`;
  const relationCount = `${visibleEdges.length} ${visibleEdges.length === 1 ? 'relation' : 'relations'}`;
  const layoutStatus = data.status ?? (data.incomplete ? 'Some graph details are unavailable.' : null);

  return <section ref={root} className="balanced-graph" aria-label="Knowledge graph" data-graph-theme={theme === 'light' ? 'light' : 'dark'}
    hidden={!active} onKeyDown={keys}
    onPointerDownCapture={event => {
      cancelCamera();
      if (help.current?.open && !help.current.contains(event.target)) help.current.open = false;
      if (event.target.closest('[data-edge-hit]')) return;
      if (canvas.current.contains(document.activeElement) && document.activeElement?.dataset.edgeHit) {
        canvas.current.focus({preventScroll: true});
      }
      hoverEdge.current = null;
      setHighlighted(null);
    }}>
    <aside className="map-panel" aria-label="Map controls">
      <h2 className="map-section-title">Map controls</h2>
      {mapPicker && <div className="map-picker">{mapPicker}</div>}
      {findForm}<h3 className="concept-types-title">Concept types</h3>{filters}
      {clearSession}
    </aside>
    <div className="graph-center" data-view={map ? 'map' : 'relations'}>
      <header className="canvas-caption">
        <h2 id={headingId}>{visibleRelationView ? 'Relations in this map' : 'Knowledge graph'}</h2>
        <span className="map-counts">{conceptCount} · {relationCount}</span>
        {map && (active && retrievalText
          ? <Tooltip content={retrievalText} className="map-request-tooltip" side="bottom" align="start"
              collisionPadding={12} hideWhenDetached>
              <p className="map-title" tabIndex={0}>{title}</p>
            </Tooltip>
          : <p className="map-title">{title}</p>)}
        <button type="button" className="icon-button mobile-only" aria-label="Open graph filters"
          onClick={() => tools.current.showModal()}><Icon name="filter"/></button>
        {mobileMapPicker && <div className="mobile-only map-picker-mobile">{mobileMapPicker}</div>}
        {map && layoutStatus && <p className="layout-context" role="status">{layoutStatus}</p>}
      </header>
      <div ref={canvas} className="graph-canvas" hidden={!map} tabIndex={0}
        aria-label="Concept map. Arrow keys navigate, Enter selects, Escape clears. Drag to reposition or pan."
        onPointerDown={beginPointer} onPointerMove={movePointer} onPointerUp={finishPointer}
        onPointerCancel={finishPointer} onLostPointerCapture={finishPointer}
        onPointerLeave={() => {hoverEdge.current = null; setHighlighted(document.activeElement?.dataset.edgeHit ?? null);}}
        onClickCapture={event => {if (event.detail && suppressClick.current) {
          suppressClick.current = false;
          event.stopPropagation();
        }}}>
        {!layout && available && <div ref={cards} className="graph-measure" aria-hidden="true">
          {nodes.map(node => <div key={node.id} className="graph-node" data-measure-node={node.id} data-node-family={node.family}>
            <Family name={node.family}/><span>{node.label}</span>
          </div>)}
        </div>}
        <div ref={stage} className="graph-stage" style={{width: layout?.width, height: layout?.height}}>
          {map && layout && <>
            <svg className="graph-edges" viewBox={`0 0 ${layout.width} ${layout.height}`} role="group" aria-label="Directed relationships">
              <defs>{['normal', 'active'].map(kind => <marker key={kind} id={`${marker}-${kind}`} markerWidth="8" markerHeight="8"
                refX="7" refY="4" orient="auto" markerUnits="userSpaceOnUse"><path d="M0 0L8 4 0 8Z" className={`arrow-${kind}`}/></marker>)}</defs>
              {visibleEdges.filter(edge => routes.has(edge.id)).sort((a, b) =>
                (b === activeEdge ? -2 : a === activeEdge ? 2 : Number(selectedEdges.has(a.id)) - Number(selectedEdges.has(b.id))))
                .map(edge => {
                  const active = selectedEdges.has(edge.id) || edge === activeEdge;
                  return <path key={edge.id} data-edge={edge.id} aria-hidden="true"
                    className={`graph-edge ${active ? 'is-active' : selected ? 'is-muted' : ''} ${edge === activeEdge ? 'is-hovered' : ''}`}
                    markerEnd={`url(#${marker}-${active ? 'active' : 'normal'})`} d={routes.get(edge.id).d}/>;
                })}
              {visibleEdges.filter(edge => routes.has(edge.id)).map(edge => <path key={edge.id}
                className="graph-edge-hit" data-edge-hit={edge.id} tabIndex={0} role="button" d={routes.get(edge.id).d}
                aria-label={`Highlight relationship: ${byId.get(edge.source).label} ${edge.label} ${byId.get(edge.target).label}${edge.synthetic ? ' (synthetic)' : ''}`}
                onFocus={() => setHighlighted(edge.id)} onBlur={() => setHighlighted(hoverEdge.current)}
                onClick={event => {event.currentTarget.focus({preventScroll: true}); setHighlighted(edge.id);}}
                onKeyDown={event => {if (event.key === 'Enter' || event.key === ' ') {
                  event.preventDefault();
                  setHighlighted(edge.id);
                }}}/>)}
            </svg>
            <div className="node-layer">{visibleNodes.map(node => {
              const {x, y} = rects.get(node.id);
              const emphasis = node.id === selected ? 'is-selected' : neighbors.has(node.id) ? 'is-neighbor' : selection ? 'is-muted' : '';
              const endpoint = activeEdge && [activeEdge.source, activeEdge.target].includes(node.id);
              return <button type="button" key={node.id} data-node={node.id} data-node-family={node.family}
                aria-label={`${node.label} · ${families[node.family]}`} aria-pressed={node.id === selected}
                className={`graph-node ${emphasis} ${endpoint ? 'is-edge-endpoint' : ''}`}
                style={{left: x, top: y, height: rects.get(node.id).h}} onClick={() => select(node.id, window.matchMedia(mobileQuery).matches)}>
                <Family name={node.family}/><span>{node.label}</span>
              </button>;
            })}</div>
            <div className="edge-label-layer" ref={labelLayer} aria-hidden="true">
              {labels.map(edge => <span key={edge.id} data-edge-label={edge.id}
                className={`edge-label ${edge === activeEdge ? 'is-hovered' : ''}`}>{edge.label}</span>)}
            </div>
          </>}
        </div>
        {layout && !visibleNodes.length && <p className="graph-empty">No concept types selected.<br/>Choose a type to restore the map.</p>}
      </div>
      {visibleRelationView && <div className="relations-view">
        {(!available || layoutStatus) && <p className="relations-status" role="status">
          {!available ? 'This graph is too large for Map. All relationships are available in Relations.' : layoutStatus}
        </p>}
        <table aria-labelledby={headingId}>
          <thead><tr><th scope="col">Source concept</th><th scope="col">Relationship</th><th scope="col">Target concept</th></tr></thead>
          <tbody>{visibleEdges.map(edge => <tr key={edge.id} className={selectedEdges.has(edge.id) ? 'is-selected' : ''}>
            <td><button type="button" aria-pressed={selected === edge.source} onClick={() => select(edge.source)}>
              <Family name={byId.get(edge.source).family}/>{byId.get(edge.source).label}</button></td>
            <td>{edge.label}{edge.synthetic && <small> · Synthetic</small>}</td>
            <td><button type="button" aria-pressed={selected === edge.target} onClick={() => select(edge.target)}>
              <Family name={byId.get(edge.target).family}/>{byId.get(edge.target).label}</button></td>
          </tr>)}</tbody>
        </table>
        {!visibleEdges.length && <p>No relationships among the visible concepts.</p>}
        {/* Isolated concepts remain reachable when the map is too large to render. */}
        {isolated.length > 0 &&
          <details className="isolated-concepts"><summary>Concepts without visible relationships</summary>
            {isolated.map(node =>
              <button type="button" key={node.id} aria-pressed={selected === node.id} onClick={() => select(node.id)}>
                <Family name={node.family}/>{node.label}
              </button>)}
          </details>}
      </div>}
      {!nodes.length && <p className="graph-unavailable" role="status">There are no concepts to display.</p>}
      <div className="graph-footer">
        <div className="view-tabs" role="group" aria-label="Graph presentation">
          <button type="button" disabled={!available} aria-pressed={map} onClick={() => changeView('map')}>Map</button>
          <button type="button" disabled={!nodes.length} aria-pressed={visibleRelationView} onClick={() => changeView('relations')}>Relations</button>
        </div>
        {selection && <button type="button" className="clear-selection" aria-label="Clear selection" onClick={clear}>
          <span className="desktop-only">Clear selection</span><span className="mobile-only">Clear</span>
        </button>}
        {map && <>
          <details ref={help} className="graph-help">
            <summary aria-label="Graph interaction tips">?</summary>
            <div className="graph-help-popover" role="region" aria-label="Graph interaction tips">
              <ul><li><span><strong>Drag a node</strong> to reposition it.</span></li>
                <li><span><strong>Drag the background</strong> to pan.</span></li>
                <li><span><strong>Scroll</strong> or use <kbd>+</kbd> / <kbd>−</kbd> to zoom.</span></li>
              </ul>
            </div>
          </details>
          <div className="zoom-controls">
            <button type="button" className="icon-button" aria-label="Zoom out" disabled={!layout} onClick={() => zoomAt(1 / 1.15)}>−</button>
            <output ref={zoomLabel} aria-label="Zoom level">{Math.round(camera.current.zoom * 100)}%</output>
            <button type="button" className="icon-button" aria-label="Zoom in" disabled={!layout} onClick={() => zoomAt(1.15)}>+</button>
            <button type="button" className="icon-button" aria-label="Fit graph" disabled={!layout}
              onClick={() => {cancelCamera(); paint(fittedCamera(canvas.current, rects));}}><Icon name="fit"/></button>
          </div>
        </>}
      </div>
      {nodes.length > 0 && <button className="inspector-peek mobile-only" type="button" onClick={() => details.current.showModal()}>
        {selection ? <><Family name={selection.family}/><span><span className="peek-label">{selection.label}</span><small>{selection.sourceIds.length ? 'Source references available' : 'No excerpts'}</small></span></>
          : <><Icon name="map"/><span>Map overview</span></>}
        <span>{selection && 'Details'} <Icon name="chevron"/></span>
      </button>}
    </div>
    <aside className="concept-inspector" aria-label="Selected concept"><Inspector {...inspector}/></aside>
    <Drawer dialog={details} label={selection ? 'Selected concept' : 'Map overview'} className="inspector-drawer">
      <Inspector {...inspector} close={() => details.current.close()}/>
    </Drawer>
    <Drawer dialog={tools} label="Graph controls" className="map-tools-drawer">
      <div className="drawer-heading"><h2>Map controls</h2>
        <button className="icon-button" type="button" aria-label="Close graph controls" onClick={() => tools.current.close()}><Icon name="close"/></button>
      </div>
      {findForm}<h3 className="concept-types-title">Concept types</h3>{filters}
      {available && <button type="button" className="secondary" onClick={restoreMap}>Restore map</button>}
      {clearSession}
    </Drawer>
  </section>;
}
