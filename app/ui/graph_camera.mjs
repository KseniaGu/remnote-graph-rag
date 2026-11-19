import {useCallback, useEffect, useRef} from 'react';

export const mobileQuery = '(max-width: 959px)';
const focusDuration = 450;

export function useGraphCamera(canvas, stage, zoomLabel) {
  const current = useRef({x: 0, y: 0, zoom: 1});
  const frame = useRef(null);
  const destination = useRef(null);

  const paint = useCallback((camera = current.current) => {
    current.current = camera;
    if (stage.current) {
      stage.current.style.transform = `translate(${camera.x}px,${camera.y}px) scale(${camera.zoom})`;
    }
    if (zoomLabel.current) zoomLabel.current.value = `${Math.round(camera.zoom * 100)}%`;
  }, [stage, zoomLabel]);

  const stop = useCallback(() => {
    if (frame.current !== null) cancelAnimationFrame(frame.current);
    frame.current = null;
    destination.current = null;
  }, []);

  useEffect(() => {
    const preference = window.matchMedia('(prefers-reduced-motion: reduce)');
    function settle() {
      if (!preference.matches || !destination.current) return;
      paint(destination.current);
      stop();
    }
    preference.addEventListener('change', settle);
    return () => {
      preference.removeEventListener('change', settle);
      stop();
    };
  }, [paint, stop]);

  const center = useCallback(([x, y], animated = false, zoom = current.current.zoom) => {
    stop();
    const target = {x: canvas.current.clientWidth / 2 - x * zoom, y: canvas.current.clientHeight / 2 - y * zoom, zoom};
    const origin = current.current;
    if (!animated || window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
      paint(target);
      return;
    }
    if (Math.hypot(target.x - origin.x, target.y - origin.y) < .5 && Math.abs(zoom - origin.zoom) < .0001) return;

    destination.current = target;
    const started = performance.now();
    const step = now => {
      const progress = Math.min((now - started) / focusDuration, 1);
      if (progress === 1) {
        paint(target);
        stop();
        return;
      }
      const eased = 1 - (1 - progress) ** 3;
      paint({
        x: origin.x + (target.x - origin.x) * eased,
        y: origin.y + (target.y - origin.y) * eased,
        zoom: origin.zoom + (target.zoom - origin.zoom) * eased,
      });
      frame.current = requestAnimationFrame(step);
    };
    frame.current = requestAnimationFrame(step);
  }, [canvas, paint, stop]);

  const zoomAt = useCallback((factor, point) => {
    stop();
    const {zoom, x, y} = current.current;
    const next = Math.max(.12, Math.min(2.4, zoom * factor));
    const [px, py] = point ?? [canvas.current.clientWidth / 2, canvas.current.clientHeight / 2];
    paint({x: px - (px - x) * next / zoom, y: py - (py - y) * next / zoom, zoom: next});
  }, [canvas, paint, stop]);

  return {current, paint, stop, center, zoomAt};
}

export function fittedCamera(canvas, rects) {
  const cards = [...rects.values()];
  const left = Math.min(0, ...cards.map(n => n.x - n.w / 2)) - 22;
  const top = Math.min(0, ...cards.map(n => n.y - n.h / 2)) - 22;
  const right = Math.max(100, ...cards.map(n => n.x + n.w / 2)) + 22;
  const bottom = Math.max(100, ...cards.map(n => n.y + n.h / 2)) + 22;
  const width = canvas.clientWidth, height = canvas.clientHeight;
  const zoom = Math.max(.12, Math.min(1.15, (width - 20) / (right - left), (height - 20) / (bottom - top)));
  return {x: (width - (right - left) * zoom) / 2 - left * zoom,
    y: (height - (bottom - top) * zoom) / 2 - top * zoom, zoom};
}
