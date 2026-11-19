import { useEffect, useId, useLayoutEffect, useRef, useState } from "react";

const THEME_KEY = "ai-practice-theme";
let mermaidReady = false;

function fitComposer(input) {
  if (!input) return;
  input.style.height = "auto";
  input.style.height = `${input.scrollHeight}px`;
}

function revealInThread(thread, target, alignStart = false) {
  const bounds = target.getBoundingClientRect();
  const viewport = thread.getBoundingClientRect();
  const padding = Number.parseFloat(getComputedStyle(thread).scrollPaddingTop) || 0;
  const start = bounds.top - (viewport.top + thread.clientTop + padding);
  const end = bounds.bottom - (viewport.top + thread.clientTop + thread.clientHeight);
  if (alignStart || start < 0) thread.scrollTop += start;
  else if (end > 0) thread.scrollTop += Math.min(start, end);
}

function highlightHistory(shell, targetId) {
  for (const link of shell.querySelectorAll("[data-history-target]")) {
    if (link.dataset.historyTarget === targetId) link.setAttribute("aria-current", "location");
    else link.removeAttribute("aria-current");
  }
}

export function MermaidDiagram({ source }) {
  const id = useId().replace(/[^a-zA-Z0-9]/g, "");
  const renderNumber = useRef(0);
  const [diagram, setDiagram] = useState(null);

  useEffect(() => {
    let active = true;
    const script = document.querySelector("[data-practice-mermaid]");
    let started = false;
    const render = async () => {
      const mermaid = window.mermaid;
      if (!mermaid || started || !active) return;
      started = true;
      if (!mermaidReady) {
        mermaid.initialize({
          startOnLoad: false,
          securityLevel: "strict",
          theme: "base",
          themeVariables: { primaryColor: "#e3e7fa", primaryTextColor: "#242e47" },
        });
        mermaidReady = true;
      }
      const renderId = `mermaid-${id}-${++renderNumber.current}`;
      try {
        // Mermaid inserts an error SVG when a streaming fence is incomplete.
        if (!await mermaid.parse(source, { suppressErrors: true }) || !active) return;
        const { svg } = await mermaid.render(renderId, source);
        if (active) setDiagram({ source, svg });
      } catch {
        document.getElementById(`d${renderId}`)?.remove();
      }
    };
    const timer = setTimeout(render, 150);
    script?.addEventListener("load", render, { once: true });
    return () => {
      active = false;
      clearTimeout(timer);
      script?.removeEventListener("load", render);
    };
  }, [id, source]);

  return (
    <div className="mermaid-wrapper">
      {diagram?.source === source
        ? <div dangerouslySetInnerHTML={{ __html: diagram.svg }} />
        : <pre><code>{source}</code></pre>}
    </div>
  );
}

export function PracticeShell({ children, screen, scrollNonce, scrollTarget, graphDraft,
  graphDraftNonce, graphSourceNonce }) {
  const root = useRef(null);
  const follow = useRef(true);
  const automaticScrollTop = useRef(null);
  const draft = useRef("");
  const draftSubmitted = useRef(false);
  const clearRequested = useRef(false);
  const welcomeDrawn = useRef(false);
  const lastHistoryRequest = useRef(null);
  const historyTarget = useRef("");
  const previousScreen = useRef(screen);
  const lastGraphDraft = useRef(graphDraftNonce);
  const lastGraphSource = useRef(graphSourceNonce);
  const openGraphSource = useRef(null);

  function scrollToBottom(thread) {
    if (!thread) return;
    thread.scrollTop = thread.scrollHeight;
    automaticScrollTop.current = thread.scrollTop;
  }

  function selectHistoryTarget(targetId) {
    if (historyTarget.current === targetId) return;
    historyTarget.current = targetId;
    highlightHistory(root.current, targetId);
  }

  useLayoutEffect(() => {
    const shell = root.current;
    const mobile = matchMedia("(width < 960px)");
    const compact = matchMedia("(width < 1280px)");
    const viewport = window.visualViewport;
    const narrow = () => mobile.matches;
    let theme = "dark";
    let toastTimer;
    let sizeFrame;
    try {
      if (localStorage.getItem(THEME_KEY) === "light") theme = "light";
    } catch {
      // A browser may restrict storage while still allowing a local theme.
    }

    function updateControls() {
      highlightHistory(shell, historyTarget.current);
      document.documentElement.dataset.practiceTheme = theme;
      for (const button of shell.querySelectorAll('[data-practice-action="theme"]')) {
        const label = `Switch to ${theme === "dark" ? "light" : "dark"} theme`;
        button.setAttribute("aria-label", label);
        button.title = label;
      }
      shell.querySelector('[data-practice-action="collapse-history"]')
        ?.setAttribute("aria-expanded", String(shell.dataset.history !== "collapsed"));
      const sourcesOpen = narrow()
        ? shell.querySelector("#sources-dialog")?.open
        : shell.dataset.sources !== "closed";
      const sourcesButton = shell.querySelector('[data-practice-action="open-sources-panel"]');
      sourcesButton?.setAttribute("aria-expanded", String(Boolean(sourcesOpen)));
      sourcesButton?.setAttribute("aria-controls", narrow() ? "sources-dialog" : "study-sources");
      if (shell.dataset.sourceRequest === shell.querySelector(".sources-content")?.dataset.sourcesKey) {
        delete shell.dataset.sourceRequest;
      }
      for (const button of shell.querySelectorAll('[data-source-selected]')) {
        const selected = !shell.dataset.sourceRequest && button.dataset.sourceSelected === "true";
        button.setAttribute("aria-pressed", String(Boolean(sourcesOpen) && selected));
      }
    }

    function onInput(event) {
      if (event.target.id !== "chat-input") return;
      draft.current = event.target.value;
      fitComposer(event.target);
    }

    function resize() {
      cancelAnimationFrame(sizeFrame);
      sizeFrame = requestAnimationFrame(() => {
        if (viewport && viewport.scale === 1) {
          shell.style.setProperty("--viewport-height", `${viewport.height}px`);
          shell.style.setProperty("--viewport-top", `${viewport.offsetTop}px`);
        } else {
          shell.style.removeProperty("--viewport-height");
          shell.style.removeProperty("--viewport-top");
        }
        fitComposer(shell.querySelector("#chat-input"));
        const active = document.activeElement;
        const thread = shell.querySelector("#chat-messages");
        if (active !== thread && thread?.contains(active) && !active.closest("dialog")) {
          const previous = thread.scrollTop;
          revealInThread(thread, active);
          if (thread.scrollTop !== previous) automaticScrollTop.current = thread.scrollTop;
        }
      });
    }

    function closeDrawers() {
      for (const drawer of shell.querySelectorAll(".practice-drawer[open]")) {
        drawer.close();
      }
    }

    function onBreakpoint() {
      if (!narrow()) {
        for (const drawer of shell.querySelectorAll(".practice-drawer[open]")) {
          if (shell.querySelector("#clear-dialog[open]")) continue;
          drawer.close();
          if (drawer.id === "history-dialog") {
            shell.querySelector('[data-practice-action="collapse-history"]')?.focus();
          } else {
            openSources();
          }
        }
      }
      updateControls();
    }

    function resetHistoryWidth() {
      shell.dataset.history = compact.matches ? "collapsed" : "expanded";
      updateControls();
    }

    function showDialog(dialog, opener) {
      if (!dialog || dialog.open) return;
      opener.focus({ preventScroll: true });
      dialog.showModal();
      updateControls();
    }

    function openSources(button) {
      if (narrow()) {
        showDialog(shell.querySelector("#sources-dialog"), button);
        return;
      }
      shell.dataset.sources = "open";
      shell.querySelector(".practice-sources h2")?.focus();
      updateControls();
    }

    openGraphSource.current = () => openSources(
      shell.querySelector('[data-practice-action="open-sources-panel"]'),
    );

    function toast(message) {
      if (!shell.isConnected) return;
      const status = shell.querySelector("#practice-toast");
      status.hidden = false;
      status.textContent = message;
      clearTimeout(toastTimer);
      toastTimer = setTimeout(() => { status.hidden = true; }, 3500);
    }

    async function copy(button, selector) {
      const content = button.closest("article, .code-block")?.querySelector(selector);
      if (!content) return;
      try {
        await navigator.clipboard.writeText(content.innerText);
        toast("Copied to clipboard.");
      } catch {
        toast("Clipboard unavailable. Select the text to copy it.");
      }
    }

    function onClick(event) {
      const button = event.target.closest("button");
      if (!button || button.disabled) return;
      if (button.dataset.draft) {
        const input = shell.querySelector("#chat-input");
        if (!input || input.readOnly) return;
        draft.current = input.value = button.dataset.draft;
        fitComposer(input);
        input.focus();
        input.setSelectionRange(input.value.length, input.value.length);
      }
      switch (button.dataset.practiceAction) {
        case "theme":
          theme = theme === "dark" ? "light" : "dark";
          try { localStorage.setItem(THEME_KEY, theme); } catch { /* Local only. */ }
          updateControls();
          if (shell.dataset.screen === "welcome") {
            const diagram = shell.querySelector(".welcome-diagram");
            if (diagram) {
              welcomeDrawn.current = true;
              diagram.dataset.draw = "done";
              if (!matchMedia("(prefers-reduced-motion: reduce)").matches) {
                // Flush the removed animation before starting the next draw.
                diagram.querySelector(".diagram-line").getAnimations();
                diagram.dataset.draw = "play";
              }
            }
          }
          break;
        case "history": showDialog(shell.querySelector("#history-dialog"), button); break;
        case "collapse-history":
          shell.dataset.history = shell.dataset.history === "collapsed" ? "expanded" : "collapsed";
          updateControls();
          break;
        case "collapse-sources":
          shell.dataset.sources = "closed";
          updateControls();
          shell.querySelector('[data-practice-action="open-sources-panel"]')?.focus();
          break;
        case "open-sources-panel": openSources(button); break;
        case "open-sources":
          shell.dataset.sourceRequest = button.closest("article").dataset.responseId;
          openSources(button);
          break;
        case "execution": {
          const panel = document.getElementById(button.getAttribute("aria-controls"));
          if (!panel) break;
          const expanded = panel.hidden;
          panel.hidden = !expanded;
          button.setAttribute("aria-expanded", String(expanded));
          if (expanded) {
            const thread = button.closest(".practice-thread");
            follow.current = false;
            revealInThread(thread, panel);
            revealInThread(thread, button);
            automaticScrollTop.current = thread.scrollTop;
          }
          break;
        }
        case "close": button.closest("dialog")?.close(); break;
        case "clear": showDialog(shell.querySelector("#clear-dialog"), button); break;
        case "confirm-clear":
          draft.current = "";
          clearRequested.current = true;
          button.closest("dialog")?.close();
          closeDrawers();
          break;
        case "history-link": closeDrawers(); break;
        case "copy-answer": copy(button, ".markdown-content"); break;
        case "copy-code": copy(button, "code"); break;
      }
    }

    function onBackdrop(event) {
      const dialog = event.target;
      if (!(dialog instanceof HTMLDialogElement)) return;
      const bounds = dialog.getBoundingClientRect();
      if (event.clientX < bounds.left || event.clientX > bounds.right ||
          event.clientY < bounds.top || event.clientY > bounds.bottom) {
        dialog.close();
      }
    }

    function onKeyDown(event) {
      if (
        event.target.id !== "chat-input" || event.key !== "Enter" ||
        event.isComposing || event.keyCode === 229 || event.shiftKey || event.altKey
      ) return;
      event.preventDefault();
      if (!event.target.readOnly) event.target.form.requestSubmit();
    }

    function onSubmit(event) {
      if (!event.target.classList.contains("practice-composer")) return;
      const input = event.target.querySelector("textarea");
      if (input.readOnly || !input.value.trim()) {
        event.preventDefault();
        event.stopPropagation();
        return;
      }
      follow.current = true;
      draft.current = "";
      draftSubmitted.current = true;
      scrollToBottom(shell.querySelector("#chat-messages"));
      resize();
    }

    function onDialogClose(event) {
      if (!narrow()) onBreakpoint();
      if (event.target.id === "clear-dialog" && !clearRequested.current && shell.dataset.screen === "graph") {
        const opener = narrow()
          ? shell.querySelector(".map-tools-drawer[open] .clear-session") ??
            shell.querySelector('.balanced-graph [aria-label="Open graph filters"]')
          : shell.querySelector(".map-panel .clear-session");
        opener?.focus({ preventScroll: true });
      }
      updateControls();
    }

    resetHistoryWidth();
    resize();
    const observer = new MutationObserver(updateControls);
    observer.observe(shell, {
      childList: true, subtree: true, attributes: true, attributeFilter: ["data-source-selected", "data-sources-key"],
    });
    shell.addEventListener("click", onClick);
    shell.addEventListener("click", onBackdrop);
    shell.addEventListener("close", onDialogClose, true);
    shell.addEventListener("keydown", onKeyDown);
    shell.addEventListener("submit", onSubmit, true);
    shell.addEventListener("input", onInput);
    mobile.addEventListener("change", onBreakpoint);
    compact.addEventListener("change", resetHistoryWidth);
    window.addEventListener("resize", resize);
    viewport?.addEventListener("resize", resize);
    viewport?.addEventListener("scroll", resize);
    return () => {
      observer.disconnect();
      cancelAnimationFrame(sizeFrame);
      clearTimeout(toastTimer);
      shell.removeEventListener("click", onClick);
      shell.removeEventListener("click", onBackdrop);
      shell.removeEventListener("close", onDialogClose, true);
      shell.removeEventListener("keydown", onKeyDown);
      shell.removeEventListener("submit", onSubmit, true);
      shell.removeEventListener("input", onInput);
      mobile.removeEventListener("change", onBreakpoint);
      compact.removeEventListener("change", resetHistoryWidth);
      window.removeEventListener("resize", resize);
      viewport?.removeEventListener("resize", resize);
      viewport?.removeEventListener("scroll", resize);
    };
  }, []);

  useLayoutEffect(() => {
    const shell = root.current;
    const changed = previousScreen.current !== screen;
    previousScreen.current = screen;
    delete shell.dataset.sourceRequest;
    for (const dialog of shell.querySelectorAll("dialog[open]")) dialog.close();
    const input = shell.querySelector("#chat-input");
    if (input) {
      input.value = draft.current;
      fitComposer(input);
    }
    if (screen === "welcome") {
      lastHistoryRequest.current = null;
      historyTarget.current = "";
      follow.current = true;
      if (clearRequested.current) shell.querySelector("#chat-input")?.focus({ preventScroll: true });
      clearRequested.current = false;
      draftSubmitted.current = false;
      const diagram = shell.querySelector(".welcome-diagram");
      const reducedMotion = matchMedia("(prefers-reduced-motion: reduce)");
      let inView = false;
      const drawWhenVisible = () => {
        if (!diagram || !inView || welcomeDrawn.current || document.visibilityState !== "visible") return;
        welcomeDrawn.current = true;
        diagram.dataset.draw = reducedMotion.matches ? "done" : "play";
      };
      const settle = () => {
        if (diagram?.dataset.draw === "play") diagram.dataset.draw = "done";
      };
      const observer = new IntersectionObserver(([entry]) => {
        inView = entry.isIntersecting;
        drawWhenVisible();
        if (welcomeDrawn.current) observer.disconnect();
      });
      if (diagram && !welcomeDrawn.current) {
        observer.observe(diagram);
        document.addEventListener("visibilitychange", drawWhenVisible);
      }
      diagram?.addEventListener("animationend", settle);
      reducedMotion.addEventListener("change", settle);
      return () => {
        document.removeEventListener("visibilitychange", drawWhenVisible);
        observer.disconnect();
        diagram?.removeEventListener("animationend", settle);
        reducedMotion.removeEventListener("change", settle);
      };
    } else if (changed && draftSubmitted.current && screen === "study") {
      shell.querySelector("#chat-messages")?.focus({ preventScroll: true });
    } else if (changed && screen === "graph") {
      shell.querySelector('[data-workspace="graph"]')?.focus({ preventScroll: true });
    }
    draftSubmitted.current = false;
  }, [screen]);

  useLayoutEffect(() => {
    if (screen === "graph" || lastGraphDraft.current === graphDraftNonce) return;
    lastGraphDraft.current = graphDraftNonce;
    if (!graphDraft) return;
    const input = root.current.querySelector("#chat-input");
    if (!input) return;
    draft.current = input.value = graphDraft;
    fitComposer(input);
    input.focus({ preventScroll: true });
    input.setSelectionRange(input.value.length, input.value.length);
  }, [screen, graphDraft, graphDraftNonce]);

  useLayoutEffect(() => {
    if (screen !== "study" || lastGraphSource.current === graphSourceNonce) return;
    lastGraphSource.current = graphSourceNonce;
    openGraphSource.current?.();
  }, [screen, graphSourceNonce]);

  useLayoutEffect(() => {
    if (screen !== "study") return;
    const thread = root.current.querySelector("#chat-messages");
    const content = thread?.firstElementChild;
    if (!content) return;
    automaticScrollTop.current = null;
    const scrollExtent = () => thread.scrollHeight - thread.clientHeight;
    let observedExtent;
    let historyFrame;
    const trackHistory = (atEnd = false) => {
      const prompts = content.querySelectorAll(".study-turn > .user-message");
      if (!prompts.length) return;
      if (atEnd) {
        selectHistoryTarget(prompts[prompts.length - 1].id);
        return;
      }
      const readingLine = thread.getBoundingClientRect().top + 35;
      let current = prompts[0];
      for (const prompt of prompts) {
        if (prompt.getBoundingClientRect().top > readingLine) break;
        current = prompt;
      }
      selectHistoryTarget(current.id);
    };
    const scrollToEnd = () => {
      observedExtent = scrollExtent();
      if (follow.current) {
        scrollToBottom(thread);
        trackHistory(true);
      }
    };
    const trackPosition = () => {
      // A queued scroll event may arrive after the next token changes scrollHeight.
      const extent = scrollExtent();
      if (extent !== observedExtent || thread.scrollTop === automaticScrollTop.current) return;
      automaticScrollTop.current = null;
      follow.current = extent - thread.scrollTop < 24;
      cancelAnimationFrame(historyFrame);
      historyFrame = requestAnimationFrame(() => {
        if (thread.scrollTop !== automaticScrollTop.current) trackHistory(follow.current);
      });
    };
    const stopOnWheelUp = (event) => {
      if (!event.ctrlKey && event.deltaY < 0) follow.current = false;
    };
    scrollToEnd();
    if (!follow.current) trackHistory();
    const observer = new ResizeObserver(scrollToEnd);
    observer.observe(thread);
    observer.observe(content);
    thread.addEventListener("scroll", trackPosition, { passive: true });
    thread.addEventListener("wheel", stopOnWheelUp, { passive: true });
    return () => {
      cancelAnimationFrame(historyFrame);
      observer.disconnect();
      thread.removeEventListener("scroll", trackPosition);
      thread.removeEventListener("wheel", stopOnWheelUp);
    };
  }, [screen]);

  useLayoutEffect(() => {
    if (screen !== "study" || !scrollTarget || lastHistoryRequest.current === scrollNonce) return;
    const thread = root.current.querySelector("#chat-messages");
    const target = document.getElementById(scrollTarget);
    if (!target || !thread?.contains(target)) return;
    lastHistoryRequest.current = scrollNonce;
    follow.current = false;
    revealInThread(thread, target, true);
    automaticScrollTop.current = thread.scrollTop;
    selectHistoryTarget(target.id);
    target.focus({ preventScroll: true });
  }, [screen, scrollNonce, scrollTarget]);

  return <div className="practice-app" data-screen={screen} ref={root}>{children}</div>;
}
