import reflex as rx
from starlette.responses import JSONResponse

from app.components import sidebar, main_content
from app.state import AppState
from app.strings import APP_PAGE_TITLE
from app.styles import COLORS, GLOBAL_STYLES
from backend.configs.constants import FAVICON_URL


def index() -> rx.Component:
    """The main page of the application."""
    return rx.box(
        rx.html(f"<style>{GLOBAL_STYLES}</style>"),
        rx.script("""
(function() {
    function renderMermaid() {
        if (typeof mermaid === 'undefined') return;
        document.querySelectorAll('pre code.language-mermaid:not([data-mermaid-done])').forEach(function(block) {
            block.setAttribute('data-mermaid-done', '1');
            var source = block.textContent || block.innerText;
            var wrapper = document.createElement('div');
            wrapper.className = 'mermaid-wrapper';
            var div = document.createElement('div');
            div.className = 'mermaid';
            div.textContent = source;
            wrapper.appendChild(div);
            var pre = block.closest('pre');
            if (pre) pre.replaceWith(wrapper);
            mermaid.run({ nodes: [div] });
        });
    }

    function initMermaid() {
        if (typeof mermaid === 'undefined') { setTimeout(initMermaid, 200); return; }
        mermaid.initialize({
            startOnLoad: false,
            theme: 'dark',
            themeVariables: {
                darkMode: true,
                background: '#061216',
                mainBkg: '#10242b',
                primaryColor: '#10242b',
                primaryTextColor: '#f3fbff',
                primaryBorderColor: '#24434d',
                lineColor: '#7f99a4',
                secondaryColor: '#0d1d23',
                tertiaryColor: '#061216',
                edgeLabelBackground: '#10242b',
                fontSize: '14px'
            },
            securityLevel: 'loose'
        });
        renderMermaid();
        new MutationObserver(function(mutations) {
            if (mutations.some(function(m) { return m.addedNodes.length > 0; })) {
                setTimeout(renderMermaid, 150);
            }
        }).observe(document.body, { childList: true, subtree: true });
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initMermaid);
    } else {
        initMermaid();
    }
})();
"""),
        sidebar(),
        main_content(),
        style={
            "min_height": "100vh",
            "background": COLORS["bg_dark"],
        },
    )


# Create the app
app = rx.App(
    theme=rx.theme(
        appearance="dark",
        has_background=True,
        radius="medium",
        accent_color="sky",
    ),
    stylesheets=[
        "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap",
        "https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap",
        "https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css",
    ],
    head_components=[
        rx.el.script(src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"),
        rx.el.link(rel="icon", href=FAVICON_URL),
    ],
)

app.add_page(index, title=APP_PAGE_TITLE, on_load=AppState.initialize_session)


# Register health probe endpoint on Reflex's internal API instance
async def healthz():
    """Health probe endpoint for Cloud Run startup and liveness checks."""
    from backend.health import is_healthy
    if is_healthy():
        return JSONResponse({"status": "ok"})
    return JSONResponse({"status": "initializing"}, status_code=503)


app._api.add_route("/healthz", healthz, methods=["GET"])
