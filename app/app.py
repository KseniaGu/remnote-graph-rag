import reflex as rx
from starlette.responses import JSONResponse

from app.components import sidebar, main_content, error_toast
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
                background: '#0b141a',
                mainBkg: '#1f2c34',
                primaryColor: '#1f2c34',
                primaryTextColor: '#e9edef',
                primaryBorderColor: '#2a3942',
                lineColor: '#8696a0',
                secondaryColor: '#202c33',
                tertiaryColor: '#0b141a',
                edgeLabelBackground: '#1f2c34',
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
        error_toast(),
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
        "https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css",
    ],
    head_components=[
        rx.el.script(src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"),
        rx.el.link(rel="icon", href=FAVICON_URL),
    ],
)

app.add_page(index, title=APP_PAGE_TITLE)


# Register health probe endpoint on Reflex's internal API instance
async def healthz():
    """Health probe endpoint for Cloud Run startup and liveness checks."""
    from backend.health import is_healthy
    if is_healthy():
        return JSONResponse({"status": "ok"})
    return JSONResponse({"status": "initializing"}, status_code=503)


app._api.add_route("/healthz", healthz, methods=["GET"])
