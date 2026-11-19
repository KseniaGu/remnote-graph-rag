import reflex as rx
from starlette.responses import JSONResponse

from app.components import main_content
from app.practice_components import PracticeShell
from app.state import AppState
from app.strings import APP_PAGE_TITLE
from app.styles import GLOBAL_STYLES
from app.ui.knowledge_graph import GRAPH_STYLESHEET
from backend.configs.constants import FAVICON_URL


def index() -> rx.Component:
    """The main page of the application."""
    return PracticeShell.create(
        rx.html(f"<style>{GLOBAL_STYLES}</style>"),
        main_content(),
        screen=rx.cond(
            AppState.active_view == "graph",
            "graph",
            rx.cond(AppState.has_messages, "study", "welcome"),
        ),
        scroll_nonce=AppState.scroll_request_nonce,
        scroll_target=AppState.scroll_target_message_dom_id,
        graph_draft=AppState.graph_draft,
        graph_draft_nonce=AppState.graph_draft_nonce,
        graph_source_nonce=AppState.graph_source_nonce,
    )


# Create the app
app = rx.App(
    theme=rx.theme(
        has_background=False,
        radius="medium",
        accent_color="indigo",
    ),
    stylesheets=["/practice.css", GRAPH_STYLESHEET],
    head_components=[
        rx.el.meta(
            name="viewport",
            content="width=device-width, initial-scale=1, viewport-fit=cover",
        ),
        rx.el.script(
            data_practice_mermaid="true",
            src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js",
            defer=True,
        ),
        rx.el.link(rel="icon", href=FAVICON_URL),
    ],
)

app.add_page(index, title=APP_PAGE_TITLE, on_load=AppState.initialize_session)


# Register health probe endpoint on Reflex's internal API instance
async def healthz(request):
    """Health probe endpoint for Cloud Run startup and liveness checks."""
    from backend.health import is_healthy

    if is_healthy():
        return JSONResponse({"status": "ok"})
    return JSONResponse({"status": "initializing"}, status_code=503)


app._api.add_route("/healthz", healthz, methods=["GET"])
