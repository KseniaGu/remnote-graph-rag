import reflex as rx
from app.styles import COLORS, GLOBAL_STYLES
from app.components import sidebar, main_content, error_toast


def index() -> rx.Component:
    """The main page of the application."""
    return rx.box(
        rx.html(f"<style>{GLOBAL_STYLES}</style>"),
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
    ],
)

app.add_page(index, title="AI Practice | Graph RAG")
