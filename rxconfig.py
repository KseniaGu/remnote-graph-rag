import reflex as rx
from reflex.plugins.sitemap import SitemapPlugin

config = rx.Config(
    app_name="app",
    assets_folder="app/assets",
    plugins=[SitemapPlugin()],
    title="AI Practice | Graph RAG",
    description="Practice AI with your personal knowledge graph",
    tailwind={
        "theme": {
            "extend": {
                "colors": {
                    "primary": {
                        "50": "#f0f9ff",
                        "100": "#e0f2fe",
                        "200": "#bae6fd",
                        "300": "#7dd3fc",
                        "400": "#38bdf8",
                        "500": "#0ea5e9",
                        "600": "#0284c7",
                        "700": "#0369a1",
                        "800": "#075985",
                        "900": "#0c4a6e",
                    },
                    "accent": {
                        "50": "#fdf4ff",
                        "100": "#fae8ff",
                        "200": "#f5d0fe",
                        "300": "#f0abfc",
                        "400": "#e879f9",
                        "500": "#d946ef",
                        "600": "#c026d3",
                        "700": "#a21caf",
                        "800": "#86198f",
                        "900": "#701a75",
                    },
                }
            }
        },
        "plugins": [],
    },
)
