"""Isolated Reflex browser fixture. Start it from a disposable app directory."""

import asyncio
import json
from pathlib import Path

import reflex as rx

from app.ui.knowledge_graph import (
    GRAPH_STYLESHEET,
    KnowledgeGraph,
    resolve_graph_action,
)
from backend.workflows.presentation import GraphPayload, ResponseSource

fixture = json.loads(Path("stage4-fixture.json").read_text())
GRAPHS = {
    name: GraphPayload.model_validate(data) for name, data in fixture["graphs"].items()
}
SOURCES = [ResponseSource.model_validate(data) for data in fixture["sources"]]


class GraphFixture(rx.State):
    graph: GraphPayload | None = GRAPHS["kv"]
    sources: list[ResponseSource] = SOURCES
    light: bool = False
    visible: bool = True
    mounted: bool = True
    tick: int = 0
    result: str = ""
    command_id: str = ""
    command_nonce: int = 0

    @rx.event
    def choose(self, fixture_name: str):
        self.graph = GRAPHS.get(fixture_name)
        self.command_id = ""

    @rx.event
    def swap_theme(self):
        self.light = not self.light

    @rx.event
    def toggle_visible(self):
        self.visible = not self.visible

    @rx.event
    def toggle_mount(self):
        self.mounted = not self.mounted

    @rx.event
    def select_kv(self):
        self.command_id = "kv"
        self.command_nonce += 1

    @rx.event
    async def stream(self):
        for _ in range(12):
            self.tick += 1
            yield
            await asyncio.sleep(0.03)

    @rx.event
    def prepare(self, artifact_id: str, concept_id: str):
        if self.graph and resolve_graph_action([self.graph], artifact_id, concept_id):
            self.result = f"explain {concept_id}"
        else:
            self.result = "rejected"

    @rx.event
    def reject(self):
        self.prepare("unowned-artifact", "kv")


def index():
    return rx.el.main(
        rx.el.header(
            rx.el.span("AI Practice"),
            rx.el.div(
                rx.el.select(
                    *[rx.el.option(name, value=name) for name in GRAPHS],
                    rx.el.option("empty", value="empty"),
                    id="fixture",
                    aria_label="Fixture",
                    on_change=GraphFixture.choose,
                    default_value="kv",
                ),
                rx.el.button("Theme", id="theme", on_click=GraphFixture.swap_theme),
                rx.el.button(
                    "Hide/show", id="visibility", on_click=GraphFixture.toggle_visible
                ),
                rx.el.button(
                    "Unmount/mount", id="mount", on_click=GraphFixture.toggle_mount
                ),
                rx.el.button("Stream", id="stream", on_click=GraphFixture.stream),
                rx.el.button(
                    "Select KV", id="command", on_click=GraphFixture.select_kv
                ),
                rx.el.button(
                    "Invalid callback", id="reject", on_click=GraphFixture.reject
                ),
                rx.el.output(GraphFixture.tick, id="tick"),
                rx.el.output(GraphFixture.result, id="result"),
                id="fixture-controls",
            ),
            id="fixture-header",
            display="flex",
            align_items="center",
            gap="16px",
            padding="0 14px",
            overflow="auto",
            height="64px",
        ),
        rx.el.div(
            rx.cond(
                GraphFixture.mounted,
                KnowledgeGraph.create(
                    graph=GraphFixture.graph,
                    sources=GraphFixture.sources,
                    theme=rx.cond(GraphFixture.light, "light", "dark"),
                    active=GraphFixture.visible,
                    selection_id=GraphFixture.command_id,
                    selection_nonce=GraphFixture.command_nonce,
                    on_prepare=GraphFixture.prepare,
                ),
            ),
            height="calc(100dvh - 64px)",
            id="stage4-host",
        ),
    )


app = rx.App(stylesheets=["/practice.css", GRAPH_STYLESHEET])
app.add_page(index)
