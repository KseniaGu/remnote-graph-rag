import reflex as rx

PRACTICE_LIBRARY = f"$/public{rx.asset('practice.jsx', shared=True)}"


class PracticeShell(rx.Component):
    library = PRACTICE_LIBRARY
    tag = "PracticeShell"

    screen: rx.Var[str]
    scroll_nonce: rx.Var[int]
    scroll_target: rx.Var[str]
    graph_draft: rx.Var[str]
    graph_draft_nonce: rx.Var[int]
    graph_source_nonce: rx.Var[int]


class MermaidDiagram(rx.Component):
    library = PRACTICE_LIBRARY
    tag = "MermaidDiagram"

    source: rx.Var[str]
