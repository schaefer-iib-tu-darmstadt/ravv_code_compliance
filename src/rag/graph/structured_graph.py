"""Deterministic clause graph of the FSAE rulebook and structure-aware retrieval for the compilation task.

Edges: HAS_SUBCLAUSE (dotted-id hierarchy) and CROSS_REFERENCES (clause ids cited in the text).
Retrieval: lexical topic seeds, plus their full sub-tree, plus one cross-reference hop.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import networkx as nx

ID_RE = re.compile(r"\b([A-Z]{1,3}(?:\.\d+)+)\b")
DOTTED_LEADER_RE = re.compile(r"\.{4,}")

REL_SUBCLAUSE = "HAS_SUBCLAUSE"
REL_CROSSREF = "CROSS_REFERENCES"


MARKER_CATALOG: dict[str, tuple[str, int, int]] = {
    "atx_heading": (r"^(?:#{1,6})\s+({ID})\b\s*(.*)$", 1, 2),
    "bold_id": (r"^\*\*({ID})\*\*\s*(.*)$", 1, 2),
    "fullbold": (r"^\*\*({ID})\s+(.*?)\*\*\s*$", 1, 2),
    "plain_line": (r"^({ID})\s+(\S.*)$", 1, 2),
}

_DEFAULT_ID_PATTERN = r"[A-Z]{1,3}(?:\.\d+)+"
_DEFAULT_MARKER_KINDS = ("atx_heading", "bold_id", "fullbold", "plain_line")
_WHITESPACE_RE = re.compile(r"\s+")


def canonicalize_id(cid: str) -> str:
    return _WHITESPACE_RE.sub("", cid)


@dataclass(frozen=True)
class ClauseGrammar:
    id_pattern: str = _DEFAULT_ID_PATTERN
    marker_kinds: tuple[str, ...] = _DEFAULT_MARKER_KINDS
    normalizations: tuple[tuple[str, str], ...] = ()
    skip_toc_dotted_leader: bool = True

    def compiled_markers(self) -> list[tuple[re.Pattern, int, int]]:
        out = []
        for kind in self.marker_kinds:
            template, id_group, title_group = MARKER_CATALOG[kind]
            out.append((re.compile(template.replace("{ID}", self.id_pattern)), id_group, title_group))
        return out

    def id_re(self) -> re.Pattern:
        return re.compile(r"\b(" + self.id_pattern + r")\b")


DEFAULT_GRAMMAR = ClauseGrammar()


def parse_clauses(md_text: str, grammar: ClauseGrammar | None = None) -> dict[str, dict]:
    grammar = grammar or DEFAULT_GRAMMAR
    for pat, repl in grammar.normalizations:
        md_text = re.sub(pat, repl, md_text)
    markers = grammar.compiled_markers()

    clauses: dict[str, dict] = {}
    current = None
    for line in md_text.splitlines():
        if grammar.skip_toc_dotted_leader and DOTTED_LEADER_RE.search(line):
            continue
        cid = title = None
        for rx, id_group, title_group in markers:
            m = rx.match(line)
            if m:
                cid, title = canonicalize_id(m.group(id_group)), m.group(title_group)
                break
        if cid is None:
            if current is not None:
                clauses[current]["body"].append(line)
            continue
        if cid not in clauses:
            clauses[cid] = {"title": title.strip(), "body": []}
        elif title.strip():
            clauses[cid]["body"].append(title.strip())
        current = cid

    for c in clauses.values():
        c["text"] = (c["title"] + " " + " ".join(c["body"])).strip()
    return clauses


def _nearest_existing_ancestor(cid: str, ids: set[str]) -> str | None:
    parts = cid.split(".")
    for i in range(len(parts) - 1, 1, -1):
        anc = ".".join(parts[:i])
        if anc in ids:
            return anc
    return None


def build_graph(clauses: dict[str, dict], id_re: re.Pattern | None = None) -> nx.MultiDiGraph:
    id_re = id_re or ID_RE
    ids = set(clauses)
    g = nx.MultiDiGraph()
    for cid, c in clauses.items():
        g.add_node(
            cid,
            title=c["title"],
            body=" ".join(c["body"]),
            text=c["text"],
            level=cid.count("."),
        )
    for cid in clauses:
        parent = _nearest_existing_ancestor(cid, ids)
        if parent is not None:
            g.add_edge(parent, cid, key=REL_SUBCLAUSE, type=REL_SUBCLAUSE)
    for cid, c in clauses.items():
        refs = {canonicalize_id(m) for m in id_re.findall(c["text"])}
        for ref in {r for r in refs if r in ids and r != cid}:
            g.add_edge(cid, ref, key=REL_CROSSREF, type=REL_CROSSREF)
    return g


def build_graph_from_markdown(
    md_path: str | Path, grammar: ClauseGrammar | None = None
) -> nx.MultiDiGraph:
    grammar = grammar or DEFAULT_GRAMMAR
    md_text = Path(md_path).read_text(encoding="utf-8")
    return build_graph(parse_clauses(md_text, grammar), id_re=grammar.id_re())


def _subclause_children(g: nx.MultiDiGraph, node: str):
    for _, child, key in g.out_edges(node, keys=True):
        if key == REL_SUBCLAUSE:
            yield child


def _crossref_targets(g: nx.MultiDiGraph, node: str):
    for _, tgt, key in g.out_edges(node, keys=True):
        if key == REL_CROSSREF:
            yield tgt


def descendants(g: nx.MultiDiGraph, node: str) -> set[str]:
    out: set[str] = set()
    stack = [node]
    while stack:
        cur = stack.pop()
        for child in _subclause_children(g, cur):
            if child not in out:
                out.add(child)
                stack.append(child)
    return out


def lexical_seeds(g: nx.MultiDiGraph, synonyms: list[str]) -> set[str]:
    pats = [re.compile(r"\b" + re.escape(s) + r"\b", re.IGNORECASE) for s in synonyms]
    return {
        n
        for n, data in g.nodes(data=True)
        if any(p.search(data.get("text", "")) for p in pats)
    }


def expand(
    g: nx.MultiDiGraph,
    seeds: set[str],
    *,
    use_subtree: bool = True,
    use_crossref: bool = True,
) -> set[str]:
    result = set(seeds)
    if use_subtree:
        for s in list(seeds):
            result |= descendants(g, s)
    if use_crossref:
        targets: set[str] = set()
        for n in list(result):
            targets.update(_crossref_targets(g, n))
        result |= targets
    return result


def topic_synonyms(question: str) -> list[str]:
    m = re.search(r"relevant to \`([^\`]+)\`", question)
    raw = m.group(1) if m else ""
    return [s.strip() for s in raw.split("/") if s.strip()]


def retrieve_ids(
    g: nx.MultiDiGraph,
    question: str,
    *,
    use_subtree: bool = True,
    use_crossref: bool = True,
) -> set[str]:
    seeds = lexical_seeds(g, topic_synonyms(question))
    return expand(g, seeds, use_subtree=use_subtree, use_crossref=use_crossref)


def format_answer(ids: set[str]) -> str:
    return ", ".join(sorted(ids))


