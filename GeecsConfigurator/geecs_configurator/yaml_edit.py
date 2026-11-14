"""Write a draft into a YAML file by changing only the text of what changed.

A save must not reformat the file it saves: the configs repository is
written by hand and reviewed in git, so a re-dump (every folded block made
a quoted flow string, every wrapped scalar re-wrapped) turns a four-line
edit into a fifty-line diff. :func:`patch` keeps the stored text and
splices into it: it composes the file with PyYAML (the resolver the store
reads with, so a bare ``OFF:`` is the boolean the store reads), walks the
nodes against the draft, and replaces, inserts or deletes the text of
the nodes that differ, and only those. Untouched bytes are never
re-emitted, so their scalar style, quotes, comments, blank lines,
indentation, wrap and line endings stay as they are.

Rules:

- A node is untouched when its value as served (a YAML date as its text)
  equals the draft's, type-strict; with ``strict_dates`` a date or a
  datetime the draft holds as text is a change, written quoted (the save's
  served-form rule, when the file as stored would not validate).
- A mapping's key matches a draft key by its value, or for a key that is
  not text (YAML 1.1's ``OFF:``, a number) by its spelling in the file.
- A list is aligned item by item (``difflib``), so an insertion in the
  middle inserts one item and moves no other.
- A new or changed value is written in the file's own manner: a
  multi-line string as a literal block, a new list item as a flow
  collection when its siblings are, a new block at the file's indentation,
  new lines in the file's line ending.
- Anything it cannot place exactly (a merge key, a change inside an
  anchored node that other nodes alias, a duplicate key, a complex key, a
  document that is not a mapping) is :class:`Unpatchable`; the caller then
  writes the file whole. The caller checks that the text reads back as
  the draft before it is committed.
"""

from __future__ import annotations

import datetime as dt
import difflib
import json
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import yaml
from pydantic_core import to_jsonable_python

from geecs_configurator.store import same

#: A JSON value as the draft holds it (``.inf``/``.nan`` restored as floats).
Json = Any  # Any: JSON, untyped by nature

_MERGE = "tag:yaml.org,2002:merge"
_TIMESTAMP = "tag:yaml.org,2002:timestamp"
_WIDE = 1 << 30


class Unpatchable(ValueError):
    """The draft cannot be spliced into this file exactly; write it whole instead."""


def served(value: Json) -> Json:
    """A loaded YAML value as the read route serves it: dates as text, non-text keys as JSON spells them, floats kept."""
    if isinstance(value, dict):
        return {k if isinstance(k, str) else to_jsonable_python(k) if not isinstance(k, bool) else str(k).lower(): served(v) for k, v in value.items()}  # fmt: skip
    if isinstance(value, list):
        return [served(v) for v in value]
    if isinstance(value, (dt.date, dt.datetime, dt.time, bytes)):
        return to_jsonable_python(value)
    return value


# ------------------------------------------------------------------ rendering


class _BlockDumper(yaml.SafeDumper):
    """Block style; a multi-line string as a literal block; a list under a key indented or not, as the file does."""

    indent_lists = True

    def increase_indent(self, flow: bool = False, indentless: bool = False) -> None:
        """Indent a list under a key unless the file writes its lists flush with their keys."""
        return super().increase_indent(flow, False if self.indent_lists else indentless)


def _str(dumper: yaml.SafeDumper, data: str) -> yaml.ScalarNode:
    style = "|" if "\n" in data.rstrip("\n") or data.endswith("\n") else None
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style=style)


_BlockDumper.add_representer(str, _str)


class _FlowDumper(yaml.SafeDumper):
    """Flow style; a string with a line break double-quoted with escapes, so it stays on one line."""


def _one_line_str(dumper: yaml.SafeDumper, data: str) -> yaml.ScalarNode:
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style='"' if "\n" in data or "\r" in data else None)  # fmt: skip


_FlowDumper.add_representer(str, _one_line_str)


def _flow(value: Json) -> str:
    """One value on one line, valid in flow and block context alike."""
    text = yaml.dump([value], Dumper=_FlowDumper, default_flow_style=True, sort_keys=False, width=_WIDE, allow_unicode=True)  # fmt: skip
    return text.strip()[1:-1]


def _plain(value: Json) -> str:
    """A scalar on one line as a block value (a ``:`` inside it needs no quotes there, unlike in flow)."""
    text = yaml.dump(value, Dumper=_FlowDumper, default_flow_style=False, width=_WIDE, allow_unicode=True)  # fmt: skip
    return text.removesuffix("\n").removesuffix("\n...").strip()


@dataclass
class _Style:
    """The file's manner, for text that is new."""

    newline: str
    indent: int  # a nested block mapping's indentation
    list_offset: int  # a list's dashes, past its key's column (0 or the indent)

    def block(self, value: Json) -> list[str]:
        """``value`` as block lines at column 0 (no line endings)."""
        dumper = type("_D", (_BlockDumper,), {"indent_lists": self.list_offset > 0})
        text = yaml.dump(value, Dumper=dumper, default_flow_style=False, sort_keys=False, allow_unicode=True, width=_WIDE, indent=self.indent)  # fmt: skip
        return text.rstrip("\n").split("\n")


def _pad(lines: list[str], col: int) -> list[str]:
    return [(" " * col + line) if line else line for line in lines]


# ------------------------------------------------------------------ the text


@dataclass
class _Edit:
    start: int
    end: int
    text: str


class _Patch:
    def __init__(self, text: str, strict_dates: bool):
        self.text, self.strict_dates = text, strict_dates
        self.loader = yaml.SafeLoader(text)
        try:
            self.root = self.loader.get_single_node()
        except yaml.YAMLError as exc:
            raise Unpatchable(f"the file does not compose: {exc}") from exc
        if not isinstance(self.root, yaml.MappingNode):
            raise Unpatchable("the document is not a mapping")
        self.shared = self._shared(self.root)
        crlf = text.count("\r\n")
        self.style = _Style(
            newline="\r\n" if crlf * 2 > text.count("\n") else "\n",
            indent=self._detect_indent(),
            list_offset=self._detect_list_offset(),
        )
        self.edits: list[_Edit] = []

    # -------------------------------------------------------------- reading

    @staticmethod
    def _shared(root: yaml.Node) -> set[int]:
        """The nodes reached more than once (an anchor and its aliases)."""
        seen: set[int] = set()
        twice: set[int] = set()
        stack = [root]
        while stack:
            node = stack.pop()
            if id(node) in seen:
                twice.add(id(node))
                continue
            seen.add(id(node))
            if isinstance(node, yaml.MappingNode):
                stack.extend(x for pair in node.value for x in pair)
            elif isinstance(node, yaml.SequenceNode):
                stack.extend(node.value)
        return twice

    def _walk_pairs(self) -> Iterator[tuple[yaml.Node, yaml.Node]]:
        stack = [self.root]
        while stack:
            node = stack.pop()
            if isinstance(node, yaml.MappingNode):
                for k, v in node.value:
                    yield k, v
                    stack.append(v)
            elif isinstance(node, yaml.SequenceNode):
                stack.extend(node.value)

    def _detect_indent(self) -> int:
        for k, v in self._walk_pairs():
            if isinstance(v, yaml.MappingNode) and not v.flow_style and v.value:
                step = v.value[0][0].start_mark.column - k.start_mark.column
                if step > 0:
                    return step
        return 2

    def _detect_list_offset(self) -> int:
        for k, v in self._walk_pairs():
            if isinstance(v, yaml.SequenceNode) and not v.flow_style and v.value:
                return max(0, self._dash(v.value[0]) - self._line_start(self._dash(v.value[0])) - k.start_mark.column)  # fmt: skip
        return 2

    def value(self, node: yaml.Node) -> Json:
        return self.loader.construct_object(node, deep=True)

    def _line_start(self, index: int) -> int:
        return self.text.rfind("\n", 0, index) + 1

    def _first_on_line(self, index: int) -> bool:
        return self.text[self._line_start(index) : index].strip() == ""

    def _content_end(self, node: yaml.Node) -> int:
        """The index just past the node's last character of content (never its trailing blank lines or comments)."""
        if (
            isinstance(node, (yaml.MappingNode, yaml.SequenceNode))
            and not node.flow_style
        ):
            last = node.value[-1][1] if isinstance(node, yaml.MappingNode) else node.value[-1]  # fmt: skip
            return self._content_end(last)
        end = node.end_mark.index
        if isinstance(node, yaml.ScalarNode) and node.style in ("|", ">"):
            while end > node.start_mark.index and self.text[end - 1] in " \t\r\n":
                end -= 1
        return end

    def _line_end(self, index: int) -> int:
        """The index past the line ending of the line holding ``index - 1`` (the end of the text when it has none)."""
        nl = self.text.find("\n", index)
        return len(self.text) if nl < 0 else nl + 1

    def _dash(self, item: yaml.Node) -> int:
        """The index of a block list item's ``-``."""
        i = item.start_mark.index - 1
        while i >= 0 and self.text[i] in " \t":
            i -= 1
        if i < 0 or self.text[i] != "-":
            # an empty item: its scalar sits past the dash, maybe on the next line
            j = self.text.rfind("-", 0, item.start_mark.index)
            if j < 0:
                raise Unpatchable("a list item's dash was not found")
            return j
        return i

    def _colon(self, key: yaml.Node) -> int:
        """The index just past a key's ``:``."""
        i = key.end_mark.index
        while i < len(self.text) and self.text[i] in " \t":
            i += 1
        if i >= len(self.text) or self.text[i] != ":":
            raise Unpatchable("a key's ':' was not found")
        return i + 1

    # -------------------------------------------------------------- the walk

    def untouched(self, node: yaml.Node, new: Json) -> bool:
        if not same(served(self.value(node)), new):
            return False
        return not (self.strict_dates and self._has_date_as_text(node, new))

    def _has_date_as_text(self, node: yaml.Node, new: Json) -> bool:
        if isinstance(node, yaml.ScalarNode):
            return isinstance(new, str) and isinstance(self.value(node), (dt.date, dt.datetime))  # fmt: skip
        if isinstance(node, yaml.MappingNode) and isinstance(new, dict):
            keys = self._keys(node)
            return any(self._has_date_as_text(v, new[k]) for k, (_, v) in keys.items() if k in new)  # fmt: skip
        if isinstance(node, yaml.SequenceNode) and isinstance(new, list):
            return any(self._has_date_as_text(v, n) for v, n in zip(node.value, new))
        return False

    def _keys(self, node: yaml.MappingNode) -> dict[str, tuple[yaml.Node, yaml.Node]]:
        """The mapping's pairs by the draft's spelling of each key."""
        out: dict[str, tuple[yaml.Node, yaml.Node]] = {}
        for k, v in node.value:
            if k.tag == _MERGE:
                raise Unpatchable("a merge key")
            if not isinstance(k, yaml.ScalarNode):
                raise Unpatchable("a complex key")
            key = self.value(k)
            name = key if isinstance(key, str) else k.value
            if name in out:
                raise Unpatchable(f"a duplicate key {name!r}")
            out[name] = (k, v)
        return out

    def walk(self, node: yaml.Node, new: Json, slot: tuple[int, int, int, bool] | None) -> None:  # fmt: skip
        """Make ``node``'s text say ``new``; ``slot`` is where the whole node may be rewritten (``None``: the root)."""
        if self.untouched(node, new):
            return
        if id(node) in self.shared:
            raise Unpatchable("a change inside an anchored node that is aliased")
        if isinstance(node, yaml.MappingNode) and isinstance(new, dict) and node.value and new and (slot is None or self._shares_a_key(node, new)):  # fmt: skip
            self._mapping(node, new)
        elif isinstance(node, yaml.SequenceNode) and isinstance(new, list) and node.value and new:  # fmt: skip
            self._sequence(node, new)
        elif slot is None:
            raise Unpatchable("the whole document changes shape")
        else:
            self._replace(node, new, slot)

    def _shares_a_key(self, node: yaml.MappingNode, new: dict[str, Json]) -> bool:
        """Whether the draft keeps any of the mapping's keys (none kept: the mapping is rewritten whole in its slot)."""
        return any(name in new for name in self._as_drafted(self._keys(node), new))

    # a slot: (start, end, column, flow) — the text that is the value, and the
    # column its owner (the key, or the list's dash) stands at; flow when the
    # value sits inside a flow collection

    def _value_slot(self, key: yaml.Node, value: yaml.Node, flow: bool) -> tuple[int, int, int, bool]:  # fmt: skip
        if flow:
            return value.start_mark.index, self._content_end(value), 0, True
        return self._colon(key), self._content_end(value), key.start_mark.column, False

    def _item_slot(self, item: yaml.Node, flow: bool) -> tuple[int, int, int, bool]:
        if flow:
            return item.start_mark.index, self._content_end(item), 0, True
        dash = self._dash(item)
        return dash + 1, self._content_end(item), dash - self._line_start(dash), False

    def _replace(self, node: yaml.Node, new: Json, slot: tuple[int, int, int, bool]) -> None:  # fmt: skip
        start, end, col, flow = slot
        if flow:
            self.edits.append(_Edit(start, end, _flow(new)))
            return
        in_list = self.text[start - 1] == "-"
        prefer_flow = isinstance(node, (yaml.MappingNode, yaml.SequenceNode)) and bool(node.flow_style) and bool(node.value)  # fmt: skip
        self.edits.append(_Edit(start, end, self._render_value(new, col, in_list, prefer_flow)))  # fmt: skip

    def _render_value(self, new: Json, col: int, in_list: bool, prefer_flow: bool) -> str:  # fmt: skip
        """The text after a key's ``:`` (or a list's ``-``) that says ``new``."""
        nl = self.style.newline
        multiline = isinstance(new, str) and "\n" in new
        if (not isinstance(new, (dict, list)) or not new or prefer_flow) and not multiline:  # fmt: skip
            return " " + (_flow(new) if isinstance(new, (dict, list)) else _plain(new))
        lines = self.style.block(new)
        if (
            multiline or in_list
        ):  # a literal block (its lines carry their own indent), or a collection on the dash's line
            pad = col if multiline else col + 2
            return " " + lines[0] + "".join(nl + line for line in _pad(lines[1:], pad))  # fmt: skip
        pad = col + (self.style.indent if isinstance(new, dict) else self.style.list_offset)  # fmt: skip
        return "".join(nl + line for line in _pad(lines, pad))

    def _mapping(self, node: yaml.MappingNode, new: dict[str, Json]) -> None:
        flow = bool(node.flow_style)
        keys = self._as_drafted(self._keys(node), new)
        pairs = list(keys.items())
        for name, (k, v) in pairs:
            if name in new:
                self.walk(v, new[name], self._value_slot(k, v, flow))
        gone = [i for i, (name, _) in enumerate(pairs) if name not in new]
        added = [name for name in new if name not in keys]
        if len(gone) == len(pairs) and not added:
            raise Unpatchable("every key removed")
        covered: set[int] = set()
        for i in gone:
            if i not in covered:
                covered |= self._delete_pair(node, pairs, i, flow, set(gone))
        if added:
            self._add_pairs(node, pairs, {n: new[n] for n in added}, flow)

    def _as_drafted(self, keys: dict[str, tuple[yaml.Node, yaml.Node]], new: dict[str, Json]) -> dict[str, tuple[yaml.Node, yaml.Node]]:  # fmt: skip
        """``keys`` with a key that is not text (YAML 1.1's ``off:``) named as the draft spells it, case aside.

        The model serves a bare ``off:`` as ``"OFF"``; matched by its
        spelling alone, an unedited ``off:`` would be removed and re-added.
        """
        out = {}
        for name, (k, v) in keys.items():
            if name not in new and not isinstance(self.value(k), str):
                drafted = [
                    d for d in new if d not in keys and d.lower() == name.lower()
                ]
                name = drafted[0] if len(drafted) == 1 else name
            out[name] = (k, v)
        return out

    def _delete_pair(self, node: yaml.MappingNode, pairs: list[tuple[str, tuple[yaml.Node, yaml.Node]]], i: int, flow: bool, gone: set[int]) -> set[int]:  # fmt: skip
        """Delete pair ``i``; the indices this deletion also removed (the run of gone keys after a list item's first)."""
        k, v = pairs[i][1]
        if id(v) in self.shared:
            raise Unpatchable("an anchored or aliased node removed")
        if flow:
            if i + 1 < len(pairs):
                self.edits.append(_Edit(k.start_mark.index, pairs[i + 1][1][0].start_mark.index, ""))  # fmt: skip
            elif i > 0:
                self.edits.append(_Edit(self._content_end(pairs[i - 1][1][1]), self._content_end(v), ""))  # fmt: skip
            else:
                raise Unpatchable("a flow mapping emptied")
            return set()
        if self._first_on_line(k.start_mark.index):
            self.edits.append(_Edit(self._line_start(k.start_mark.index), self._line_end(self._content_end(v)), ""))  # fmt: skip
            return set()
        # the first key on a list item's line ("- a: 1"): the next kept key takes its place on the dash's line
        kept = next((j for j in range(i + 1, len(pairs)) if j not in gone), None)
        if kept is None:
            raise Unpatchable("a list item's every key removed")
        for j in range(i + 1, kept):
            if id(pairs[j][1][1]) in self.shared:
                raise Unpatchable("an anchored or aliased node removed")
        self.edits.append(_Edit(k.start_mark.index, pairs[kept][1][0].start_mark.index, ""))  # fmt: skip
        return set(range(i + 1, kept))

    def _add_pairs(self, node: yaml.MappingNode, pairs: list[tuple[str, tuple[yaml.Node, yaml.Node]]], added: dict[str, Json], flow: bool) -> None:  # fmt: skip
        last_v = pairs[-1][1][1]
        at = self._content_end(last_v)
        if flow:
            text = "".join(f", {_flow(k)}: {_flow(v)}" for k, v in added.items())
            self.edits.append(_Edit(at, at, text))
            return
        col = pairs[0][1][0].start_mark.column
        siblings_flow = any(isinstance(v, (yaml.MappingNode, yaml.SequenceNode)) and v.flow_style for _, (_, v) in pairs)  # fmt: skip
        out = [" " * col + _flow(k) + ":" + self._render_value(v, col, False, siblings_flow and isinstance(v, (dict, list))) for k, v in added.items()]  # fmt: skip
        self._insert_lines(at, out)

    def _insert_lines(self, after: int, lines: list[str]) -> None:
        """Insert whole lines after the line holding index ``after - 1``."""
        nl = self.style.newline
        at = self._line_end(after)
        body = "".join(line + nl for line in lines)
        if at == len(self.text) and not self.text.endswith("\n"):
            body = nl + body.removesuffix(nl)
        self.edits.append(_Edit(at, at, body))

    def _sequence(self, node: yaml.SequenceNode, new: list[Json]) -> None:
        flow = bool(node.flow_style)
        items = node.value
        old_keys = [self._fingerprint(served(self.value(n))) for n in items]
        new_keys = [self._fingerprint(v) for v in new]
        ops = difflib.SequenceMatcher(None, old_keys, new_keys, autojunk=False).get_opcodes()  # fmt: skip
        if all(tag == "delete" for tag, *_ in ops if tag != "equal") and len([1 for tag, i1, i2, *_ in ops if tag == "delete" for _ in range(i2 - i1)]) == len(items):  # fmt: skip
            raise Unpatchable("every item removed")
        for tag, i1, i2, j1, j2 in ops:
            if tag == "equal":
                if self.strict_dates:
                    for i, j in zip(range(i1, i2), range(j1, j2)):
                        self.walk(items[i], new[j], self._item_slot(items[i], flow))
                continue
            pairs = min(i2 - i1, j2 - j1) if tag == "replace" else 0
            for d in range(pairs):
                self.walk(items[i1 + d], new[j1 + d], self._item_slot(items[i1 + d], flow))  # fmt: skip
            for i in range(i1 + pairs, i2):
                self._delete_item(items, i, flow)
            if j2 - j1 > pairs:
                self._insert_items(items, i1 + pairs, new[j1 + pairs : j2], flow)

    @staticmethod
    def _fingerprint(value: Json) -> str:
        return json.dumps(value, sort_keys=True, default=str, allow_nan=True)

    def _delete_item(self, items: list[yaml.Node], i: int, flow: bool) -> None:
        item = items[i]
        if id(item) in self.shared:
            raise Unpatchable("an anchored or aliased node removed")
        if flow:
            if i + 1 < len(items):
                self.edits.append(_Edit(item.start_mark.index, items[i + 1].start_mark.index, ""))  # fmt: skip
            elif i > 0:
                self.edits.append(_Edit(self._content_end(items[i - 1]), self._content_end(item), ""))  # fmt: skip
            else:
                raise Unpatchable("a flow list emptied")
            return
        dash = self._dash(item)
        if not self._first_on_line(dash):
            raise Unpatchable("a list item not first on its line")
        self.edits.append(_Edit(self._line_start(dash), self._line_end(self._content_end(item)), ""))  # fmt: skip

    def _insert_items(self, items: list[yaml.Node], before: int, values: list[Json], flow: bool) -> None:  # fmt: skip
        if flow:
            if before < len(items):
                at = items[before].start_mark.index
                self.edits.append(
                    _Edit(at, at, "".join(_flow(v) + ", " for v in values))
                )
            else:
                at = self._content_end(items[-1])
                self.edits.append(
                    _Edit(at, at, "".join(", " + _flow(v) for v in values))
                )
            return
        ref = items[min(before, len(items) - 1)]
        dash = self._dash(ref)
        if not self._first_on_line(dash):
            raise Unpatchable("a list item not first on its line")
        col = dash - self._line_start(dash)
        siblings_flow = any(isinstance(n, (yaml.MappingNode, yaml.SequenceNode)) and n.flow_style for n in items)  # fmt: skip
        lines = [" " * col + "-" + self._render_value(v, col, True, siblings_flow and isinstance(v, (dict, list))) for v in values]  # fmt: skip
        if before < len(items):
            at = self._line_start(dash)
            nl = self.style.newline
            self.edits.append(_Edit(at, at, "".join(line + nl for line in lines)))
        else:
            self._insert_lines(self._content_end(items[-1]), lines)

    # -------------------------------------------------------------- applying

    def apply(self) -> str:
        edits = sorted(self.edits, key=lambda e: (e.start, e.end), reverse=True)
        floor = len(self.text) + 1
        out = self.text
        for e in edits:
            if e.end > floor:
                raise Unpatchable("two edits overlap")
            out = out[: e.start] + e.text + out[e.end :]
            floor = e.start
        return out


def patch(text: str, draft: dict[str, Json], *, strict_dates: bool = False) -> str:
    """``text`` with only what ``draft`` changes in it rewritten (the module docstring); :class:`Unpatchable` when it cannot."""
    p = _Patch(text, strict_dates)
    p.walk(p.root, draft, None)
    return p.apply()
