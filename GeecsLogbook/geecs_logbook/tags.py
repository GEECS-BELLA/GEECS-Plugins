"""Tags: the words an author marked with ``#`` in a body.

A type button inserts a template whose prefill carries its tag (``#laser``);
a typist writes the same thing by hand. Either way the body is the truth
and this module reads the tags back out of it at save, so the store can
index them without anyone maintaining a category column.

What counts as a tag is deliberately narrow: ``#`` followed by a letter,
then letters, digits, ``-`` or ``_``, at most 32 characters, and not glued
to a preceding word character. ``#1`` (an issue number), ``C#``, a URL
fragment and a markdown heading (``# Title`` — the space rules it out) are
all left alone. Anything inside a code span or a fenced block is skipped:
``#include`` in a snippet is not a category.
"""

from __future__ import annotations

import re

_FENCE = re.compile(r"```.*?```|~~~.*?~~~", re.S)
_INLINE_CODE = re.compile(r"`[^`\n]*`")
_TAG = re.compile(r"(?<![\w#&/.])#([A-Za-z][\w-]{0,31})(?![\w-])")


def parse_tags(body_md: str) -> list[str]:
    """Return the tags in a body, lower-cased, first occurrence first."""
    text = _INLINE_CODE.sub(" ", _FENCE.sub(" ", body_md or ""))
    seen: dict[str, None] = {}
    for match in _TAG.finditer(text):
        seen.setdefault(match.group(1).lower(), None)
    return list(seen)
