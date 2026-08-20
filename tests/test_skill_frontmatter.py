"""Pin that every repo skill under ``.claude/skills/`` can surface.

A skill whose ``SKILL.md`` frontmatter is malformed, whose ``name`` does not
match its directory, or whose ``description`` is missing silently drops out
of agent sessions — there is no error anywhere. This suite makes that
failure loud.
"""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SKILLS_DIR = REPO_ROOT / ".claude" / "skills"

FRONTMATTER_RE = re.compile(r"\A---\n(.*?)\n---\n", re.DOTALL)


def _skill_files() -> list[Path]:
    return sorted(SKILLS_DIR.glob("*/SKILL.md"))


def test_skills_directory_is_populated():
    """The repo ships skills; an empty glob means the layout moved."""
    assert _skill_files(), f"no */SKILL.md found under {SKILLS_DIR}"


def test_frontmatter_is_wellformed():
    """Each SKILL.md opens with a frontmatter block naming its directory."""
    for skill in _skill_files():
        text = skill.read_text(encoding="utf-8")
        match = FRONTMATTER_RE.match(text)
        assert match, f"{skill}: no leading '---' frontmatter block"
        frontmatter = match.group(1)

        name = re.search(r"^name:\s*(\S+)\s*$", frontmatter, re.MULTILINE)
        assert name, f"{skill}: frontmatter has no 'name:' line"
        assert name.group(1) == skill.parent.name, (
            f"{skill}: name '{name.group(1)}' != directory "
            f"'{skill.parent.name}' — the skill will not resolve"
        )

        description = re.search(r"^description:\s*(.+)$", frontmatter, re.MULTILINE)
        assert description, f"{skill}: frontmatter has no 'description:' line"
        # A folded scalar ('>' / '>-') needs at least one indented content line;
        # an inline scalar is its own content.
        if description.group(1).strip() in {">", ">-", "|", "|-"}:
            body = frontmatter[description.end() :]
            assert re.search(r"^\s+\S", body, re.MULTILINE), (
                f"{skill}: 'description:' folded block has no content"
            )
