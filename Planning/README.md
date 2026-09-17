# Planning/

Design notes for work that is **live** — open questions, deferred items,
strategy that the code and the `CLAUDE.md` files do not yet record. It is
a semi-temporary store, useful while a change is being developed across
several branches and not yet settled anywhere durable.

## The rule

**Delete a plan in the PR that finishes the work it describes.** A design
doc is scaffolding; it comes down when the building stands. Anything in it
that is still load-bearing moves first:

| What | Where it goes |
|---|---|
| An invariant the code must keep | The docstring of the code that keeps it — one self-contained sentence |
| A rule that binds more than one package | The owning package's `CLAUDE.md`, or the root one for a cross-package invariant |
| Work that is genuinely still open | A GitHub issue |
| Why we chose this over the alternative | Nowhere — it is in the git history of the deleted file, and of the PRs that executed it |

That last row is the one people get wrong. Deleting a plan destroys
nothing: `git log --follow -- Planning/<file>.md` and `git show` reach
every word of it forever, and unlike a path in a docstring, a git
reference cannot dangle.

**Docstrings state the rule, not its provenance.** Do not write
`see Planning/x.md §4.4` — a reader then has to spend a read on a
several-hundred-line design doc, or risk missing the constraint. Say what
the rule *is*. Accumulated pointers are a tax paid on every future read.

## Audit log

`Planning/` is audited periodically and the audits are recorded in
`CONTRIBUTING.md` § "Planning/ is development scratch, not documentation".
