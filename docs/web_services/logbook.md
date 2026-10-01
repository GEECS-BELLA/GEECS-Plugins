# Logbook

The logbook is where people write down what happened. It has two books:

- the **scan log**, one page per day, built from that day's scan folders,
  where notes attach to a scan, sit between two scans, or cover the day;
- the **ops book**, one page per month, for routine operations that are
  not about a particular scan (a laser alignment, a vacuum pump swap).

It runs on the worker host on port **8400**
(`http://<worker-host>:8400/`), beside the [scanner](scanner.md) (8300)
and the [Data Portal](data_portal.md) (8200).

## The scan log

`/day/2026-09-11` is that day's document. Every scan folder that exists
for the day appears as a block with its number, mode, shot count, status,
start time, saved devices and description, read from the scan folder
when the page loads. Nothing has to be created first: a scan shows up in
the log because its folder exists, and the logbook itself stores none of
those facts, so they can never disagree with the data.

- The rail has a calendar (**Go to a day**), previous and next day, and
  a list of the day's scans to jump to.
- Each scan block folds; **Collapse all** / **Expand all** folds them
  together, and a day with many scans opens folded.
- **Ops book →** in the header goes to this month's ops book, and the
  day page says how many ops notes were written that day.

## The ops book

`/month/2026-09` lists every ops note of the month, newest day first.
Tags written in notes (`#laser`, `#vacuum`) become chips at the top;
click one to show only those notes, **× clear filter** to show them all.
The **Filter notes…** box searches the text. Each day links to that
day's scan log with **scan log →**.

The ops book reads only the logbook's own store, never the data share,
so it stays fast when the share is slow.

## Write a note

Both books use the same composer.

1. Pick where the note goes: a scan's block, the gap between two scans,
   the day (**Day intro**), or a day in the ops book.
2. Optionally start from a type button (the site's templates, such as
   *result*, *fault*, *laser*, *handover* or *maintenance*). A template
   carries its tag.
3. Write in markdown. Type `#tag` anywhere to tag the note. Paste or
   drop an image to attach it; paste a table from a spreadsheet and it
   becomes a markdown table.
4. Enter your name and press **Save**.

After saving, a note can be edited or deleted. Every change keeps the
previous version, and a deleted note is hidden rather than erased, so it
can be recovered. If two
people edit the same note, the second save is refused with the current
text rather than silently overwriting the first.

Notes written by an agent arrive as drafts; **Keep in log** makes one
part of the record.

## Link notes together

Every note's **Link** button copies its permalink, `/entry/<id>`. Paste
it into another note and it becomes a labelled link to that note. A
permalink opens whichever page shows the note, in either book.

Figures can also arrive from the Data Portal: its **send to scan log**
button posts a plot into that scan's entry.

## Where notes are kept

Notes live in the logbook's own database on the worker host, and each is
also written out as a markdown file under `{experiment}/logbook/` on the
data share, in the same year/month/day layout as the scan data but
outside the scan folders. That copy is readable without the logbook
running and is backed up with the share. The logbook never creates or
changes anything inside a scan folder.
