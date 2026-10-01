# Web Services

Three pages in a browser cover the day-to-day work at the beamline: run a
scan, look at what it recorded, and write down what happened. Nothing is
installed on an operator machine. Each is a service on the worker host,
on its own port, so each gets its own bookmark:
`http://<worker-host>:<port>/`. The host and the units behind each
service are on the [fleet map](../platform/fleet_map.md).

Every page's header carries a **Docs** link back to its page here.

<div class="grid cards" markdown>

-   :material-camera-iris:{ .lg .middle } **GEECS Scanner** · port 8300

    ---

    Run scans. Pick a preset or compose a scan, start it, watch it shot
    by shot, pause or stop it; move a device, run an action plan, check
    shot-offset calibration.

    [:octicons-arrow-right-24: GEECS Scanner](scanner.md)

-   :material-magnify:{ .lg .middle } **Data Portal** · port 8200

    ---

    Look at what a scan recorded. Day → scan → metadata, scalar plots,
    2D grid maps, camera images and traces; run a saved analysis on a
    scan and edit its recipe.

    [:octicons-arrow-right-24: Data Portal](data_portal.md)

-   :material-notebook-edit:{ .lg .middle } **Logbook** · port 8400

    ---

    Write down what happened. The **scan log** is a day document over
    that day's scan folders; the **ops book** holds routine operations
    by month. Notes link to each other by permalink.

    [:octicons-arrow-right-24: Logbook](logbook.md)

</div>

The ports are the reference deployment's defaults; a site can move them.
The three services talk to each other only through links: the portal's
run page links to that scan in the logbook, and the portal's Plot tab can
send a figure into the scan's log entry.
