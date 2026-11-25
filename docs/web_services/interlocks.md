# Interlocks

The interlock server watches device values, and status packets from
programs outside GEECS, against a configuration you choose, and tells
**Master Control** whether it is safe. It only reports: what a trip stops is
Master Control's decision.

It runs headless, as a systemd unit on the services box. Anyone on the
network watches it on its page, where an operator also arms the settings
checks and reloads the configuration; configurations are edited in the
[Configurator](configurator.md).

## Start it

```bash
poetry run geecs-interlocks serve --config main
```

`main` is a configuration name: the file `interlocks/main.yaml` in the
experiment's folder of the configs repository. Without `--config` the
server lists the saved configurations and stops. SIGTERM or Ctrl-C stops
it; the log goes to stderr. That is for a test by hand: in the lab the
server runs as the `geecs-interlocks` unit (one per experiment, restarted
on its own if it stops or its evaluation stalls); its maintainer's notes
are [GEECS-Interlocks/DEPLOYMENT.md](https://github.com/GEECS-BELLA/GEECS-Plugins/blob/master/GEECS-Interlocks/DEPLOYMENT.md).

| Flag | What it does |
|---|---|
| `--config NAME` | Run the configuration `interlocks/NAME.yaml` |
| `--interlocks FILE` | Run a configuration file from anywhere instead |
| `--experiment NAME` | The experiment, when `config.ini` names none |
| `--arm-at-start` | Arm the settings checks with the first values that arrive (by hand only; refused under systemd, see below) |
| `--no-web` | Do not serve the web page |
| `--fake-devices` | Fake GEECS devices on this computer: try everything without the lab |

Where the server listens comes from `~/.config/geecs_python_api/config.ini`:

```ini
[interlocks]
; Master Control's stream
status_host = 0.0.0.0
status_port = 5001
; the web page
web_host = 0.0.0.0
web_port = 8600
; external (UDP) signals
external_host = 0.0.0.0
```

Comments go on their own line: a comment after a value becomes part of it.
With none of these set, everything listens on this computer only.

## How it fits with the Control Room

The interlocks do not replace the [Control Room](control_room.md): they
answer one narrower question, "is it safe to run?", for Master Control, and
keep answering it when the browser surfaces are down.

| | Control Room | Archiver | Interlocks |
|---|---|---|---|
| Question | What is every device doing, and what should I change? | What did a value do over hours or days? | Is it safe to run, right now? |
| Read by | Operators, in a browser | Operators and analysis, after the fact | Master Control every 0.1 s; operators on the page |
| Acts on devices | Yes: operators set values through it | No | No: it only reports |
| Reaches devices through | Canopy and its GEECS module | The CA gateway | Its own `geecs_core` subscriptions, plus UDP from programs outside GEECS |

- **Independent path.** The interlocks read devices directly, so an outage
  of Canopy, the gateway or a browser does not blind Master Control.
- **Rules, not displays.** An interlock combines checks across devices and
  outside programs, with hysteresis, settle times and armed settings; the
  Control Room shows each device and leaves the judgment to the operator.
- **Fail-safe.** The Control Room flags a device that is not live; the
  interlocks turn silence, a disconnect or an unreadable value into a trip
  Master Control acts on.

## What it watches

A configuration is a list of **interlocks**. Each interlock holds one or
more **checks** and is unsafe while **any** of them is, so several checks
can be reported under one name (a camera's saturation and its pointing,
say). Master Control sees one line per interlock.

| Check | Unsafe when |
|---|---|
| Threshold | A variable crosses a limit (`>`, `>=`, `<`, `<=`, `==`, `!=`) |
| Alignment | A variable is the tolerance or further from a target variable |
| Stale | A device disconnects, or (with a maximum age) stops publishing |
| External signal | A program outside GEECS sends a UDP packet that does not start with `safe`, until it sends one that does (from its `sender` address, when one is set) |
| Settings unchanged | A device's settings move from the values recorded when it was armed |

An interlock is also unsafe before anything has been heard from its
devices, while a device is disconnected, when a device has sent nothing for
the configuration's `stale_after_s` (5 s unless changed), when a value is
not a number, and while it is **settling**: clear again after a trip but
not yet clear for its settle time.

A packet whose **first word** is `safe` (any case) means it is OK to fire;
whatever follows changes nothing (`safe stage1 parked`): it is shown in the
alarm reason of an unsafe packet and logged when the state changes. Send
plain text: a packet starting with a byte-order mark or another control
character never reads safe. Every other packet is unsafe: `stage1 safe`,
`not safe`, `moving` or an empty packet all trip the signal, so a sender
needs no special word for a hazard, only to stop starting with `safe`. A
sender that also sends keepalive or status packets trips the signal with
each one unless they too start with `safe`. Set the check's `sender` to the sending computer's address so
no other host can clear the signal: a packet from anywhere else counts as
unsafe.

!!! warning "Arming settings"
    A **settings unchanged** check is unsafe until it is armed, and arming
    records whatever values the device reports at that moment. Arm from the
    web page (below) once you know the settings are right. Every start of
    the server and every reload leaves the checks unarmed: the unit never
    arms on its own, since a restart would arm a setting someone had just
    changed. `--arm-at-start` (arm the first values that arrive) is for a
    server started by hand, and the server refuses it under systemd.

## Edit a configuration

Configurations are the `interlocks` kind in the
[Configurator](configurator.md), tier **maintainer**: a change opens a pull
request on the configs repository, so a new limit is reviewed before it
runs. Once it is merged, press **Reload** on the web page (or restart the
unit) to run it; every interlock then waits for fresh values before
reporting safe, and the settings checks must be armed again. A file that
does not validate is refused and the running interlocks are kept.

## The web page

`http://<server>:<port>/` (the port is the site's; 8600 on a computer of
your own) shows the verdict and the interlocks to anyone on the network.
It has two actions, both on the server, never on the experiment.

![The web page: the verdict, the interlocks, and the inspector with one trend per check against its limit](assets/interlocks_03_web.png)

Click an interlock to see each check's last five minutes against its limit,
and its history. The address keeps the selection
(`?interlock=pointing%20camera`), so you can send someone the link.

### Arm and Reload

Enter your name in **operator** at the top right first (the browser
remembers it); an action without it is refused. Each asks you to confirm,
is sent once, and shows the server's answer under the verdict. The server's
log records who did what.

| Button | What it does | Refused when |
|---|---|---|
| **Arm all** (shown only when the configuration has a settings check) | Records what every settings check sees now as the values to hold | A setting has no value yet or its device is disconnected (nothing is armed then) |
| **Arm this interlock** (in the inspector, for an interlock that holds settings) | The same, for that interlock only | As above |
| **Reload** | Re-reads the running configuration file; every interlock waits for fresh values and every settings check is unarmed again | The file does not validate: the running interlocks are kept |

!!! warning "No signal"
    If the page stops hearing from the server for 2 s it says
    **No signal**, shows the last verdict with its time, and greys every
    state. It never keeps showing a verdict it is no longer receiving.

## What Master Control receives

A message every 0.1 s on the stream port, holding a single keyword:

```text
SAFE | OK: chamber vacuum | OK: pointing camera
WARNING! TRIPPED | TRIPPED: pointing camera - MaxCounts=4113 (limit: >=4000) | OK: chamber vacuum
```

If the server's evaluation ever stops (one second without a new
evaluation), every frame becomes a trip, `TRIPPED: interlock server -
evaluation stopped N s ago`, until it recovers. Master Control is never
sent an old verdict.

Master Control fires a shot on a **new** frame holding `SAFE`, and treats a
missing stream as unsafe. So `SAFE` appears only at the front of an
all-clear frame, never beside a tripped interlock; the original server's
`SAFE: <name>` per interlock is gone for that reason. Master Control finds
the server through the GEECS database, which holds its address and port.

`poetry run geecs-interlocks listen <server>:5001` prints what it receives.

## The scanner's shot gate

The scanner can wait for the interlocks too, without Master Control. With
`GEECS_INTERLOCKS_GATE_PORT` set (5002 in the example site.env), the server
also serves the shot gate on that port. A trigger profile names it in its
`shot_gate` field (`host`: the server, `port`: the gate port), and the scan
worker then holds every strict single shot until the interlocks say safe on
readings that arrived after the previous shot. A shot that waits longer
than the profile's `hold_pause_s` pauses the scan.

`/api/status` counts the gate's readers in `gate_clients`.

## When the server stops

The unit restarts it within seconds if it exits, and also if its
evaluation stalls: the evaluation feeds systemd's watchdog, so a server
that is running but no longer judging is restarted too. Meanwhile Master
Control has no stream, which it treats as unsafe, and reconnects on its
own. After the restart, arm the settings checks again.

## Try it on any computer

```bash
poetry install
poetry run geecs-interlocks serve --fake-devices --arm-at-start --interlocks examples/demo_interlocks.yaml
```

The fake devices are real GEECS device servers on this computer, read the
way the lab's are; the demo also sends its own UDP packets and moves some
settings now and then, so every state appears within a couple of minutes.
