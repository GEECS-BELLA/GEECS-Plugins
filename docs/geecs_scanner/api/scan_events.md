# Scan Events

The typed event hierarchy emitted by the scan engine. Every state transition, every device command, every error, and every dialog request is delivered as an instance of one of these classes through the `on_event` callback registered with `ScanManager`.

The narrative description of when each event fires and how to consume them lives at [Architecture — Event vocabulary](../architecture.md#event-vocabulary). This page is the formal API. The vocabulary now lives in `geecs_bluesky.events`; `geecs_scanner.engine.scan_events` remains a re-export shim so existing imports keep working.

## ScanState

The lifecycle states a scan can be in. Inherits `str` so values serialise naturally to JSON and log messages.

::: geecs_bluesky.events.ScanState
    options:
      show_source: false
      show_root_heading: false
      show_root_toc_entry: true
      heading_level: 3

## Base event

::: geecs_bluesky.events.ScanEvent
    options:
      show_source: false
      show_root_heading: false
      show_root_toc_entry: true
      merge_init_into_class: true
      heading_level: 3

## Lifecycle events

::: geecs_bluesky.events.ScanLifecycleEvent
    options:
      show_source: false
      show_root_heading: false
      show_root_toc_entry: true
      merge_init_into_class: true
      heading_level: 3

## Step progress

::: geecs_bluesky.events.ScanStepEvent
    options:
      show_source: false
      show_root_heading: false
      show_root_toc_entry: true
      merge_init_into_class: true
      heading_level: 3

## Device command outcomes

::: geecs_bluesky.events.DeviceCommandEvent
    options:
      show_source: false
      show_root_heading: false
      show_root_toc_entry: true
      merge_init_into_class: true
      heading_level: 3

## Errors and restore failures

::: geecs_bluesky.events.ScanErrorEvent
    options:
      show_source: false
      show_root_heading: false
      show_root_toc_entry: true
      merge_init_into_class: true
      heading_level: 3

::: geecs_bluesky.events.ScanRestoreFailedEvent
    options:
      show_source: false
      show_root_heading: false
      show_root_toc_entry: true
      merge_init_into_class: true
      heading_level: 3

## Dialog requests

::: geecs_bluesky.events.ScanDialogEvent
    options:
      show_source: false
      show_root_heading: false
      show_root_toc_entry: true
      merge_init_into_class: true
      heading_level: 3
