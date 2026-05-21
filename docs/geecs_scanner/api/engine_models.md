# Engine Configuration Models

These are the typed Pydantic models that live at the GUI/engine boundary. They're the validated shape of every scan request — the GUI builds them, the engine consumes them. If you're scripting a scan, writing a custom evaluator, or building an alternate UI, these are the models you'll construct or read.

## Scan execution

The top-level configuration for a scan run.

::: geecs_scanner.engine.models.scan_execution_config.ScanExecutionConfig
    options:
      show_source: false
      show_root_heading: false
      show_root_toc_entry: true
      merge_init_into_class: true
      heading_level: 3
      filters: ["!^_"]
      show_signature_annotations: true

## Scan options

Engine-level execution options (rep rate, time-sync settings, file-save destination).

::: geecs_scanner.engine.models.scan_options.ScanOptions
    options:
      show_source: false
      show_root_heading: false
      show_root_toc_entry: true
      merge_init_into_class: true
      heading_level: 3
      filters: ["!^_"]
      show_signature_annotations: true

## Save device configuration

The schema that backs the save-element YAML files. Documented narratively at [Save Elements](../save_elements.md); this is the formal model.

::: geecs_scanner.engine.models.save_devices.SaveDeviceConfig
    options:
      show_source: false
      show_root_heading: false
      show_root_toc_entry: true
      merge_init_into_class: true
      heading_level: 3
      filters: ["!^_"]
      show_signature_annotations: true

::: geecs_scanner.engine.models.save_devices.DeviceConfig
    options:
      show_source: false
      show_root_heading: false
      show_root_toc_entry: true
      merge_init_into_class: true
      heading_level: 3
      filters: ["!^_"]
      show_signature_annotations: true

## Action steps

The five step types that make up `setup_action` and `closeout_action` sequences. Documented narratively at [Save Elements — Action Sequences](../save_elements.md#action-sequences).

::: geecs_scanner.engine.models.actions.WaitStep
    options:
      show_source: false
      show_root_heading: false
      show_root_toc_entry: true
      heading_level: 4

::: geecs_scanner.engine.models.actions.SetStep
    options:
      show_source: false
      show_root_heading: false
      show_root_toc_entry: true
      heading_level: 4

::: geecs_scanner.engine.models.actions.GetStep
    options:
      show_source: false
      show_root_heading: false
      show_root_toc_entry: true
      heading_level: 4

::: geecs_scanner.engine.models.actions.ExecuteStep
    options:
      show_source: false
      show_root_heading: false
      show_root_toc_entry: true
      heading_level: 4

::: geecs_scanner.engine.models.actions.RunStep
    options:
      show_source: false
      show_root_heading: false
      show_root_toc_entry: true
      heading_level: 4

::: geecs_scanner.engine.models.actions.ActionSequence
    options:
      show_source: false
      show_root_heading: false
      show_root_toc_entry: true
      heading_level: 4
