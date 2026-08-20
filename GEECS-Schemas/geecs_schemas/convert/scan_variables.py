"""Convert legacy scan-variable YAMLs to one :class:`ScanVariables` document.

Legacy dialects (two files per experiment under ``scan_devices/``):

``scan_devices.yaml``::

    single_scan_devices:
      JetZ (mm): U_ESP_JetXYZ:Position.Axis 3
      EMQ1 Current: U_EMQTripletBipolar:Current_Limit.Ch1

``composite_variables.yaml``::

    composite_variables:
      ALine_e_beam_angle_offset_x:
        mode: relative
        components:
        - device: U_S3H
          variable: Current
          relation: composite_var * 1
        - device: U_S4H
          variable: Current
          relation: composite_var * -2

Mapping:

- Each ``single_scan_devices`` entry becomes a :class:`ScanVariable` with
  ``kind: setpoint`` — the legacy engine's set-and-wait semantics.  Upgrading
  a real positioner to ``kind: motor`` (readback-tolerance polling) is a
  deliberate manual edit, not something the converter guesses.
- Each composite becomes a :class:`PseudoScanVariable`: component
  ``device``/``variable`` pairs fuse into ``Device:Variable`` targets, each
  ``relation`` string is carried **verbatim** into ``forward`` (numexpr in
  terms of ``composite_var``), and ``mode`` keeps its legacy meaning.
- A name defined in both files is a conflict and raises.
"""

from __future__ import annotations

from typing import Optional

from geecs_schemas.convert._common import (
    LegacyDocument,
    SchemaConversionError,
    load_legacy,
    require_known_keys,
)
from geecs_schemas.scan_variables import ScanVariables


def convert_scan_variables(
    scan_devices: Optional[LegacyDocument] = None,
    composite_variables: Optional[LegacyDocument] = None,
) -> ScanVariables:
    """Convert the legacy scan-variable pair into one :class:`ScanVariables`.

    Parameters
    ----------
    scan_devices : dict or Path or str, optional
        The legacy ``scan_devices.yaml`` document or its path.
    composite_variables : dict or Path or str, optional
        The legacy ``composite_variables.yaml`` document or its path.

    Returns
    -------
    ScanVariables
        The merged, validated catalog.

    Raises
    ------
    SchemaConversionError
        Naming any key, malformed relation, or name conflict that could not
        be mapped.
    """
    variables: dict[str, dict] = {}

    if scan_devices is not None:
        document = load_legacy(scan_devices)
        require_known_keys(document, ["single_scan_devices"], "scan_devices.yaml")
        for name, target in (document.get("single_scan_devices") or {}).items():
            if not isinstance(target, str):
                raise SchemaConversionError(
                    f"scan variable {name!r}: expected a 'Device:Variable' "
                    f"string, got {target!r}."
                )
            variables[name] = {"target": target, "kind": "setpoint"}

    if composite_variables is not None:
        document = load_legacy(composite_variables)
        require_known_keys(
            document, ["composite_variables"], "composite_variables.yaml"
        )
        for name, body in (document.get("composite_variables") or {}).items():
            context = f"composite variable {name!r}"
            if name in variables:
                raise SchemaConversionError(
                    f"{context}: also defined in scan_devices.yaml — rename "
                    "one of the two."
                )
            require_known_keys(body, ["components", "mode"], context)
            targets = []
            for component in body.get("components") or []:
                require_known_keys(
                    component, ["device", "variable", "relation"], context
                )
                relation = str(component["relation"])
                if "composite_var" not in relation:
                    raise SchemaConversionError(
                        f"{context}: relation {relation!r} does not mention "
                        "'composite_var' — cannot carry it over verbatim."
                    )
                targets.append(
                    {
                        "target": f"{component['device']}:{component['variable']}",
                        "forward": relation,
                    }
                )
            variables[name] = {
                "kind": "pseudo",
                "targets": targets,
                "mode": body.get("mode"),
            }

    return ScanVariables(variables=variables)
