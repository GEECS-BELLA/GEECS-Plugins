# Wire payload fixtures

Raw values of GEECS device variables as they arrive over the device TCP
push stream, captured live 2026-09-21 on the reference deployment and
stored byte-for-byte (`str` values re-encoded with `latin-1`, the transport's
convention) under a `.bin` extension so no line-ending or end-of-file hook
ever rewrites them (the CSV payloads end in CRLF on the wire; the folder is
also `-text` in `.gitattributes`). They pin `geecs_data_utils.io.arrays`:

| file | device variable | format |
|---|---|---|
| `magspec_interpSpec.bin` | MagSpecCamera `interpSpec` (285 rows, magnet energized) | nested `[x,y]` pairs |
| `magspec_interpDiv.bin` | MagSpecCamera `interpDiv` (189 rows) | nested `[x,y]` pairs |
| `magspec_EnergyAxis.bin` | MagSpecCamera `EnergyAxis` (285 values) | CSV, CRLF-terminated |
| `magspec_AngleAxis.bin` | MagSpecCamera `AngleAxis` (189 values) | CSV, CRLF-terminated |
| `picoscope_scopeTrace_Channel0.bin` | PicoscopeV2 `scopeTrace.Channel0` (3000 samples, internal trigger, no beam) | LabVIEW flattened waveform |
