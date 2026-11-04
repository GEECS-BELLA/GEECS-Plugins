# Screenshots: raw-frames R2, the camera page's raw mode

The real camera page (`/device/camera/Cam1`) over GeecsCanopy's own test
module (`geecs_canopy.testing.make_env`), the test-only `frame.raw` fixture
(`GeecsControlRoom/tests/raw_fixture.py`: 600×600, a dark floor of ~56 ± 3
counts, one hot pixel of 4000 at x 412, y 187) and a fake Hare socket
(`transform` = None). No device, no IOC. The fixture's tag never advances,
so the chip reads 0.5 then 0.0 new frames/s.

- `raw_auto_log_1to1_*`: Raw on, log scale, auto levels (47–65), 1:1. The
  percentile levels clip the lone hot pixel white with the floor's brightest
  0.1 %; the cursor readout sits on it (`counts 4000`).
- `raw_floor_to_full_log_1to1_*`: the same, white raised to the full scale
  (4095): the floor is snow, the hot pixel the one white pixel (inside the
  dashed analysis ROI).
- `raw_floor_to_full_log_16x_*`: 16× on the hot pixel: one image pixel.

Reproduce: `bench.py <GeecsControlRoom dir> <port>` (needs `wsproto` in the
env), headless Chrome with `--remote-debugging-port=<cdp>
--remote-allow-origins=*`, then `node shoot.mjs <port> <cdp> <outdir>`.
