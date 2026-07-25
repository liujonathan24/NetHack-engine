# Experiment 1 · run1 — status

Model `google/gemini-3-flash-preview` via Prime (team-billed). Tier `full_nle`
(uncapped), pinned Valkyrie, seeds 0-15, n=16, 150 turns, no compaction,
netplay skills. Banner-bleed fix `9548e24` applied.

## Cells

| cell | variant | status |
|---|---|---|
| B0 | uncompressed ASCII | LAUNCHED (task brfkjwnwv) |
| JSON | structured object | LAUNCHED (task boalvarf1) |
| TOON | compact structured | LAUNCHED (task bopbw5nqj) |
| IMG_TTY | tty-raster image (vision) | LAUNCHED (task b08xpcwqc) |
| IMG | pixel tileset (vision) | **DEFERRED — blocked** |

## IMG blocker

IMG (pixel tileset) render (`nethack_harness/prompt/image_render.py`) needs the
MiniHack `GlyphMapper` tileset, which requires the upstream `nle` package.
This repo uses its own NetHack fork (`nethack_core`), not pip `nle`, so `nle`
is absent. `minihack==1.0.2` was installed `--no-deps` (safe; the 4 running
cells don't import it) but its GlyphMapper still fails: `No module named 'nle'`.

Two ways to fill IMG later (neither blocks the 4-cell result):
1. Provide a canonical NetHack tileset sprite sheet and set `NETHACK_TILESET`
   (+ `NETHACK_TILESET_SIZE`) — the `_custom_tiles` path renders without nle.
2. Install `nle` (heavy C build) so MiniHack's built-in GlyphMapper works, then
   `bash tools/encoding_eval/launch_cell.sh IMG outputs/encoding_eval/run1/IMG 150`.

The vision axis is still represented by IMG_TTY, so the primary encoding
comparison (ASCII vs JSON vs TOON vs vision) is complete without IMG.
