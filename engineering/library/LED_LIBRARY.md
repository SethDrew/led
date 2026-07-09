# LED Library

Per-product facts about our physical LED hardware: wire color order and
measured power draw, one table row per product or test. Datasheet figures
(60 mA/pixel, "RGB") routinely disagree with what's on the wire — plan from
these numbers, not folklore.

## Color order

We get this wrong constantly. Source of truth is the deployed firmware's
NeoPixelBus feature, named per row.

| Product | Wire order | NeoPixelBus feature | Source of truth |
|---------|-----------|---------------------|-----------------|
| WS2811 bulb strings (tree of record, bulb fleet, bench chain) | RGB | `NeoRgbFeature` | tree-of-record main.cpp, bulb-fleet |
| WS2812 strip, 150 LED (eth boards) | GRB | `NeoGrbFeature` | tree_of_record_eth.cpp |
| SK6812 bullet nodes | GRBW | `NeoGrbwFeature` | bench test 2026-06 (same chip as flex strip, different product) |
| SK6812 flex strip | RGBW | `NeoRgbwFeature` | bench test 2026-06 |

## Power measurements

Test rig unless noted: throwaway firmware driving the whole chain one solid
color (mind the color-order table above), bench PSU current readout.

| Date | Product | Feed | Color | Current | Per-pixel | Far-end V | Notes |
|------|---------|------|-------|---------|-----------|-----------|-------|
| 2026-07-08 | WS2811 bulb string, 50 ct | 5V one end, data same end (bench-bulbs GPIO18) | white 255 | 1.6 A | 32 mA | 3.0 V | Droop-limited: 2 V sag means tail blue is starved, healthy-feed draw is higher |

## Caveats

- End-fed measurements underreport true full-brightness draw whenever far-end
  voltage sags below ~4 V — re-measure fed from both ends before sizing supplies.
