# Axismundi — Installation Audio

Ambient audio bed for the Axis Mundi installation, running **24/7** unattended.

## Goal

Assemble ~24 hours of albums / DJ sets / mixes, downloaded for **offline** playback
(no streaming, no network dependency), and play them on a **dedicated Android phone**
that survives reboots and power loss without a human present.

## Approach

- **Files** — full albums/sets as single `.opus` files (best seek precision + size).
  Whole-album files keep interrupts clean; no mid-track gaps.
- **Downloading** — `yt-dlp -x --audio-format opus --audio-quality 0` from YouTube.
- **Playback** — mpv (headless, in Termux) or Poweramp, looping gapless. TBD.
- **Scheduling** — time-of-day is the leading option: on boot, read the wall clock
  and resume the correct point, so a power blip self-heals with nobody watching.
  (A single exactly-24h file with `mpv --start=$SECONDS_SINCE_MIDNIGHT --loop-file`
  is the simpler alternative if the day is authored once and left alone.)

## Priority tiers

`tiers.yaml` ranks tracks high / middle / low — high plays more (prime hours),
low is filler. Source of truth for what's in the library and its weighting.

## Status

Collecting toward ~24h. See `tiers.yaml` for current inventory. Playback app and
scheduler not yet built.
