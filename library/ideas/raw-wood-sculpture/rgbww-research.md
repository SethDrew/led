# RGBWW LED Strip Research for Wood Vein Art

## The Core Question

Can you sell driftwood sculptures with LED-filled epoxy veins and offer a reasonable warranty?

## Failure Risk Assessment

WS2812B (standard addressable) has ~50,000 hr MTBF spec but real-world reports suggest ~1% failure per 1,000 hours on large installs. Failures are mostly manufacturing defects (bad solder, moisture sensitivity) not LED die burnout. WS2812B is a serial protocol — one dead LED can break the entire downstream chain. This cascade failure is the main reason people don't sell epoxy-potted LED art with confidence.

## Mitigation Strategy (Recommended)

1. Use WS2815 or SK6812 with dual data line — failed LED is bypassed, rest of chain keeps working
2. High density (144/m) — single dead pixel is barely noticeable
3. Burn-in test every piece at full white for 48-72 hours before selling
4. Refund policy rather than repair — simpler and cheaper than designing for repairability

With these mitigations, a dead pixel is cosmetic, not catastrophic.

## Warranty Recommendation

- 2-year warranty on wood/epoxy/diffuser (structural)
- 1-year warranty on LED function
- Refund or replace if customer is unhappy
- Cost of a warranty claim: price of the piece, but failure rate post-burn-in should be very low

## Strip Selection

### WS2815 RGBW (Preferred for color-changing pieces)

- 12V, dual data line (backup signal bypasses dead LEDs)
- 4-in-1 SMD5050 with dedicated white channel
- Dedicated warm white (3000K) looks much better than faking warm tones with RGB
- Available up to 144 LEDs/m
- Hard to find in warm white from major sellers like BTF-Lighting
- Sources: LEDLightsWorld.com, SuperLightingLED.com, AliExpress suppliers

### WS2805 RGBWW (Premium option)

- 5-in-1 chip: RGB + warm white + cool white (tunable white)
- Dual data line
- Can dial in any white temperature — maximum flexibility for matching wood tone
- More expensive, niche product
- Source: SuperLightingLED.com

### SK6812 RGBW (Good availability, no dual data line)

- Available from BTF-Lighting (trusted, ~$42/5m on Amazon) and cheap AliExpress sellers like Balaber (~$13/5m)
- 4x price difference is Amazon fees + QC/binning + US warehouse, not fundamentally different product
- No dual data line — single LED failure can cascade (same risk as WS2812B)
- For potted art pieces, pay the BTF premium for solder quality and LED binning
- Cheap strips fine for prototyping

### Non-addressable constant current (Single color pieces)

- Warm white or green only veins
- No IC per LED = inherently most reliable, no cascade failure possible
- Single LED failure = one slightly dim spot, everything else keeps running
- Lowest risk option for potted-in-epoxy applications

## Key Decision: Epoxy is Non-Negotiable

Vein patterns in wood require a fill material — can't friction-fit tubing into organic branching channels. Epoxy gives the professional, solid finish needed for sellable art. Silicone is cuttable/repairable but looks DIY.

Repairability is not worth the aesthetic tradeoff. Dual data line + high density + burn-in makes the failure risk acceptably low.

## Thermal Notes

- Epoxy is a thermal insulator — LEDs can't shed heat normally
- At art-piece brightness levels (low-to-medium), thermal load is minimal
- Wood acts as thermal mass, absorbs and slowly dissipates
- WS2815 at 12V is more efficient than 5V strips (less heat per LED)
- Full white at max brightness could be an issue but art pieces wouldn't run that way
- Biggest thermal risk is epoxy exotherm during curing — pour in thin layers

## Supplier Notes

- BTF-Lighting: reliable, good QC, Amazon-available, but no WS2815 RGBW
- Balaber (AliExpress): ~4x cheaper for SK6812, likely same chips, unknown QC — fine for prototyping, burn-in test aggressively before using in sellable pieces
- SuperLightingLED / LEDLightsWorld / Suntechlite: Chinese suppliers with WS2815 RGBW warm white and WS2805 RGBWW — niche products, direct ordering
