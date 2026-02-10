# Beat Tap Analysis: Tool - Opiate

Analysis of user tap annotations to understand what audio features they track.

## Overall Statistics

- Total taps: 72
- Key repeat candidates: 5

### Feature Alignment Summary

| Feature | Within 30ms | Within 50ms | Median Dist (ms) | Coverage |
|---------|-------------|-------------|------------------|----------|
| Onsets | 7/33 | 16/33 | 50.4 | 33/72 |
| RMS Peaks | 32/49 | 47/49 | 21.5 | 49/72 |
| Bass Peaks | 39/52 | 52/52 | 19.1 | 52/72 |
| Spectral Flux | 42/52 | 51/52 | 20.1 | 52/72 |

## Section Analysis

Sections defined by 'changes' layer timestamps:

| Section | Time Range | Taps | BPM | Consistency | Best Feature | Median Dist (ms) |
|---------|------------|------|-----|-------------|--------------|------------------|
| 0 | 0.00s - 8.07s | 11 | 129.3 | loose (1.89) | centroid_peak | 20.5 |
| 1 | 8.07s - 17.11s | 14 | 105.8 | loose (0.47) | flux_peak | 4.7 |
| 2 | 17.11s - 24.27s | 11 | 99.3 | loose (0.26) | flux_peak | 12.2 |
| 3 | 24.27s - 28.70s | 6 | 85.2 | tight (0.09) | flux_peak | 18.5 |
| 4 | 28.70s - 34.49s | 16 | 120.2 | loose (0.51) | centroid_peak | 12.9 |
| 5 | 34.49s - 37.37s | 6 | 115.4 | loose (0.28) | centroid_peak | 10.6 |
| 6 | 37.37s - 41.33s | 8 | 146.0 | loose (0.28) | centroid_peak | 6.1 |

### Section Details

#### Section 0: 0.00s - 8.07s

- **BPM**: 129.3 (median interval: 464.0ms, std: 875.3ms)
- **Consistency**: 1.886 (coefficient of variation)
- **Best aligned feature**: centroid_peak (median 20.5ms)

Feature alignments in this section:

- onset: 71.5ms median (2 taps)
- rms_peak: no matches
- bass_peak: 25.9ms median (3 taps)
- flux_peak: 67.8ms median (2 taps)
- centroid_peak: 20.5ms median (11 taps)

#### Section 1: 8.07s - 17.11s

- **BPM**: 105.8 (median interval: 567.0ms, std: 266.4ms)
- **Consistency**: 0.470 (coefficient of variation)
- **Best aligned feature**: flux_peak (median 4.7ms)

Feature alignments in this section:

- onset: 46.4ms median (2 taps)
- rms_peak: 24.5ms median (2 taps)
- bass_peak: 12.9ms median (2 taps)
- flux_peak: 4.7ms median (3 taps)
- centroid_peak: 18.1ms median (14 taps)

#### Section 2: 17.11s - 24.27s

- **BPM**: 99.3 (median interval: 604.5ms, std: 157.4ms)
- **Consistency**: 0.260 (coefficient of variation)
- **Best aligned feature**: flux_peak (median 12.2ms)

Feature alignments in this section:

- onset: 67.2ms median (9 taps)
- rms_peak: 14.3ms median (11 taps)
- bass_peak: 14.9ms median (11 taps)
- flux_peak: 12.2ms median (11 taps)
- centroid_peak: 15.5ms median (11 taps)

#### Section 3: 24.27s - 28.70s

- **BPM**: 85.2 (median interval: 704.0ms, std: 62.9ms)
- **Consistency**: 0.089 (coefficient of variation)
- **Best aligned feature**: flux_peak (median 18.5ms)

Feature alignments in this section:

- onset: 36.4ms median (4 taps)
- rms_peak: 35.5ms median (6 taps)
- bass_peak: 24.0ms median (6 taps)
- flux_peak: 18.5ms median (6 taps)
- centroid_peak: 21.9ms median (6 taps)

#### Section 4: 28.70s - 34.49s

- **BPM**: 120.2 (median interval: 499.0ms, std: 254.8ms)
- **Consistency**: 0.511 (coefficient of variation)
- **Best aligned feature**: centroid_peak (median 12.9ms)

Feature alignments in this section:

- onset: 36.2ms median (9 taps)
- rms_peak: 23.7ms median (16 taps)
- bass_peak: 15.2ms median (16 taps)
- flux_peak: 20.4ms median (16 taps)
- centroid_peak: 12.9ms median (16 taps)

#### Section 5: 34.49s - 37.37s

- **BPM**: 115.4 (median interval: 520.0ms, std: 144.9ms)
- **Consistency**: 0.279 (coefficient of variation)
- **Best aligned feature**: centroid_peak (median 10.6ms)

Feature alignments in this section:

- onset: 58.5ms median (3 taps)
- rms_peak: 26.6ms median (6 taps)
- bass_peak: 25.0ms median (6 taps)
- flux_peak: 22.2ms median (6 taps)
- centroid_peak: 10.6ms median (6 taps)

#### Section 6: 37.37s - 41.33s

- **BPM**: 146.0 (median interval: 411.0ms, std: 116.8ms)
- **Consistency**: 0.284 (coefficient of variation)
- **Best aligned feature**: centroid_peak (median 6.1ms)

Feature alignments in this section:

- onset: 54.5ms median (4 taps)
- rms_peak: 14.6ms median (8 taps)
- bass_peak: 19.8ms median (8 taps)
- flux_peak: 19.1ms median (8 taps)
- centroid_peak: 6.1ms median (8 taps)

## Key Repeat Candidates

Found 5 potential key-repeat artifacts (taps <100ms apart, not aligned with distinct onsets):

- Tap 44 at 29.329s (84.0ms after previous tap)
- Tap 45 at 29.413s (84.0ms after previous tap)
- Tap 46 at 29.498s (85.0ms after previous tap)
- Tap 47 at 29.581s (83.0ms after previous tap)
- Tap 53 at 32.231s (85.0ms after previous tap)

## Recommendations

- **Onset alignment**: 48.5% of taps within 50ms of an onset
- **Bass peak alignment**: 100.0% of taps within 50ms of a bass peak

**Recommendation**: Taps track bass peaks more than general onsets. Consider bass-specific beat tracking.

### Section-specific insights

- **Section 0** (0.00s - 8.07s): Tracks centroid_peak, 129.3 BPM, 11 taps
- **Section 1** (8.07s - 17.11s): Tracks flux_peak, 105.8 BPM, 14 taps
- **Section 2** (17.11s - 24.27s): Tracks flux_peak, 99.3 BPM, 11 taps
- **Section 3** (24.27s - 28.70s): Tracks flux_peak, 85.2 BPM, 6 taps
- **Section 4** (28.70s - 34.49s): Tracks centroid_peak, 120.2 BPM, 16 taps
- **Section 5** (34.49s - 37.37s): Tracks centroid_peak, 115.4 BPM, 6 taps
- **Section 6** (37.37s - 41.33s): Tracks centroid_peak, 146.0 BPM, 8 taps

User appears to track **different features in different sections**. This richness may be valuable for LED scene transitions.
