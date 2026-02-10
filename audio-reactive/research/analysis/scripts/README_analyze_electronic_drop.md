# Electronic Drop Structure Analyzer

Analyzes electronic music recordings to detect build/drop structure and create structured annotations.

## Purpose

Automatically detect and annotate the structural sections of electronic music drops:
- Normal song section
- Tease/edge builds (cyclic builds without full payoff)
- Real build (sustained energy accumulation)
- Bridge (transitional moment before drop)
- Drop/crescendo (maximum intensity)

## Usage

### Basic Usage

```bash
source /Users/KO16K39/Documents/led/venv/bin/activate
python analyze_electronic_drop.py
```

The script will:
1. Look for audio files in `/Users/KO16K39/Documents/led/audio-reactive/research/audio-segments/`
2. Prioritize files matching expected naming patterns
3. Analyze and create outputs automatically

### Output Files

For input file `track_name.wav`, creates:

**Annotations:**
- `track_name.annotations.yaml` - Detailed annotations with labeled sections
- `track_name.annotations_simple.yaml` - Simple format (sections, build, drop, cycles)

**Analysis:**
- `track_name_analysis.md` - Detailed analysis report
- `track_name_analysis.png` - Multi-panel visualization

### Customization

To analyze a different file, edit the `main()` function:

```python
# Change these paths
segments_dir = Path('/path/to/audio/segments')
output_dir = Path('/path/to/output')

# Change file search
primary_candidates = ['your_file.wav', 'alternative_name.wav']
```

## How It Works

### Feature Extraction (~23ms resolution)

- **RMS energy:** Overall loudness
- **Spectral centroid:** Brightness (average frequency)
- **Spectral flux:** Rate of spectral change
- **Onset strength:** Rhythmic activity
- **Frequency bands:** Bass (20-250 Hz), Mids (250-4000 Hz), Highs (4000-16000 Hz)
- **HPSS:** Harmonic vs percussive content
- **Spectral flatness:** Noise vs tone character

### Section Detection Algorithm

1. **Tease start:** Significant drop in RMS/bass around 15-30s
2. **Real build start:** Sustained upward trend in mid-energy around 75-95s
3. **Bridge:** Spectral flatness spike or unusual centroid after build
4. **Drop:** Maximum bass × RMS after bridge
5. **Cyclic builds:** Local maxima in mid-energy during tease section

### Key Parameters

```python
hop_length = 1024  # ~23ms frames at 44.1kHz
smoothing_sigma = 20  # Gaussian smoothing for trends
window_size = 50  # ~1s window for trend calculation
```

## Expected Structure

The analyzer expects this typical electronic drop structure:

```
[Normal Song] → [Tease/Edge] → [Real Build] → [Bridge] → [Drop]
   15-30s         40-65s          20-40s        5-10s     5-10s
```

**Normal Song:** Established groove, baseline energy

**Tease:** Sparse arrangement with cyclic builds that don't pay off

**Real Build:** Sustained energy increase, brighter spectral content

**Bridge:** Short transitional section with unusual sounds

**Drop:** Maximum intensity, full-spectrum

## Interpreting Results

### Section Characteristics

Check `track_name_analysis.md` for detailed feature analysis of each section.

**Key indicators:**
- **Tease cycles:** How many mini-builds were detected
- **Feature changes at boundaries:** What changed when sections transition
- **Spectral centroid trend:** Rising = building brightness/energy
- **Bass energy:** Should peak at drop

### Validation

Compare detected timestamps to your own perception:
- Do the section boundaries feel correct?
- Are cyclic builds aligned with actual musical events?
- Does the bridge capture the transitional moment?

If detection is off, consider:
- Adjusting search ranges in `detect_sections()`
- Tuning smoothing parameters for your genre
- Different prominence thresholds for peak detection

## Genre Adaptations

### Progressive House/Trance
- Increase smoothing windows (longer builds)
- Wider search ranges for build detection
- Higher emphasis on spectral centroid changes

### Dubstep/Trap
- Shorter smoothing windows
- Higher weight on bass band
- Emphasis on percussive component

### Minimal/Deep House
- More sensitive thresholds (subtle changes)
- Longer trend windows
- Lower prominence for peak detection

## Troubleshooting

**"Normal song" section too long/short:**
- Adjust search range in `detect_sections()` (line ~150)

**Tease cycles not detected:**
- Lower prominence threshold in `detect_cyclic_builds()` (line ~220)
- Check if mid-energy is the right feature for your track

**Bridge not found:**
- Track may not have a distinct bridge (some go build → drop directly)
- Try adjusting spectral flux threshold

**Drop detection wrong:**
- Ensure bridge is detected correctly first
- Check if bass × RMS is the right metric for your track style

## Dependencies

- librosa
- numpy
- scipy
- matplotlib
- soundfile
- pyyaml

All available in project venv: `/Users/KO16K39/Documents/led/venv`

## Future Improvements

- [ ] Command-line arguments for file selection
- [ ] Adjustable detection parameters via config file
- [ ] Multiple genre presets
- [ ] Real-time streaming version
- [ ] Comparison to user tap annotations
- [ ] Batch processing of multiple files
