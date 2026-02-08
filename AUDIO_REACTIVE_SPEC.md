# Audio-Reactive LED System - Research & Implementation Spec

## Goal
Research existing audio-reactive LED libraries and build a custom streaming-based audio-reactive system that responds intelligently to different music genres with beat detection, frequency analysis, and future sentiment-based color changes.

## Done when
- [ ] Research report completed: markdown document comparing top audio-reactive libraries (scottlawsonbc + others) with popularity metrics, use cases, complexity analysis
- [ ] Key takeaways identified: what's worth using vs building ourselves
- [ ] Custom system architecture designed: streaming-based (like nebula), computer does audio analysis + frame generation
- [ ] Testing methodology established: 10-second audio segment recording/playback for apples-to-apples comparison
- [ ] Autonomy boundaries defined: clear documentation of what Opus can do autonomously vs what needs user evaluation
- [ ] LED evaluation schema created: structured format for user to provide feedback on LED behavior for specific audio segments
- [ ] First working prototype: basic audio-reactive effect with beat detection OR frequency analysis (whichever is most architecturally representative). Can be on music I play for us or a segment we capture and you can replay.

## Approach

### Phase 1: Research (Autonomous - Opus handles this)
**Goal**: Understand landscape, don't reinvent wheel unnecessarily

Research existing audio-reactive LED libraries:
- **scottlawsonbc/audio-reactive-led-strip** (already tried, user not happy out of box)
- Other popular options (GitHub stars >500, active in last year)

For each library, document:
- **Popularity**: Stars, forks, last commit, community activity
- **Quality**: Code quality, documentation, test coverage
- **Use case**: What it's designed for (club lights, ambient, specific hardware)
- **Complexity**: Learning curve, customization difficulty
- **Architecture**: How it works (FFT? ML? Beat detection approach?)
- **Extensibility**: How hard to add custom features like genre modes or sentiment

**Deliverable**: `RESEARCH_FINDINGS.md` with:
- Comparison table
- Key takeaways from each library
- Complexity analysis: use existing vs build custom
- Recommendation on whether to fork/extend or build from scratch

### Phase 2: Architecture Design (Collaborative - Opus proposes, User approves)
**Goal**: Design system that supports user's vision

Design custom streaming audio-reactive system:
- **Hardware**: Streaming setup (computer Python → Arduino receiver, like nebula)
- **Audio input**: Start with audio files (10-sec test segments), plan for live capture later
- **Features to support** (implement in order of: ease + architectural representativeness):
  1. Beat detection (tempo, downbeats) → brightness pulses
  2. Frequency analysis (bass/mid/high) → different LED zones or colors
  3. Genre modes (dance vs prog/rock) → different response patterns
  4. Sentiment/mood analysis (advanced) → color based on key/lyrics/emotion

**Deliverable**: `ARCHITECTURE.md` with:
- System diagram
- Audio processing pipeline (capture → analysis → LED frame generation → serial stream)
- Module breakdown (which parts are reusable from nebula, what's new)
- Testing methodology: how to record/playback 10-sec segments for consistent evaluation
- Data flow: audio file → features → LED colors

### Phase 3: Testing Methodology (Autonomous)
**Goal**: Consistent evaluation across implementations

Design testing framework:
- **Test suite**: Collection of 10-second audio segments covering:
  - Dance music (steady 4/4, strong bass)
  - Rock/prog (Tool-style odd time signatures)
  - Ambient/slow (mood/sentiment focus)
  - High energy (fast tempo, complex)
- **Playback system**: Consistent audio file → LED frame generation → visual output
- **Evaluation schema**: Structured format for user to document:
  - Audio segment ID
  - Expected LED behavior (brightness on beat? color change on mood?)
  - Actual LED behavior
  - Rating (1-5 scale)
  - Notes

**Deliverable**:
- `test_segments/` directory structure
- `TESTING.md` guide for recording and evaluating
- `evaluation_template.yaml` - structured schema for user feedback

### Phase 4: Prototype Implementation (Mostly Autonomous, User evaluates results)
**Goal**: Working proof-of-concept for one feature (beat detection OR frequency analysis)

Implement first audio-reactive effect:
- Reuse `nebula_stream.py` architecture (LEDStreamer, frame generation)
- New `AudioReactiveEffect` class
- Audio library selection (librosa? aubio? sounddevice?)
- Start with whichever is easiest + most architecturally representative:
  - **Beat detection**: Detect tempo, trigger brightness pulses on beats
  - **Frequency analysis**: FFT → bass/mid/high → color zones

**Deliverable**:
- `streaming/single-strip/controller/audio_reactive_stream.py`
- Test with one 10-second segment
- Documentation: how to extend to other features
- Clear TODO comments for genre modes, sentiment analysis

### Phase 5: Define Autonomy Boundaries (Autonomous)
**Goal**: Clear understanding of what Opus can/can't do without user

Document in `AUTONOMY.md`:
- **Opus can do autonomously**:
  - Code research, library comparison
  - Audio analysis algorithm implementation
  - Effect logic (brightness, color calculations)
  - Code testing, debugging
  - Architectural decisions (within spec)
- **User must be in loop for**:
  - LED visual evaluation (does it look good? does it match music feel?)
  - Creative decisions (which colors for happy vs sad?)
  - Genre mode definitions (what makes Tool different from house music?)
  - Test segment selection (which songs/sections to test)
  - Final approval of each phase

**Training possibility**: If user provides structured examples:
- "For segment X (high-energy dance), LEDs should: [brightness 80%+, pulse on every beat, blue-purple colors]"
- Opus could learn patterns and propose LED behaviors for new segments
- Still needs user validation

## Out of scope
- **Legal research** on recording copyrighted music (user handles separately)
- **Live audio capture** implementation (design for it, implement later)
- **Complete implementation** of all features (start with one, document expansion path)
- **Arduino standalone** audio processing (computer does all analysis)
- **Real-time performance optimization** (prove concept first, optimize later)
- **Hardware purchases** or changes (use existing Arduino Nano + LED strips)

## Technical Constraints
- **Hardware**: Existing Arduino Nano (controller1/controller2), 150 LEDs
- **Streaming protocol**: Reuse existing serial streaming (1 Mbps, 60 FPS target)
- **Python libraries**: Prefer standard audio libraries (librosa, numpy, scipy)
- **User's audio experience**: Some familiarity with concepts, explain but don't over-explain

## Success Metrics
- Research report is comprehensive but actionable (not analysis paralysis)
- Custom implementation is cleaner/simpler than adapting existing library
- Testing methodology allows consistent, repeatable evaluation
- User can easily provide structured feedback via evaluation schema
- First prototype responds visibly to music (even if crude)
- Clear path forward for adding genre modes, sentiment analysis

## Autonomous work instructions
Opus should:
- Work through phases sequentially
- Use tasks to track progress
- Commit logical chunks of work (research → architecture → prototype)
- When blocked on creative decisions, propose 2-3 options with pros/cons
- For LED evaluation, create tools for user feedback rather than guessing
- Check in after Phase 1 (research) and Phase 2 (architecture) before implementing
- Document everything assuming future expansion

## Notes
- User already tried scottlawsonbc, wasn't happy out of box
- User's vision: genre-aware modes, beat pulses, sentiment colors (ambitious but cool!)
- Start simple (beat detection), prove architecture, then expand
- Testing with recorded segments lets us iterate without music playing constantly
- This is Opus territory: complex, long-horizon, architectural thinking required
