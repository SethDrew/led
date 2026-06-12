# AC Switch Failure + DC-Moded Record Fallback

Written install day (June 11, 2026), when the AC lever stopped working
reliably. It briefly recovered with mechanical effort, then failed
again the same night — **the fallback below is ACTIVE in the deployed
firmware**. When the switch is physically replaced, restore the original
two-switch mapping by reverting the `loop()` block to the version shown
in the git history (and the matching header/latch comments). It is a
tree-side-only change: no knob-box firmware or wiring involved.

## The failure

Symptom: lever ON, but the tree never starts recording.

Diagnosis split point: the tree prints a `[bs26]` status line over
serial every 2 s with the decoded packet fields. We saw:

```
[bs26] LIVE pkts=1349 ... | ac=0 dc=0 slot=0 rec=idle
```

`LIVE` with packets incrementing = radio + decode fine. `ac=0` while
the lever was physically ON = fault upstream of the tree, on the BS-26.

The AC sensing circuit (kaylab-knobs, GPIO 13, `pinMode(INPUT)`, external
pullup) is inverted from intuition: the oxidized contact (~100 Ω closed)
pulls the pin LOW when the lever is **OFF**; lever **ON** opens the
contact and the pullup takes the pin HIGH. A steady `ac=0` with the
lever ON means the contact is still conducting — the lever mechanism
no longer opens it. Confirmed mechanically: working the lever hard
sometimes frees it. The switch needs replacement; until then it is
flaky.

(If instead the pin had floated — pullup wire broken — the reading would
also sit low. Distinguish by lifting one switch wire with the lever ON:
tree shows `ac=1` → switch stuck closed; still `ac=0` → wiring/pullup.)

## The fallback design (DC does both jobs, moded by the effect dial)

The DC switch is the only other input with free semantic room. Rather
than timing gestures (flicks/holds — fragile against ESP-NOW packet
loss), the effect dial modes it:

| Control state            | DC ON means                      |
|--------------------------|----------------------------------|
| decouter 0–10            | **record** to the selected slot  |
| decouter 11 (rainbow)    | **play back** the selected slot  |

- Position 11 still renders rainbow normally while DC is off — nothing
  is lost from the dial.
- The action **binds at the DC rising edge** and holds until release
  (or the 30 s record cap), so moving the dial across the 11-boundary
  mid-action can never flip record↔playback. Dial moves mid-recording
  still record as effect changes, as always.
- `s.ac` is ignored entirely — the dying contact chatters as it
  oxidizes further, and honoring it would fire phantom records.
- The existing latch semantics survive unchanged: after a 30 s cap
  auto-stop, flip DC off to re-arm.
- Lost vs. the original mapping: record-overrides-playback (you must
  release, re-dial, re-flip — one extra step), and recording *at*
  position 11 (rainbow stays recordable mid-performance by dialing
  through it).

Muscle-memory hazard: hands trained on "DC = playback" will overwrite
the selected slot when flipping DC at dial 0–10. Park the hundreds knob
on an expendable slot when not deliberately recording.

## The duck is gated on the same dead switch

Found hours later, presenting as "the duck crashes after a while": the
duck (original-duck sender_v3) gates **all** of its forwarding — mic
waveform, audio telemetry, accel, gyro — on the same `"ac"` field in
the BS-26 broadcasts ("Gate ALL forwarding on the record switch —
confirmed design intent"). With the lever stuck closed, `ac=0` forever,
so the duck sits healthy but deliberately silent: recordings capture no
audio, playback is mute, and live effects lose all sensor reactivity.
The earlier on-again-off-again behavior was the lever being forced and
then settling closed again, not a crash.

Fix (keeps the frozen duck untouched): kaylab-knobs broadcasts the DC
state in the `"ac"` field while the lever is dead — see the TEMP
comment in its `broadcastState()`. The duck then streams whenever DC is
on, which covers the tree's whole record window (it also streams during
dial-11 playback; harmless — the tree ignores live audio while
playing). **Revert this together with the tree fallback** when the
lever is replaced: restore `(int)acOn` in kaylab's broadcastState.

## The patch

In `loop()`, replace the AC/DC handling block (search for
`AC switch = record`) with:

```cpp
        // One switch, two jobs: the AC lever broke at install (contact stuck
        // closed), so DC alone drives record AND playback, moded by the effect
        // dial — 0–10 = record the live performance, 11 (rainbow) = play back.
        // The action binds at the rising edge and holds until release, so dial
        // moves mid-action can't flip record↔playback. s.ac is ignored (the
        // dying contact chatters). LEVEL semantics otherwise unchanged: on =
        // active until release or the 30 s record cap; the latches keep the
        // cap from instantly restarting while the switch is still held.
        static bool dcWasOn = false, dcBoundReplay = false;
        bool dcOn = s.dc;
        if (dcOn && !dcWasOn) dcBoundReplay = (s.decouter >= FX_COUNT - 1);
        dcWasOn = dcOn;
        bool recOn  = dcOn && !dcBoundReplay;   // record request
        bool playOn = dcOn && dcBoundReplay;    // playback request
        if (!recOn)  recAcLatch = false;        // released → re-armed
        if (!playOn) recDcLatch = false;

        if (recState == REC_RECORDING) {
            if (!recOn) recStopRecording();                // switch off → stop
        } else if (recState == REC_PLAYING) {
            if (!playOn) recStopPlayback();                // switch off → stop
        } else {  // REC_IDLE
            if (recOn && !recAcLatch) {
                recStartRecording();
                recAcLatch = true;
            } else if (playOn && !recDcLatch) {
                recStartPlayback();
                recDcLatch = true;
            }
        }
```

Then build + flash:

```bash
cd festicorn/tree-of-record
../../.venv/bin/pio run -e tree-of-record-v2 -t upload --upload-port <port>
```

Verification sequence after flash (hundreds on an expendable slot):
dial 0–10 + DC on → serial shows `rec=REC`; DC off; dial 11 + DC on →
`rec=PLAY`, recording plays back.
