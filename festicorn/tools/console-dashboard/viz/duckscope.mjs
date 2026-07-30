// duckscope.mjs — the duck audio pipeline, drawn.
//
// Shows why and when the noise gate trips: raw mic RMS against the adaptive
// floor, the effective floor, the open threshold, the leaky ceiling, and the
// energy that falls out the far side.
//
// None of that is computed here. Every sample is replayed through the real
// lib/duck_energy math inside the WASM engine (duck_probe_feed) and this file
// only plots what comes back, so the panel cannot drift away from what the
// installation actually runs. The engine instance is its own — separate from
// the one led3d.mjs renders with — so probing never perturbs the sculpture.
//
// Samples come from /api/duck/stream, not /api/state: the duck sends at 25 Hz
// and the state poll would alias it.

const WINDOW_S = 60;
const KEEP = 4000;
const POLL_MS = 100;
const DT_MIN = 0.004, DT_MAX = 0.5;

const PAD = { l: 48, r: 42, t: 12, b: 20 };
const COL = {
  grid: "#242424", axis: "#5a5a5a",
  rms: "#e6e6e6", rmsMax: "#5b5b5b",
  floor: "#c8833c", eff: "#b05555", open: "#4a8a68",
  ceil: "#3c4a5a", energy: "#5b9bd5",
  gate: "rgba(74,138,104,0.10)", gateBar: "#4a8a68",
};

const dB = (v) => 20 * Math.log10(Math.max(v, 1));
const fmt = (v) => (v >= 10000 ? Math.round(v).toLocaleString() : v.toFixed(v < 10 ? 3 : 1));

export default async function mountDuckScope(root) {
  const initEngine = (await import("./engine/engine.mjs")).default;
  const M = await initEngine();
  const probeReset = M.cwrap("duck_probe_reset", null, []);
  const probeFeed = M.cwrap("duck_probe_feed", "number",
                            ["number", "number", "number", "number"]);
  const constCount = M.cwrap("duck_const_count", "number", [])();
  const constName = M.cwrap("duck_const_name", "string", ["number"]);
  const constValue = M.cwrap("duck_const_value", "number", ["number"]);

  const consts = [];
  for (let i = 0; i < constCount; i++) consts.push([constName(i), constValue(i)]);

  root.innerHTML = `
    <div class="dsplot"><canvas id="dscanvas"></canvas>
      <div class="dstip" id="dstip"></div>
      <div class="dslegend" id="dslegend"></div></div>
    <div class="dsside">
      <div class="sect">Live</div>
      <div id="dsread"></div>
      <div class="sect" style="margin-top:9px">Constants</div>
      <div class="dsconst" id="dsconst"></div>
    </div>`;

  root.querySelector("#dslegend").innerHTML = [
    ["rms mean", COL.rms], ["rms max", COL.rmsMax], ["floor", COL.floor],
    ["eff floor / close", COL.eff], ["open thr", COL.open], ["ceiling", COL.ceil],
    ["energy", COL.energy],
  ].map(([n, c]) => `<span><i style="background:${c}"></i>${n}</span>`).join("");

  root.querySelector("#dsconst").innerHTML = consts
    .map(([n, v]) => `<div class="srow"><span>${n.replace(/^DUCK_/, "")}</span><b>${fmt(v)}</b></div>`)
    .join("");

  const canvas = root.querySelector("#dscanvas");
  const ctx = canvas.getContext("2d");
  const tip = root.querySelector("#dstip");
  const readout = root.querySelector("#dsread");

  const frames = [];
  const events = [];
  let cursor = 0, lastT = null, paused = false, pausedAt = 0;
  let hoverX = null, yLo = 80, yHi = 108, dead = null;

  function resetProbe(why) {
    probeReset();
    lastT = null;
    if (why) pushEvent(frames.length ? frames[frames.length - 1].t : Date.now() / 1000,
                       "reset", why);
  }

  function pushEvent(t, kind, text, extra) {
    events.push({ t, kind, text, ...(extra || {}) });
    if (events.length > 600) events.splice(0, events.length - 600);
  }

  function feed(s) {
    let dt = lastT === null ? 1 / 25 : s.t - lastT;
    if (!(dt > DT_MIN)) dt = DT_MIN;
    if (dt > DT_MAX) { resetProbe("stream gap — floor rebuilt"); dt = DT_MAX; }
    lastT = s.t;

    const p = probeFeed(s.m, s.x, s.s || 0, dt) >> 2;
    const o = M.HEAPF32.subarray(p, p + 17);
    const f = {
      t: s.t, m: s.m, x: s.x, s: s.s,
      rms: o[0], rmsMax: o[1], floor: o[2], eff: o[3], open: o[4], ceil: o[5],
      energy: o[6], gate: o[7] > 0.5, hold: o[8], below: o[9],
      onFast: o[10], onSlow: o[11], rise: o[12], onThresh: o[13],
      onset: o[14], lock: o[15], dbAmb: o[16],
    };

    const prev = frames[frames.length - 1];
    if (prev) {
      if (f.gate && !prev.gate)
        pushEvent(f.t, "open", `gate open · ${f.dbAmb.toFixed(1)} dB over ambient`);
      else if (!f.gate && prev.gate)
        pushEvent(f.t, "close", `gate close · ${f.dbAmb.toFixed(1)} dB over ambient`);
      // A rebase is the floor tracking DOWN onto a new ambient (the belowCnt
      // branch), not the slow leak upward — those are different stories.
      if (f.below >= 3 && Math.abs(f.floor - prev.floor) / Math.max(prev.floor, 1) > 0.02)
        pushEvent(f.t, "rebase", `floor ${fmt(prev.floor)} → ${fmt(f.floor)}`);
    }
    if (f.onset > 0)
      pushEvent(f.t, "onset", `onset ${f.onset.toFixed(2)} · rise ${fmt(f.rise)}`,
                { rise: f.rise, value: f.onset });

    frames.push(f);
    if (frames.length > KEEP) frames.splice(0, frames.length - KEEP);
  }

  function poll() {
    fetch("/api/duck/stream?since=" + cursor, { signal: AbortSignal.timeout(1500) })
      .then((r) => r.json())
      .then((d) => {
        if (d.gap) resetProbe("dropped samples — floor rebuilt");
        cursor = d.cursor;
        for (const s of d.samples) feed(s);
        dead = null;
      })
      .catch(() => { dead = "no dashboard stream"; })
      .finally(() => setTimeout(poll, POLL_MS));
  }

  function windowEnd() {
    if (paused) return pausedAt;
    return frames.length ? frames[frames.length - 1].t : Date.now() / 1000;
  }

  function visible() {
    const t1 = windowEnd(), t0 = t1 - WINDOW_S;
    return { t0, t1, fr: frames.filter((f) => f.t >= t0 && f.t <= t1) };
  }

  function autoRange(fr) {
    let lo = Infinity, hi = -Infinity;
    for (const f of fr)
      for (const v of [f.rms, f.rmsMax, f.floor, f.eff, f.open, f.ceil]) {
        const d = dB(v);
        if (d < lo) lo = d;
        if (d > hi) hi = d;
      }
    if (!isFinite(lo)) { lo = 80; hi = 108; }
    lo -= 2; hi += 2;
    if (hi - lo < 24) { const c = (hi + lo) / 2; lo = c - 12; hi = c + 12; }
    yLo += 0.25 * (lo - yLo);
    yHi += 0.25 * (hi - yHi);
  }

  function draw() {
    const w = canvas.clientWidth, h = canvas.clientHeight;
    const dpr = Math.min(devicePixelRatio, 2);
    if (canvas.width !== w * dpr || canvas.height !== h * dpr) {
      canvas.width = w * dpr; canvas.height = h * dpr;
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);

    const { t0, t1, fr } = visible();
    const x0 = PAD.l, x1 = w - PAD.r, y0 = PAD.t, y1 = h - PAD.b;
    const X = (t) => x0 + ((t - t0) / WINDOW_S) * (x1 - x0);
    const Y = (v) => y1 - ((dB(v) - yLo) / (yHi - yLo)) * (y1 - y0);
    const YE = (e) => y1 - e * (y1 - y0);

    autoRange(fr);

    ctx.font = "9px 'SF Mono',Menlo,monospace";
    ctx.strokeStyle = COL.grid;
    ctx.lineWidth = 1;
    for (let d = Math.ceil(yLo / 6) * 6; d <= yHi; d += 6) {
      const y = Math.round(Y(Math.pow(10, d / 20))) + 0.5;
      ctx.beginPath(); ctx.moveTo(x0, y); ctx.lineTo(x1, y); ctx.stroke();
      ctx.fillStyle = COL.axis; ctx.textAlign = "right";
      ctx.fillText(d.toFixed(0) + " dB", x0 - 5, y + 3);
    }
    for (let s = 0; s <= WINDOW_S; s += 10) {
      const x = Math.round(X(t1 - s)) + 0.5;
      ctx.strokeStyle = COL.grid;
      ctx.beginPath(); ctx.moveTo(x, y0); ctx.lineTo(x, y1); ctx.stroke();
      ctx.fillStyle = COL.axis; ctx.textAlign = "center";
      ctx.fillText(s ? "-" + s + "s" : "now", x, y1 + 13);
    }
    for (const [e, lbl] of [[0, "0"], [0.5, "0.5"], [1, "1.0"]]) {
      ctx.fillStyle = COL.energy; ctx.textAlign = "left";
      ctx.fillText(lbl, x1 + 5, YE(e) + 3);
    }

    if (!fr.length) {
      ctx.fillStyle = "#555"; ctx.textAlign = "center";
      ctx.fillText(dead || "waiting for duck packets…", (x0 + x1) / 2, (y0 + y1) / 2);
      return;
    }

    ctx.save();
    ctx.beginPath(); ctx.rect(x0, y0, x1 - x0, y1 - y0); ctx.clip();

    const gateSpan = (a, b) => {
      ctx.fillStyle = COL.gate;
      ctx.fillRect(X(a), y0, Math.max(X(b) - X(a), 1), y1 - y0);
      ctx.fillStyle = COL.gateBar;
      ctx.fillRect(X(a), y0, Math.max(X(b) - X(a), 1), 3);
    };

    let spanFrom = fr[0].gate ? fr[0].t : null;
    for (const f of fr) {
      if (f.gate && spanFrom === null) spanFrom = f.t;
      else if (!f.gate && spanFrom !== null) { gateSpan(spanFrom, f.t); spanFrom = null; }
    }
    if (spanFrom !== null) gateSpan(spanFrom, fr[fr.length - 1].t);

    const line = (key, color, width, dash, scale, alpha) => {
      ctx.strokeStyle = color; ctx.lineWidth = width;
      ctx.globalAlpha = alpha === undefined ? 1 : alpha;
      ctx.setLineDash(dash || []);
      ctx.beginPath();
      fr.forEach((f, i) => {
        const y = scale ? YE(f[key]) : Y(f[key]);
        i ? ctx.lineTo(X(f.t), y) : ctx.moveTo(X(f.t), y);
      });
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.globalAlpha = 1;
    };

    line("ceil", COL.ceil, 1);
    line("rmsMax", COL.rmsMax, 1, null, false, 0.7);
    line("energy", COL.energy, 1, null, true, 0.7);
    line("open", COL.open, 1, [4, 3]);
    line("eff", COL.eff, 1, [2, 3]);
    line("floor", COL.floor, 1.5);
    line("rms", COL.rms, 1.4);

    for (const e of events) {
      if (e.t < t0 || e.t > t1) continue;
      const x = X(e.t);
      if (e.kind === "onset") {
        ctx.strokeStyle = "#e0c250"; ctx.lineWidth = 1.5;
        ctx.beginPath(); ctx.moveTo(x, y1); ctx.lineTo(x, y1 - 10 - 22 * (e.value || 0));
        ctx.stroke();
      } else if (e.kind === "rebase") {
        ctx.fillStyle = COL.floor;
        ctx.fillRect(x - 1, y1 - 3, 2, 3);
      } else if (e.kind === "reset") {
        ctx.strokeStyle = "#a44"; ctx.setLineDash([2, 2]);
        ctx.beginPath(); ctx.moveTo(x, y0); ctx.lineTo(x, y1); ctx.stroke();
        ctx.setLineDash([]);
      }
    }
    ctx.restore();

    if (hoverX !== null) {
      const t = t0 + ((hoverX - x0) / (x1 - x0)) * WINDOW_S;
      let best = null, bd = Infinity;
      for (const f of fr) { const d = Math.abs(f.t - t); if (d < bd) { bd = d; best = f; } }
      if (best && bd < 1.5) {
        const x = X(best.t);
        ctx.strokeStyle = "#666"; ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(x, y0); ctx.lineTo(x, y1); ctx.stroke();
        showTip(best, x, w);
      } else tip.style.display = "none";
    } else tip.style.display = "none";

    writeReadout(fr[fr.length - 1]);
  }

  function showTip(f, x, w) {
    const near = events.filter((e) => Math.abs(e.t - f.t) < 0.08).map((e) => e.text);
    const ts = new Date(f.t * 1000).toLocaleTimeString("en-US", { hour12: false });
    tip.innerHTML =
      `<b>${ts}</b>` +
      row("rms", fmt(f.rms)) + row("floor", fmt(f.floor)) +
      row("eff floor", fmt(f.eff)) + row("open thr", fmt(f.open)) +
      row("over ambient", f.dbAmb.toFixed(1) + " dB") +
      row("gate", f.gate ? "OPEN · hold " + f.hold.toFixed(2) + "s" : "closed") +
      row("energy", f.energy.toFixed(3)) +
      row("rise / thr", fmt(f.rise) + " / " + fmt(f.onThresh)) +
      (near.length ? `<div class="dstipev">${near.join("<br>")}</div>` : "");
    tip.style.display = "block";
    tip.style.left = Math.min(x + 10, w - tip.offsetWidth - 6) + "px";
  }

  const row = (k, v) => `<div class="srow"><span>${k}</span><b>${v}</b></div>`;

  function writeReadout(f) {
    if (!f) { readout.innerHTML = row("state", dead || "no data"); return; }
    const age = Date.now() / 1000 - f.t;
    readout.innerHTML =
      row("rms", fmt(f.rms)) +
      row("floor", fmt(f.floor)) +
      row("eff floor", fmt(f.eff)) +
      row("open at", fmt(f.open)) +
      row("over ambient", f.dbAmb.toFixed(1) + " dB") +
      `<div class="srow"><span>gate</span><b style="color:${f.gate ? "#4c5" : "#888"}">` +
      (f.gate ? "OPEN" : "closed") + "</b></div>" +
      row("energy", f.energy.toFixed(3)) +
      row("ceiling", fmt(f.ceil)) +
      row("age", age > 3 ? age.toFixed(1) + "s stale" : age.toFixed(2) + "s");
  }

  canvas.addEventListener("mousemove", (e) => {
    hoverX = e.clientX - canvas.getBoundingClientRect().left;
  });
  canvas.addEventListener("mouseleave", () => { hoverX = null; });

  (function loop() { draw(); requestAnimationFrame(loop); })();
  poll();

  return {
    get paused() { return paused; },
    togglePause() {
      paused = !paused;
      if (paused) pausedAt = frames.length ? frames[frames.length - 1].t : Date.now() / 1000;
      return paused;
    },
    reset() {
      frames.length = 0; events.length = 0;
      resetProbe(null);
    },
    exportWindow() {
      const { t0, t1, fr } = visible();
      return {
        window: { from: t0, to: t1, seconds: WINDOW_S },
        constants: Object.fromEntries(consts),
        events: events.filter((e) => e.t >= t0 && e.t <= t1),
        samples: fr.map((f) => ({
          t: f.t, m: f.m, x: f.x, s: f.s,
          rms: f.rms, floor: f.floor, eff: f.eff, open: f.open, ceil: f.ceil,
          energy: f.energy, gate: f.gate ? 1 : 0, onset: f.onset,
          onFast: f.onFast, onSlow: f.onSlow, rise: f.rise, dbAmb: f.dbAmb,
        })),
      };
    },
  };
}
