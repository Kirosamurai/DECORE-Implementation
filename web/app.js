// In-browser inference for the DECORE demo (ONNX Runtime Web, CPU/WASM).
const CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
                 "dog", "frog", "horse", "ship", "truck"];
const MEAN = [0.4914, 0.4822, 0.4465];
const STD = [0.2470, 0.2435, 0.2616];
const EXAMPLES = [
  "0_airplane.png", "1_automobile.png", "2_bird.png", "3_cat.png", "4_deer.png",
  "5_dog.png", "6_frog.png", "7_horse.png", "8_ship.png", "9_truck.png",
];

// CRITICAL: tell ORT where its .wasm files live (same CDN as ort.min.js),
// otherwise it looks next to index.html, 404s, and init hangs.
ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/";
// Single-threaded WASM (HF static hosting has no cross-origin isolation / SharedArrayBuffer).
ort.env.wasm.numThreads = 1;

const SESS_OPTS = { executionProviders: ["wasm"], graphOptimizationLevel: "all" };
let sessBase = null, sessPruned = null, modelsReady = false;
let loadTimeBase = 0, loadTimePruned = 0; // ms: download + ORT session init + warmup, per model
const $ = (id) => document.getElementById(id);

function setStatus(msg) { $("status").textContent = msg; }

async function warmup(sess) {
  const zeros = new ort.Tensor("float32", new Float32Array(3 * 32 * 32), [1, 3, 32, 32]);
  await sess.run({ [sess.inputNames[0]]: zeros });
}

async function loadModels() {
  try {
    setStatus("Downloading + initializing models (first load only)…");

    // The FIRST ort.InferenceSession.create() call anywhere pays a one-time
    // WASM-engine bootstrap cost (instantiating ort's own .wasm binary) that
    // has nothing to do with model size. Left alone, whichever model loads
    // first eats that fixed cost and makes the other look unfairly fast, so
    // we pay it here with a throwaway, cache-busted, untimed load before
    // timing either real model.
    const warm = await ort.InferenceSession.create(
      `models/pruned.onnx?warmup=${Date.now()}`, SESS_OPTS);
    await warmup(warm);

    // Timed + sequential per model (WASM backend is single-threaded / not
    // re-entrant, so concurrent run() calls raise "Session already started").
    // Cache-busted so neither timed load rides on the warm-up fetch's cache.
    let t0 = performance.now();
    sessBase = await ort.InferenceSession.create(
      `models/baseline.onnx?t=${Date.now()}`, SESS_OPTS);
    await warmup(sessBase);
    loadTimeBase = performance.now() - t0;
    $("load-base").textContent = `📦 loaded in ${loadTimeBase.toFixed(0)} ms · 58.9 MB`;

    t0 = performance.now();
    sessPruned = await ort.InferenceSession.create(
      `models/pruned.onnx?t=${Date.now()}`, SESS_OPTS);
    await warmup(sessPruned);
    loadTimePruned = performance.now() - t0;
    $("load-pruned").textContent = `📦 loaded in ${loadTimePruned.toFixed(0)} ms · 21.8 MB`;

    modelsReady = true;
    setStatus("Models ready — pick an example or upload an image.");
    updateRoi();
    if ($("preview").getAttribute("src")) classify();  // classify a pre-selected image
  } catch (e) {
    setStatus("Failed to load models: " + (e && e.message ? e.message : e));
    console.error(e);
  }
}

// Resize the current preview image into the model's fixed 32x32 input and
// return a CHW tensor. Uploaded images can be any resolution or aspect ratio
// (a full-res phone photo, a wide screenshot, ...) — center-crop to a square
// first, then downscale, so every upload always produces a well-formed
// 32x32x3 tensor without stretching/distorting it.
function preprocess() {
  const img = $("preview");
  const canvas = $("canvas");
  const ctx = canvas.getContext("2d");
  const sw = img.naturalWidth, sh = img.naturalHeight;
  const side = Math.min(sw, sh);
  const sx = (sw - side) / 2, sy = (sh - side) / 2;
  ctx.clearRect(0, 0, 32, 32);
  ctx.drawImage(img, sx, sy, side, side, 0, 0, 32, 32);
  const { data } = ctx.getImageData(0, 0, 32, 32); // RGBA, HWC
  const arr = new Float32Array(1 * 3 * 32 * 32);
  for (let y = 0; y < 32; y++) {
    for (let x = 0; x < 32; x++) {
      const p = (y * 32 + x) * 4;
      for (let c = 0; c < 3; c++) {
        const v = data[p + c] / 255.0;
        arr[c * 32 * 32 + y * 32 + x] = (v - MEAN[c]) / STD[c];
      }
    }
  }
  return new ort.Tensor("float32", arr, [1, 3, 32, 32]);
}

function softmax(logits) {
  const m = Math.max(...logits);
  const ex = logits.map((v) => Math.exp(v - m));
  const s = ex.reduce((a, b) => a + b, 0);
  return ex.map((v) => v / s);
}

function renderBars(el, probs) {
  const idx = probs.map((p, i) => [p, i]).sort((a, b) => b[0] - a[0]).slice(0, 3);
  el.innerHTML = "";
  for (const [p, i] of idx) {
    const row = document.createElement("div");
    row.className = "bar-row";
    row.innerHTML = `<span class="cls">${CLASSES[i]}</span>
      <span class="track"><span class="fill" style="width:${(p * 100).toFixed(1)}%"></span></span>
      <span class="pct">${(p * 100).toFixed(1)}%</span>`;
    el.appendChild(row);
  }
}

function argmax(a) { let b = 0; for (let i = 1; i < a.length; i++) if (a[i] > a[b]) b = i; return b; }

function renderTop1(el, probs) {
  const i = argmax(probs);
  el.innerHTML = `<span class="cls-big">${CLASSES[i]}</span>` +
                 `<span class="conf">${(probs[i] * 100).toFixed(1)}% confident</span>`;
}

function renderVerdict(pb, pp) {
  const v = $("verdict");
  const ib = argmax(pb), ip = argmax(pp);
  if (ib === ip) {
    v.className = "verdict agree";
    v.textContent = `✓ Both agree: ${CLASSES[ib]} — the 63%-smaller model made the same call.`;
  } else {
    v.className = "verdict disagree";
    v.textContent = `△ Disagreement — full: ${CLASSES[ib]}, pruned: ${CLASSES[ip]}.`;
  }
}

async function runSession(sess, input) {
  const t0 = performance.now();
  const out = await sess.run({ [sess.inputNames[0]]: input });
  const dt = performance.now() - t0;
  const logits = Array.from(out[sess.outputNames[0]].data);
  return { probs: softmax(logits), dt };
}

// ROI calculator: a model's real-world "worth" is load time (download + init —
// what a cold-starting serverless/autoscaled replica or a first-time visitor
// pays) PLUS inference time (what every request pays). Both come from live
// in-browser measurements (loadModels() and classify()); cost math is a
// simple "compute-time-proportional" estimate.
let lastInferBase = 0, lastInferPruned = 0; // ms, most recent classify() run
let lastSpeedup = 2.0; // sane default (matches README headline) before any run
const fmtUSD = (n) => "$" + Math.round(n).toLocaleString("en-US");
const fmtMs = (n) => n.toFixed(0) + " ms";

function updateRoi() {
  const totalBase = loadTimeBase + lastInferBase;
  const totalPruned = loadTimePruned + lastInferPruned;
  lastSpeedup = totalBase / Math.max(totalPruned, 1e-3);

  $("roi-speedup").textContent = `${lastSpeedup.toFixed(2)}×`;
  $("roi-speedup-label").textContent = `~${lastSpeedup.toFixed(1)}×`;
  $("roi-breakdown").textContent =
    `Load ${fmtMs(loadTimeBase)} → ${fmtMs(loadTimePruned)}  ·  ` +
    `Inference ${fmtMs(lastInferBase)} → ${fmtMs(lastInferPruned)}  ·  ` +
    `Total ${fmtMs(totalBase)} → ${fmtMs(totalPruned)}`;

  const spend = parseFloat($("roi-spend").value);
  if (!spend || spend <= 0 || !isFinite(spend)) {
    $("roi-monthly").textContent = "$0";
    $("roi-annual").textContent = "$0";
    return;
  }
  const savingsFraction = Math.max(0, 1 - 1 / lastSpeedup);
  const monthly = spend * savingsFraction;
  $("roi-monthly").textContent = fmtUSD(monthly);
  $("roi-annual").textContent = fmtUSD(monthly * 12);
}

$("roi-spend").addEventListener("input", updateRoi);
updateRoi();

let busy = false;
async function classify() {
  if (busy || !modelsReady || !$("preview").getAttribute("src")) return;
  const img = $("preview");
  if (!img.naturalWidth || !img.naturalHeight) {
    setStatus("Couldn't read that image — try a JPG, PNG, or WebP.");
    return;
  }
  busy = true;
  setStatus("Running inference…");
  try {
    const input = preprocess();
    const b = await runSession(sessBase, input);   // sequential (WASM not re-entrant)
    const p = await runSession(sessPruned, input);
    renderTop1($("top-base"), b.probs);
    renderTop1($("top-pruned"), p.probs);
    renderBars($("out-base"), b.probs);
    renderBars($("out-pruned"), p.probs);
    lastInferBase = b.dt;
    lastInferPruned = p.dt;
    const inferSpeedup = b.dt / Math.max(p.dt, 1e-3);
    $("lat-base").textContent = `⏱ ${b.dt.toFixed(1)} ms  ·  full model`;
    $("lat-pruned").textContent = `⏱ ${p.dt.toFixed(1)} ms  ·  ${inferSpeedup.toFixed(1)}× faster`;
    renderVerdict(b.probs, p.probs);
    updateRoi();
    setStatus("Both models ran locally in your browser.");
  } catch (e) {
    setStatus("Inference error: " + (e && e.message ? e.message : e));
    console.error(e);
  } finally {
    busy = false;
  }
}

let lastObjectUrl = null;
function loadImageSrc(src) {
  const img = $("preview");
  img.onload = () => { $("preview-wrap").classList.add("has-img"); classify(); };
  img.onerror = () => {
    $("preview-wrap").classList.remove("has-img");
    setStatus("Couldn't read that image — try a JPG, PNG, or WebP.");
  };
  img.src = src;
  // Release the previous upload's blob URL — a new file:// / example click
  // means it's no longer referenced, and these otherwise leak for the page's
  // lifetime.
  if (lastObjectUrl) URL.revokeObjectURL(lastObjectUrl);
  lastObjectUrl = src.startsWith("blob:") ? src : null;
}

function initExamples() {
  const box = $("examples");
  for (const name of EXAMPLES) {
    const t = document.createElement("img");
    t.src = "examples/" + name;
    t.className = "thumb";
    t.title = name.split("_")[1].replace(".png", "");
    t.onclick = () => loadImageSrc(t.src);
    box.appendChild(t);
  }
}

$("file").addEventListener("change", (e) => {
  const f = e.target.files[0];
  if (f) loadImageSrc(URL.createObjectURL(f));
});

initExamples();
loadModels();
