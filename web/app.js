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
let sessBase = null, sessPruned = null;
const $ = (id) => document.getElementById(id);

function setStatus(msg) { $("status").textContent = msg; }

async function warmup(sess) {
  const zeros = new ort.Tensor("float32", new Float32Array(3 * 32 * 32), [1, 3, 32, 32]);
  await sess.run({ [sess.inputNames[0]]: zeros });
}

async function loadModels() {
  try {
    setStatus("Downloading + initializing models (first load only)…");
    sessBase = await ort.InferenceSession.create("models/baseline.onnx", SESS_OPTS);
    sessPruned = await ort.InferenceSession.create("models/pruned.onnx", SESS_OPTS);
    // Run sequentially: the WASM backend is single-threaded / not re-entrant,
    // so concurrent run() calls raise "Session already started".
    await warmup(sessBase);
    await warmup(sessPruned);
    $("run").disabled = false;
    $("run").textContent = "Classify";
    setStatus("Models loaded — running locally in your browser.");
  } catch (e) {
    setStatus("Failed to load models: " + (e && e.message ? e.message : e));
    console.error(e);
  }
}

// Draw the current preview image into the 32x32 canvas and return a CHW tensor.
function preprocess() {
  const img = $("preview");
  const canvas = $("canvas");
  const ctx = canvas.getContext("2d");
  ctx.drawImage(img, 0, 0, 32, 32);
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

async function runSession(sess, input) {
  const t0 = performance.now();
  const out = await sess.run({ [sess.inputNames[0]]: input });
  const dt = performance.now() - t0;
  const logits = Array.from(out[sess.outputNames[0]].data);
  return { probs: softmax(logits), dt };
}

let busy = false;
async function classify() {
  if (busy || !sessBase || !sessPruned || !$("preview").src) return;
  busy = true;
  $("run").disabled = true;
  setStatus("Running inference…");
  try {
    const input = preprocess();
    const b = await runSession(sessBase, input);   // sequential (WASM not re-entrant)
    const p = await runSession(sessPruned, input);
    renderBars($("out-base"), b.probs);
    renderBars($("out-pruned"), p.probs);
    $("lat-base").textContent = `⏱ ${b.dt.toFixed(1)} ms`;
    $("lat-pruned").textContent = `⏱ ${p.dt.toFixed(1)} ms`;
    setStatus("Done — both models ran locally in your browser.");
  } catch (e) {
    setStatus("Inference error: " + (e && e.message ? e.message : e));
    console.error(e);
  } finally {
    busy = false;
    $("run").disabled = false;
  }
}

function loadImageSrc(src) {
  const img = $("preview");
  img.onload = () => { if (!$("run").disabled) classify(); };
  img.src = src;
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
$("run").addEventListener("click", classify);

initExamples();
loadModels();
