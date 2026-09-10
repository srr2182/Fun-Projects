const { useState, useRef, useEffect } = React;

let nextId = 1;
const API_KEY_STORAGE = "photodev:apiKey";

const REFERENCE_PROMPT = `You are a professional colorist analyzing sample photographs from a photographer's portfolio to reverse-engineer their editing style, so it can be applied to other photographs using traditional darkroom adjustments only (no generative changes).

Look at these reference images and describe the color grading and tonal style as concrete, numeric adjustment values on the following traditional controls. Use your best judgment for typical values that would recreate this look starting from a neutral, unedited photo.

Respond with ONLY JSON, no prose, no markdown fences:
{
  "description": "one sentence describing the aesthetic",
  "tempShift": -100 to 100 (negative = cooler/blue, positive = warmer/orange),
  "tint": -100 to 100 (negative = green, positive = magenta),
  "exposure": -100 to 100,
  "contrast": -100 to 100,
  "blackPoint": 0 to 40,
  "whitePoint": 215 to 255,
  "highlights": -100 to 100,
  "shadows": -100 to 100,
  "saturation": -100 to 100,
  "vibrance": -100 to 100,
  "sharpen": 0 to 100,
  "grain": 0 to 100,
  "vignette": 0 to 100
}`;

const RETOUCH_PROMPT = `You are assisting a professional photo retoucher. Look at this photo and identify ONLY small, removable surface imperfections that a professional retoucher would routinely clean up without changing the photo's content or composition.

Flag things like: a visible bra strap or undergarment line, a sleep crease on skin, a stray flyaway hair, a skin blemish or temporary mark, a dust or sensor spot, a small stain or lint on clothing.

Do NOT flag: people, animals, or objects that are part of the scene; background elements, walls, or their colors; anything that would change who or what is in the photo; anything you are not confident is a minor, removable imperfection rather than a real part of the scene. If in doubt, leave it out.

Respond with ONLY a JSON array, no prose, no markdown fences. Each item: {"label": short description, "category": one of "strap","sleep_line","blemish","stray_hair","dust_spot","stain","other_minor", "box": [x, y, w, h] as fractions of image width/height (0 to 1), "confidence": 0 to 1}. If nothing qualifies, return [].`;

const GEOMETRY_PROMPT = `You are a professional photo editor suggesting straightening and cropping only — nothing else. Look at this photo and suggest a small straighten angle and crop insets that improve composition (level a tilted horizon, tighten framing, remove dead space) while keeping every person, animal, and major subject fully inside the frame. If the photo is already well composed, suggest little or no change.

Respond with ONLY JSON, no prose, no markdown fences:
{"rotationDeg": -10 to 10, "cropTopPct": 0 to 20, "cropBottomPct": 0 to 20, "cropLeftPct": 0 to 20, "cropRightPct": 0 to 20}`;

function loadImage(src) {
  return new Promise((resolve) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.src = src;
  });
}
function fileToDataUrl(file) {
  return new Promise((resolve) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.readAsDataURL(file);
  });
}
function stripFences(text) {
  return text.replace(/```json/gi, "").replace(/```/g, "").trim();
}

async function callClaudeVision(apiKey, promptText, imageDataUrls) {
  if (!apiKey) throw new Error("no-key");
  const content = imageDataUrls.map((url) => ({
    type: "image",
    source: { type: "base64", media_type: "image/png", data: url.split(",")[1] },
  }));
  content.push({ type: "text", text: promptText });
  const response = await fetch("https://api.anthropic.com/v1/messages", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "x-api-key": apiKey,
      "anthropic-version": "2023-06-01",
      "anthropic-dangerous-direct-browser-access": "true",
    },
    body: JSON.stringify({
      model: "claude-sonnet-4-6",
      max_tokens: 1000,
      messages: [{ role: "user", content }],
    }),
  });
  if (!response.ok) {
    const errText = await response.text().catch(() => "");
    throw new Error("api-error:" + response.status + " " + errText.slice(0, 200));
  }
  const data = await response.json();
  const textBlock = (data.content || []).find((c) => c.type === "text");
  if (!textBlock) throw new Error("No text response");
  return JSON.parse(stripFences(textBlock.text));
}

function downscaleToDataUrl(imgOrCanvas, maxDim) {
  const w0 = imgOrCanvas.width, h0 = imgOrCanvas.height;
  const scale = Math.min(1, maxDim / Math.max(w0, h0));
  const w = Math.round(w0 * scale), h = Math.round(h0 * scale);
  const c = document.createElement("canvas");
  c.width = w; c.height = h;
  c.getContext("2d").drawImage(imgOrCanvas, 0, 0, w, h);
  return c.toDataURL("image/png");
}

function measureLighting(img) {
  const c = document.createElement("canvas");
  const w = 64, h = Math.round((img.height / img.width) * 64) || 64;
  c.width = w; c.height = h;
  const ctx = c.getContext("2d");
  ctx.drawImage(img, 0, 0, w, h);
  const data = ctx.getImageData(0, 0, w, h).data;
  let sumLum = 0, sumR = 0, sumB = 0, n = 0;
  for (let i = 0; i < data.length; i += 4) {
    const r = data[i], g = data[i + 1], b = data[i + 2];
    sumLum += (0.299 * r + 0.587 * g + 0.114 * b) / 255;
    sumR += r; sumB += b; n++;
  }
  return { luminance: sumLum / n, warmth: (sumR - sumB) / (n * 255) };
}

function clusterPhotos(photos, k) {
  const points = photos.map((p) => [p.lighting.luminance, p.lighting.warmth * 2]);
  let centroids = [];
  for (let i = 0; i < k; i++) centroids.push(points[Math.floor((i * points.length) / k)] || points[0]);
  let assignments = new Array(points.length).fill(0);
  for (let iter = 0; iter < 12; iter++) {
    assignments = points.map((p) => {
      let best = 0, bestD = Infinity;
      centroids.forEach((c, ci) => {
        const d = (p[0] - c[0]) ** 2 + (p[1] - c[1]) ** 2;
        if (d < bestD) { bestD = d; best = ci; }
      });
      return best;
    });
    centroids = centroids.map((c, ci) => {
      const members = points.filter((_, i) => assignments[i] === ci);
      if (members.length === 0) return c;
      const avg = members.reduce((a, m) => [a[0] + m[0], a[1] + m[1]], [0, 0]);
      return [avg[0] / members.length, avg[1] / members.length];
    });
  }
  return assignments;
}
function labelForCluster(avgLum, avgWarmth) {
  const bright = avgLum > 0.5 ? "Bright" : "Low-light";
  const warm = avgWarmth > 0.02 ? "warm" : avgWarmth < -0.02 ? "cool" : "neutral";
  return `${bright}, ${warm} tone`;
}

function defaultSettings() {
  return { tempShift: 0, tint: 0, exposure: 0, contrast: 0, blackPoint: 0, whitePoint: 255,
    highlights: 0, shadows: 0, saturation: 0, vibrance: 0, sharpen: 0, grain: 0, vignette: 0 };
}

function rgbToHsl(r, g, b) {
  r /= 255; g /= 255; b /= 255;
  const max = Math.max(r, g, b), min = Math.min(r, g, b);
  let h, s, l = (max + min) / 2;
  if (max === min) { h = s = 0; }
  else {
    const d = max - min;
    s = l > 0.5 ? d / (2 - max - min) : d / (max + min);
    switch (max) {
      case r: h = (g - b) / d + (g < b ? 6 : 0); break;
      case g: h = (b - r) / d + 2; break;
      default: h = (r - g) / d + 4;
    }
    h /= 6;
  }
  return [h, s, l];
}
function hslToRgb(h, s, l) {
  let r, g, b;
  if (s === 0) { r = g = b = l; }
  else {
    const hue2rgb = (p, q, t) => {
      if (t < 0) t += 1; if (t > 1) t -= 1;
      if (t < 1 / 6) return p + (q - p) * 6 * t;
      if (t < 1 / 2) return q;
      if (t < 2 / 3) return p + (q - p) * (2 / 3 - t) * 6;
      return p;
    };
    const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
    const p = 2 * l - q;
    r = hue2rgb(p, q, h + 1 / 3); g = hue2rgb(p, q, h); b = hue2rgb(p, q, h - 1 / 3);
  }
  return [r * 255, g * 255, b * 255];
}

function applyGrade(sourceCanvas, settings) {
  const w = sourceCanvas.width, h = sourceCanvas.height;
  const out = document.createElement("canvas");
  out.width = w; out.height = h;
  const octx = out.getContext("2d");
  octx.drawImage(sourceCanvas, 0, 0);
  const imageData = octx.getImageData(0, 0, w, h);
  const data = imageData.data;

  const expMul = Math.pow(2, settings.exposure / 100);
  const contrastFactor = (259 * (settings.contrast + 255)) / (255 * (259 - settings.contrast));
  const tempR = 1 + settings.tempShift / 300;
  const tempB = 1 - settings.tempShift / 300;
  const tintG = 1 - settings.tint / 300;
  const blackPt = settings.blackPoint, whitePt = settings.whitePoint;
  const cx = w / 2, cy = h / 2, maxDist = Math.hypot(cx, cy);

  for (let i = 0; i < data.length; i += 4) {
    let r = data[i], g = data[i + 1], b = data[i + 2];
    r *= tempR; b *= tempB; g *= tintG;
    r *= expMul; g *= expMul; b *= expMul;

    const lum = (0.299 * r + 0.587 * g + 0.114 * b) / 255;
    if (settings.highlights !== 0) {
      const hlAmt = (settings.highlights / 100) * Math.pow(Math.max(0, lum), 2);
      r -= hlAmt * 60; g -= hlAmt * 60; b -= hlAmt * 60;
    }
    if (settings.shadows !== 0) {
      const shAmt = (settings.shadows / 100) * Math.pow(Math.max(0, 1 - lum), 2);
      r += shAmt * 60; g += shAmt * 60; b += shAmt * 60;
    }

    r = ((r - blackPt) / (whitePt - blackPt)) * 255;
    g = ((g - blackPt) / (whitePt - blackPt)) * 255;
    b = ((b - blackPt) / (whitePt - blackPt)) * 255;

    r = contrastFactor * (r - 128) + 128;
    g = contrastFactor * (g - 128) + 128;
    b = contrastFactor * (b - 128) + 128;

    r = Math.max(0, Math.min(255, r));
    g = Math.max(0, Math.min(255, g));
    b = Math.max(0, Math.min(255, b));

    if (settings.saturation !== 0 || settings.vibrance !== 0) {
      const [hh, ss, ll] = rgbToHsl(r, g, b);
      let newS = ss * (1 + settings.saturation / 100);
      if (settings.vibrance !== 0) newS += (settings.vibrance / 100) * (1 - ss);
      newS = Math.max(0, Math.min(1, newS));
      const [nr, ng, nb] = hslToRgb(hh, newS, ll);
      r = nr; g = ng; b = nb;
    }

    if (settings.vignette > 0) {
      const x = (i / 4) % w, y = Math.floor((i / 4) / w);
      const dist = Math.hypot(x - cx, y - cy) / maxDist;
      const falloff = 1 - (settings.vignette / 100) * Math.pow(dist, 2.2);
      r *= falloff; g *= falloff; b *= falloff;
    }
    if (settings.grain > 0) {
      const noise = (Math.random() - 0.5) * settings.grain * 0.6;
      r += noise; g += noise; b += noise;
    }

    data[i] = Math.max(0, Math.min(255, r));
    data[i + 1] = Math.max(0, Math.min(255, g));
    data[i + 2] = Math.max(0, Math.min(255, b));
  }
  octx.putImageData(imageData, 0, 0);
  if (settings.sharpen > 0) return unsharpMask(out, settings.sharpen / 100);
  return out;
}

function unsharpMask(canvas, amount) {
  const w = canvas.width, h = canvas.height;
  const ctx = canvas.getContext("2d");
  const original = ctx.getImageData(0, 0, w, h);
  const blurCanvas = document.createElement("canvas");
  blurCanvas.width = w; blurCanvas.height = h;
  const bctx = blurCanvas.getContext("2d");
  bctx.filter = "blur(1.5px)";
  bctx.drawImage(canvas, 0, 0);
  const blurred = bctx.getImageData(0, 0, w, h);
  const out = ctx.createImageData(w, h);
  for (let i = 0; i < original.data.length; i += 4) {
    for (let c = 0; c < 3; c++) {
      const orig = original.data[i + c], blur = blurred.data[i + c];
      out.data[i + c] = Math.max(0, Math.min(255, orig + (orig - blur) * amount * 1.5));
    }
    out.data[i + 3] = original.data[i + 3];
  }
  ctx.putImageData(out, 0, 0);
  return canvas;
}

function inpaintRegion(imageData, w, h, boxX, boxY, boxW, boxH, iterations = 120) {
  const data = imageData.data;
  const pad = Math.round(Math.max(boxW, boxH) * 0.15) + 3;
  const x0 = Math.max(0, boxX - pad), y0 = Math.max(0, boxY - pad);
  const x1 = Math.min(w - 1, boxX + boxW + pad), y1 = Math.min(h - 1, boxY + boxH + pad);
  const mx0 = boxX, my0 = boxY, mx1 = boxX + boxW, my1 = boxY + boxH;
  const idx = (x, y) => (y * w + x) * 4;
  const isMasked = (x, y) => x >= mx0 && x < mx1 && y >= my0 && y < my1;
  let sumR = 0, sumG = 0, sumB = 0, count = 0;
  for (let y = y0; y <= y1; y++) for (let x = x0; x <= x1; x++) {
    if (!isMasked(x, y)) { const i = idx(x, y); sumR += data[i]; sumG += data[i + 1]; sumB += data[i + 2]; count++; }
  }
  const avgR = count ? sumR / count : 128, avgG = count ? sumG / count : 128, avgB = count ? sumB / count : 128;
  for (let y = my0; y < my1; y++) for (let x = mx0; x < mx1; x++) {
    const i = idx(x, y); data[i] = avgR; data[i + 1] = avgG; data[i + 2] = avgB;
  }
  for (let iter = 0; iter < iterations; iter++) {
    for (let y = my0; y < my1; y++) for (let x = mx0; x < mx1; x++) {
      const i = idx(x, y);
      let rSum = 0, gSum = 0, bSum = 0, n = 0;
      for (const [nx, ny] of [[x - 1, y], [x + 1, y], [x, y - 1], [x, y + 1]]) {
        if (nx < x0 || nx > x1 || ny < y0 || ny > y1) continue;
        const ni = idx(nx, ny); rSum += data[ni]; gSum += data[ni + 1]; bSum += data[ni + 2]; n++;
      }
      if (n > 0) { data[i] = rSum / n; data[i + 1] = gSum / n; data[i + 2] = bSum / n; }
    }
  }
  return imageData;
}

function clamp(v, min, max) { return Math.max(min, Math.min(max, v)); }
function photoSrc(p) { return p.processedCanvas ? p.processedCanvas.toDataURL() : p.img.src; }

function defaultGeometry() {
  return { rotate90: 0, flipH: false, straighten: 0, cropTop: 0, cropBottom: 0, cropLeft: 0, cropRight: 0 };
}

// geometry only: 90-degree turns, horizontal flip, small straighten angle, edge crop insets.
// this never touches pixel content, only which pixels are kept and how the frame is oriented.
function applyGeometry(source, geo) {
  const srcW = source.width, srcH = source.height;
  let w = srcW, h = srcH;
  if (geo.rotate90 % 2 !== 0) { w = srcH; h = srcW; }
  const c1 = document.createElement("canvas");
  c1.width = w; c1.height = h;
  const ctx1 = c1.getContext("2d");
  ctx1.save();
  ctx1.translate(w / 2, h / 2);
  ctx1.rotate((geo.rotate90 * 90 * Math.PI) / 180);
  if (geo.flipH) ctx1.scale(-1, 1);
  ctx1.drawImage(source, -srcW / 2, -srcH / 2);
  ctx1.restore();

  let c2 = c1;
  if (geo.straighten) {
    const rad = (geo.straighten * Math.PI) / 180;
    const overscale = 1 + Math.min(0.35, (Math.abs(geo.straighten) / 10) * 0.15);
    const tmp = document.createElement("canvas");
    tmp.width = c1.width; tmp.height = c1.height;
    const tctx = tmp.getContext("2d");
    tctx.save();
    tctx.translate(c1.width / 2, c1.height / 2);
    tctx.rotate(rad);
    tctx.scale(overscale, overscale);
    tctx.drawImage(c1, -c1.width / 2, -c1.height / 2);
    tctx.restore();
    c2 = tmp;
  }

  const cw = c2.width, ch = c2.height;
  const left = Math.round(cw * ((geo.cropLeft || 0) / 100));
  const right = Math.round(cw * ((geo.cropRight || 0) / 100));
  const top = Math.round(ch * ((geo.cropTop || 0) / 100));
  const bottom = Math.round(ch * ((geo.cropBottom || 0) / 100));
  const outW = Math.max(1, cw - left - right), outH = Math.max(1, ch - top - bottom);
  const c3 = document.createElement("canvas");
  c3.width = outW; c3.height = outH;
  c3.getContext("2d").drawImage(c2, left, top, outW, outH, 0, 0, outW, outH);
  return c3;
}

// --- minimal zip writer (store/no compression) so "download all" needs no external library ---
const CRC_TABLE = (() => {
  const table = [];
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    table[n] = c;
  }
  return table;
})();
function crc32(buf) {
  let crc = 0 ^ -1;
  for (let i = 0; i < buf.length; i++) crc = (crc >>> 8) ^ CRC_TABLE[(crc ^ buf[i]) & 0xff];
  return (crc ^ -1) >>> 0;
}
function dataUrlToUint8Array(dataUrl) {
  const binary = atob(dataUrl.split(",")[1]);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return bytes;
}
function createZip(files) {
  const encoder = new TextEncoder();
  const localParts = [], centralParts = [];
  let offset = 0;
  const now = new Date();
  const dosTime = ((now.getHours() << 11) | (now.getMinutes() << 5) | (now.getSeconds() / 2)) & 0xffff;
  const dosDate = (((now.getFullYear() - 1980) << 9) | ((now.getMonth() + 1) << 5) | now.getDate()) & 0xffff;

  files.forEach((f) => {
    const nameBytes = encoder.encode(f.name);
    const data = f.data;
    const crc = crc32(data);
    const localHeader = new Uint8Array(30 + nameBytes.length);
    const dv = new DataView(localHeader.buffer);
    dv.setUint32(0, 0x04034b50, true);
    dv.setUint16(4, 20, true);
    dv.setUint16(6, 0, true);
    dv.setUint16(8, 0, true);
    dv.setUint16(10, dosTime, true);
    dv.setUint16(12, dosDate, true);
    dv.setUint32(14, crc, true);
    dv.setUint32(18, data.length, true);
    dv.setUint32(22, data.length, true);
    dv.setUint16(26, nameBytes.length, true);
    dv.setUint16(28, 0, true);
    localHeader.set(nameBytes, 30);
    localParts.push(localHeader, data);

    const centralHeader = new Uint8Array(46 + nameBytes.length);
    const cdv = new DataView(centralHeader.buffer);
    cdv.setUint32(0, 0x02014b50, true);
    cdv.setUint16(4, 20, true);
    cdv.setUint16(6, 20, true);
    cdv.setUint16(8, 0, true);
    cdv.setUint16(10, 0, true);
    cdv.setUint16(12, dosTime, true);
    cdv.setUint16(14, dosDate, true);
    cdv.setUint32(16, crc, true);
    cdv.setUint32(20, data.length, true);
    cdv.setUint32(24, data.length, true);
    cdv.setUint16(28, nameBytes.length, true);
    cdv.setUint16(30, 0, true);
    cdv.setUint16(32, 0, true);
    cdv.setUint16(34, 0, true);
    cdv.setUint16(36, 0, true);
    cdv.setUint32(38, 0, true);
    cdv.setUint32(42, offset, true);
    centralHeader.set(nameBytes, 46);
    centralParts.push(centralHeader);

    offset += localHeader.length + data.length;
  });

  const centralSize = centralParts.reduce((s, p) => s + p.length, 0);
  const centralOffset = offset;
  const eocd = new Uint8Array(22);
  const edv = new DataView(eocd.buffer);
  edv.setUint32(0, 0x06054b50, true);
  edv.setUint16(8, files.length, true);
  edv.setUint16(10, files.length, true);
  edv.setUint32(12, centralSize, true);
  edv.setUint32(16, centralOffset, true);

  return new Blob([...localParts, ...centralParts, eocd], { type: "application/zip" });
}

function App() {
  const [apiKey, setApiKey] = useState(() => localStorage.getItem(API_KEY_STORAGE) || "");
  const [showSettings, setShowSettings] = useState(false);
  const [step, setStep] = useState("reference");
  const [referenceImages, setReferenceImages] = useState([]);
  const [referenceProfile, setReferenceProfile] = useState(null);
  const [analyzingRef, setAnalyzingRef] = useState(false);
  const [photos, setPhotos] = useState([]);
  const [groups, setGroups] = useState([]);
  const [numGroups, setNumGroups] = useState(3);
  const [status, setStatus] = useState("");
  const [cleanupSuggestions, setCleanupSuggestions] = useState({});
  const [checkingCleanup, setCheckingCleanup] = useState(null);

  const refInputRef = useRef(null);
  const batchInputRef = useRef(null);

  useEffect(() => {
    localStorage.setItem(API_KEY_STORAGE, apiKey);
  }, [apiKey]);

  const needsKeyNotice = (fn) => async (...args) => {
    if (!apiKey) {
      setStatus("Add your Anthropic API key in Settings to use AI-assisted features.");
      setShowSettings(true);
      setTimeout(() => setStatus(""), 4000);
      return;
    }
    return fn(...args);
  };

  const handleRefUpload = async (e) => {
    const files = Array.from(e.target.files || []);
    const dataUrls = await Promise.all(files.map(fileToDataUrl));
    const imgs = await Promise.all(dataUrls.map(loadImage));
    setReferenceImages(imgs);
    e.target.value = "";
  };

  const analyzeReference = needsKeyNotice(async () => {
    if (referenceImages.length === 0) return;
    setAnalyzingRef(true);
    setStatus("Analyzing the reference aesthetic...");
    try {
      const small = referenceImages.slice(0, 4).map((img) => downscaleToDataUrl(img, 700));
      const profile = await callClaudeVision(apiKey, REFERENCE_PROMPT, small);
      setReferenceProfile(profile);
      setStatus("Reference profile ready.");
    } catch (err) {
      setStatus("Couldn't analyze the reference photos: " + (err.message || "error"));
    }
    setAnalyzingRef(false);
    setTimeout(() => setStatus(""), 4000);
  });

  const handleBatchUpload = async (e) => {
    const files = Array.from(e.target.files || []);
    setStatus(`Loading ${files.length} photo(s)...`);
    const loaded = [];
    for (const file of files) {
      const dataUrl = await fileToDataUrl(file);
      const img = await loadImage(dataUrl);
      const lighting = measureLighting(img);
      loaded.push({ id: nextId++, name: file.name, img, lighting, processedCanvas: null, groupId: null });
    }
    setPhotos((prev) => [...prev, ...loaded]);
    setStatus(`${loaded.length} photo(s) loaded.`);
    e.target.value = "";
    setTimeout(() => setStatus(""), 2000);
  };

  const runGrouping = () => {
    if (photos.length === 0 || !referenceProfile) return;
    const k = Math.min(numGroups, photos.length);
    const assignments = clusterPhotos(photos, k);
    const newGroups = [];
    for (let gi = 0; gi < k; gi++) {
      const memberIdx = assignments.map((a, i) => (a === gi ? i : -1)).filter((i) => i >= 0);
      if (memberIdx.length === 0) continue;
      const avgLum = memberIdx.reduce((s, i) => s + photos[i].lighting.luminance, 0) / memberIdx.length;
      const avgWarmth = memberIdx.reduce((s, i) => s + photos[i].lighting.warmth, 0) / memberIdx.length;
      const targetLum = 0.45;
      const exposureCorrection = Math.round((targetLum - avgLum) * 150);
      const tempCorrection = Math.round(-avgWarmth * 80);
      const settings = {
        ...defaultSettings(),
        tempShift: clamp(referenceProfile.tempShift + tempCorrection, -100, 100),
        tint: referenceProfile.tint,
        exposure: clamp(referenceProfile.exposure + exposureCorrection, -100, 100),
        contrast: referenceProfile.contrast,
        blackPoint: referenceProfile.blackPoint,
        whitePoint: referenceProfile.whitePoint,
        highlights: referenceProfile.highlights,
        shadows: referenceProfile.shadows,
        saturation: referenceProfile.saturation,
        vibrance: referenceProfile.vibrance,
        sharpen: referenceProfile.sharpen,
        grain: referenceProfile.grain,
        vignette: referenceProfile.vignette,
      };
      newGroups.push({ id: nextId++, label: labelForCluster(avgLum, avgWarmth), photoIds: memberIdx.map((i) => photos[i].id), settings, geometry: defaultGeometry(), approved: false });
    }
    setGroups(newGroups);
    setStep("review");
  };

  const updateGroupSetting = (groupId, key, value) => {
    setGroups((prev) => prev.map((g) => (g.id === groupId ? { ...g, settings: { ...g.settings, [key]: value } } : g)));
  };
  const updateGroupGeometry = (groupId, key, value) => {
    setGroups((prev) => prev.map((g) => (g.id === groupId ? { ...g, geometry: { ...g.geometry, [key]: value } } : g)));
  };

  const suggestGeometry = needsKeyNotice(async (groupId) => {
    const group = groups.find((g) => g.id === groupId);
    if (!group) return;
    const repPhoto = photos.find((p) => p.id === group.photoIds[0]);
    if (!repPhoto) return;
    setStatus("Checking composition...");
    try {
      const dataUrl = downscaleToDataUrl(repPhoto.img, 1024);
      const r = await callClaudeVision(apiKey, GEOMETRY_PROMPT, [dataUrl]);
      setGroups((prev) => prev.map((g) => g.id === groupId ? {
        ...g, geometry: { ...g.geometry, straighten: r.rotationDeg || 0,
          cropTop: r.cropTopPct || 0, cropBottom: r.cropBottomPct || 0,
          cropLeft: r.cropLeftPct || 0, cropRight: r.cropRightPct || 0 },
      } : g));
      setStatus("Composition suggestion applied to the sliders — review before applying.");
    } catch (err) {
      setStatus("Couldn't get a composition suggestion: " + (err.message || "error"));
    }
    setTimeout(() => setStatus(""), 3000);
  });

  const applyGroup = (groupId) => {
    const group = groups.find((g) => g.id === groupId);
    if (!group) return;
    setPhotos((prev) => prev.map((p) => {
      if (!group.photoIds.includes(p.id)) return p;
      const geoCanvas = applyGeometry(p.img, group.geometry);
      const processed = applyGrade(geoCanvas, group.settings);
      return { ...p, processedCanvas: processed, groupId: group.id };
    }));
    setGroups((prev) => prev.map((g) => (g.id === groupId ? { ...g, approved: true } : g)));
  };

  const checkCleanup = needsKeyNotice(async (photo) => {
    setCheckingCleanup(photo.id);
    setStatus("Checking for optional touch-ups...");
    try {
      const source = photo.processedCanvas || (() => {
        const c = document.createElement("canvas");
        c.width = photo.img.width; c.height = photo.img.height;
        c.getContext("2d").drawImage(photo.img, 0, 0);
        return c;
      })();
      const dataUrl = downscaleToDataUrl(source, 1024);
      const results = await callClaudeVision(apiKey, RETOUCH_PROMPT, [dataUrl]);
      const mapped = results.map((r) => ({
        id: nextId++, label: r.label, category: r.category, confidence: r.confidence,
        box: { x: Math.round(r.box[0] * source.width), y: Math.round(r.box[1] * source.height),
               w: Math.round(r.box[2] * source.width), h: Math.round(r.box[3] * source.height) },
        status: "pending",
      }));
      setCleanupSuggestions((prev) => ({ ...prev, [photo.id]: mapped }));
      setStatus(mapped.length ? `Found ${mapped.length} possible touch-up(s).` : "Nothing flagged.");
    } catch (err) {
      setStatus("Couldn't check this photo: " + (err.message || "error"));
    }
    setCheckingCleanup(null);
    setTimeout(() => setStatus(""), 3000);
  });

  const applyCleanupSuggestion = (photo, sugg) => {
    const source = photo.processedCanvas || (() => {
      const c = document.createElement("canvas");
      c.width = photo.img.width; c.height = photo.img.height;
      c.getContext("2d").drawImage(photo.img, 0, 0);
      return c;
    })();
    const w = source.width, h = source.height;
    const off = document.createElement("canvas");
    off.width = w; off.height = h;
    const octx = off.getContext("2d");
    octx.drawImage(source, 0, 0);
    const imageData = octx.getImageData(0, 0, w, h);
    inpaintRegion(imageData, w, h, sugg.box.x, sugg.box.y, sugg.box.w, sugg.box.h);
    octx.putImageData(imageData, 0, 0);
    setPhotos((prev) => prev.map((p) => (p.id === photo.id ? { ...p, processedCanvas: off } : p)));
    setCleanupSuggestions((prev) => ({ ...prev, [photo.id]: prev[photo.id].map((s) => (s.id === sugg.id ? { ...s, status: "applied" } : s)) }));
  };

  const downloadPhoto = (photo) => {
    const canvas = photo.processedCanvas;
    if (!canvas) return;
    const a = document.createElement("a");
    a.href = canvas.toDataURL("image/jpeg", 0.92);
    a.download = `edited-${photo.name.replace(/\.[^.]+$/, "")}.jpg`;
    a.click();
  };

  const [zipping, setZipping] = useState(false);
  const downloadAllZip = () => {
    const ready = photos.filter((p) => p.processedCanvas);
    if (ready.length === 0) return;
    setZipping(true);
    setStatus(`Zipping ${ready.length} photo(s)...`);
    setTimeout(() => {
      try {
        const files = ready.map((p) => ({
          name: `edited-${p.name.replace(/\.[^.]+$/, "")}.jpg`,
          data: dataUrlToUint8Array(p.processedCanvas.toDataURL("image/jpeg", 0.92)),
        }));
        const blob = createZip(files);
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = "edited-photos.zip";
        a.click();
        URL.revokeObjectURL(url);
        setStatus("Zip downloaded.");
      } catch (err) {
        setStatus("Couldn't build the zip: " + (err.message || "error"));
      }
      setZipping(false);
      setTimeout(() => setStatus(""), 2500);
    }, 30);
  };

  const allApproved = groups.length > 0 && groups.every((g) => g.approved);

  return (
    <div style={{ minHeight: "100vh", paddingBottom: 40 }}>
      <div style={{ position: "sticky", top: 0, background: "#1a1a1a", zIndex: 10, padding: "14px 16px 10px",
        borderBottom: "1px solid #2a2a2a", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <div style={{ fontSize: 16, fontWeight: 600 }}>Photo Developer</div>
        <button onClick={() => setShowSettings((s) => !s)} style={iconBtnStyle}>{"\u2699"}</button>
      </div>

      {showSettings && (
        <div style={{ padding: 16, background: "#202020", borderBottom: "1px solid #2a2a2a" }}>
          <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 6 }}>Anthropic API key</div>
          <div style={{ fontSize: 11, color: "#9a958a", marginBottom: 8, lineHeight: 1.5 }}>
            Needed only for AI-assisted reference analysis and touch-up suggestions. Stored on this device only —
            never share this app's URL with your key entered, since it would be visible to anyone using it.
          </div>
          <input type="password" value={apiKey} onChange={(e) => setApiKey(e.target.value)}
            placeholder="sk-ant-..." style={{ ...inputStyle, marginBottom: 8 }} />
          <div style={{ fontSize: 11, color: "#6b675f" }}>Grading (colors, curves, crop) always works offline, with no key.</div>
        </div>
      )}

      <div style={{ padding: 16, maxWidth: 720, margin: "0 auto" }}>
        <div style={{ display: "flex", gap: 6, marginBottom: 14 }}>
          {["reference", "upload", "review", "cleanup"].map((s) => (
            <button key={s} onClick={() => setStep(s)} style={{ ...btnStyle(step === s), flex: 1, fontSize: 12, padding: "10px 4px" }}>
              {{ reference: "Reference", upload: "Upload", review: "Review", cleanup: "Cleanup" }[s]}
            </button>
          ))}
        </div>

        {status && <div style={{ marginBottom: 12, fontSize: 13, color: "#c9a15c" }}>{status}</div>}

        {step === "reference" && (
          <div>
            <input ref={refInputRef} type="file" accept="image/*" multiple onChange={handleRefUpload} style={{ display: "none" }} />
            <button onClick={() => refInputRef.current.click()} style={{ ...btnStyle(true), width: "100%" }}>
              Upload photographer's sample photos
            </button>
            <div style={{ display: "flex", gap: 8, marginTop: 12, flexWrap: "wrap" }}>
              {referenceImages.map((img, i) => (
                <img key={i} src={img.src} alt="" style={{ width: 84, height: 84, objectFit: "cover", borderRadius: 8 }} />
              ))}
            </div>
            {referenceImages.length > 0 && (
              <button onClick={analyzeReference} disabled={analyzingRef} style={{ ...btnStyle(false, analyzingRef), width: "100%", marginTop: 12 }}>
                {analyzingRef ? "Analyzing..." : "Analyze aesthetic"}
              </button>
            )}
            {referenceProfile && (
              <div style={{ marginTop: 16, padding: 14, background: "#232323", borderRadius: 10 }}>
                <div style={{ fontSize: 13, color: "#c9a15c", marginBottom: 10 }}>{referenceProfile.description}</div>
                <ProfileGrid settings={referenceProfile} onChange={(k, v) => setReferenceProfile((p) => ({ ...p, [k]: v }))} />
                <button onClick={() => setStep("upload")} style={{ ...btnStyle(true), width: "100%", marginTop: 14 }}>Continue</button>
              </div>
            )}
          </div>
        )}

        {step === "upload" && (
          <div>
            <input ref={batchInputRef} type="file" accept="image/*" multiple onChange={handleBatchUpload} style={{ display: "none" }} />
            <button onClick={() => batchInputRef.current.click()} style={{ ...btnStyle(true), width: "100%" }}>Add photos</button>
            <div style={{ fontSize: 12, color: "#8f8b81", margin: "10px 0" }}>{photos.length} photo(s) loaded</div>
            <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
              {photos.map((p) => (
                <img key={p.id} src={p.img.src} alt="" style={{ width: 76, height: 76, objectFit: "cover", borderRadius: 8 }} />
              ))}
            </div>
            {photos.length > 0 && (
              <div style={{ marginTop: 18 }}>
                <label style={{ ...labelStyle, marginBottom: 6 }}>Lighting groups <span style={monoStyle}>{numGroups}</span></label>
                <input type="range" min="1" max="6" value={numGroups} onChange={(e) => setNumGroups(Number(e.target.value))} />
                <button onClick={runGrouping} style={{ ...btnStyle(true), width: "100%", marginTop: 10 }}>Detect lighting groups</button>
              </div>
            )}
          </div>
        )}

        {step === "review" && (
          <div>
            {groups.map((g) => (
              <div key={g.id} style={{ marginBottom: 18, padding: 14, background: "#232323", borderRadius: 10,
                border: g.approved ? "1px solid #5c8a5c" : "1px solid #3a3a3a" }}>
                <div style={{ fontSize: 14, fontWeight: 600, marginBottom: 8 }}>{g.label} · {g.photoIds.length} photo(s)</div>
                <div style={{ display: "flex", gap: 6, marginBottom: 10, flexWrap: "wrap" }}>
                  {photos.filter((p) => g.photoIds.includes(p.id)).slice(0, 6).map((p) => (
                    <img key={p.id} src={photoSrc(p)} alt=""
                      style={{ width: 60, height: 60, objectFit: "cover", borderRadius: 6 }} />
                  ))}
                </div>
                <div style={{ marginBottom: 10 }}>
                  <div style={{ fontSize: 12, fontWeight: 600, color: "#b0aca2", marginBottom: 6 }}>Geometry (crop, rotate, straighten)</div>
                  <div style={{ display: "flex", gap: 6, marginBottom: 8 }}>
                    <button onClick={() => updateGroupGeometry(g.id, "rotate90", (g.geometry.rotate90 + 3) % 4)} style={{ ...btnStyle(false), flex: 1, fontSize: 11, padding: "6px 4px" }}>⟲ Rotate</button>
                    <button onClick={() => updateGroupGeometry(g.id, "rotate90", (g.geometry.rotate90 + 1) % 4)} style={{ ...btnStyle(false), flex: 1, fontSize: 11, padding: "6px 4px" }}>⟳ Rotate</button>
                    <button onClick={() => updateGroupGeometry(g.id, "flipH", !g.geometry.flipH)} style={{ ...btnStyle(g.geometry.flipH), flex: 1, fontSize: 11, padding: "6px 4px" }}>Flip</button>
                  </div>
                  <button onClick={() => suggestGeometry(g.id)} style={{ ...btnStyle(false), width: "100%", fontSize: 11, padding: "6px 4px", marginBottom: 8 }}>Auto-suggest straighten & crop</button>
                  <label style={labelStyle}>Straighten <span style={monoStyle}>{g.geometry.straighten}°</span></label>
                  <input type="range" min="-10" max="10" value={g.geometry.straighten} onChange={(e) => updateGroupGeometry(g.id, "straighten", Number(e.target.value))} style={{ width: "100%", marginBottom: 6 }} />
                  {["cropTop", "cropBottom", "cropLeft", "cropRight"].map((k) => (
                    <div key={k}>
                      <label style={labelStyle}>{{ cropTop: "Crop top", cropBottom: "Crop bottom", cropLeft: "Crop left", cropRight: "Crop right" }[k]} <span style={monoStyle}>{g.geometry[k]}%</span></label>
                      <input type="range" min="0" max="30" value={g.geometry[k]} onChange={(e) => updateGroupGeometry(g.id, k, Number(e.target.value))} style={{ width: "100%", marginBottom: 6 }} />
                    </div>
                  ))}
                </div>
                <ProfileGrid settings={g.settings} onChange={(k, v) => updateGroupSetting(g.id, k, v)} />
                <button onClick={() => applyGroup(g.id)} style={{ ...btnStyle(g.approved), width: "100%", marginTop: 10 }}>
                  {g.approved ? "Re-apply" : "Apply to group"}
                </button>
              </div>
            ))}
            {allApproved && (
              <button onClick={() => setStep("cleanup")} style={{ ...btnStyle(true), width: "100%" }}>Continue to optional cleanup</button>
            )}
          </div>
        )}

        {step === "cleanup" && (
          <div>
            <div style={{ fontSize: 12, color: "#c9a15c", marginBottom: 14, lineHeight: 1.5 }}>
              Separate from grading above — this can remove small things like a bra strap or sleep line, only where you approve it.
            </div>
            <button onClick={downloadAllZip} disabled={zipping || photos.every((p) => !p.processedCanvas)}
              style={{ ...btnStyle(true, zipping || photos.every((p) => !p.processedCanvas)), width: "100%", marginBottom: 14 }}>
              {zipping ? "Zipping..." : "Download all as ZIP"}
            </button>
            <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
              {photos.map((p) => (
                <div key={p.id} style={{ padding: 12, background: "#232323", borderRadius: 10 }}>
                  <div style={{ display: "flex", gap: 12 }}>
                    <img src={photoSrc(p)} alt="" style={{ width: 90, height: 90, objectFit: "cover", borderRadius: 8, flexShrink: 0 }} />
                    <div style={{ flex: 1, minWidth: 0 }}>
                      <div style={{ fontSize: 12, marginBottom: 8, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{p.name}</div>
                      <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
                        <button onClick={() => checkCleanup(p)} disabled={checkingCleanup === p.id} style={{ ...btnStyle(false, checkingCleanup === p.id), fontSize: 11, padding: "6px 8px" }}>
                          {checkingCleanup === p.id ? "Checking..." : "Check touch-ups"}
                        </button>
                        <button onClick={() => downloadPhoto(p)} disabled={!p.processedCanvas} style={{ ...btnStyle(false, !p.processedCanvas), fontSize: 11, padding: "6px 8px" }}>Download</button>
                      </div>
                    </div>
                  </div>
                  {(cleanupSuggestions[p.id] || []).map((s) => (
                    <div key={s.id} style={{ fontSize: 12, display: "flex", justifyContent: "space-between", alignItems: "center", padding: "8px 0", borderTop: "1px solid #2c2c2c", marginTop: 8 }}>
                      <span>{s.label}</span>
                      {s.status === "pending" ? (
                        <div style={{ display: "flex", gap: 6 }}>
                          <button onClick={() => applyCleanupSuggestion(p, s)} style={{ ...btnStyle(true), padding: "5px 10px", fontSize: 11 }}>Approve</button>
                          <button onClick={() => setCleanupSuggestions((prev) => ({ ...prev, [p.id]: prev[p.id].filter((x) => x.id !== s.id) }))} style={{ ...btnStyle(false), padding: "5px 10px", fontSize: 11 }}>Dismiss</button>
                        </div>
                      ) : (<span style={{ color: "#7fb37f" }}>Applied</span>)}
                    </div>
                  ))}
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

const SLIDER_DEFS = [
  ["tempShift", "Temperature", -100, 100], ["tint", "Tint", -100, 100],
  ["exposure", "Exposure", -100, 100], ["contrast", "Contrast", -100, 100],
  ["blackPoint", "Black point", 0, 40], ["whitePoint", "White point", 215, 255],
  ["highlights", "Highlights", -100, 100], ["shadows", "Shadows", -100, 100],
  ["saturation", "Saturation", -100, 100], ["vibrance", "Vibrance", -100, 100],
  ["sharpen", "Sharpen", 0, 100], ["grain", "Grain", 0, 100], ["vignette", "Vignette", 0, 100],
];

function ProfileGrid({ settings, onChange }) {
  return (
    <div>
      {SLIDER_DEFS.map(([key, label, min, max]) => (
        <div key={key} style={{ marginBottom: 8 }}>
          <label style={labelStyle}>{label} <span style={monoStyle}>{Math.round(settings[key] ?? 0)}</span></label>
          <input type="range" min={min} max={max} value={settings[key] ?? 0} onChange={(e) => onChange(key, Number(e.target.value))} style={{ width: "100%" }} />
        </div>
      ))}
    </div>
  );
}

function btnStyle(active, disabled) {
  return {
    padding: "10px 14px", borderRadius: 8,
    border: active ? "1px solid #b8874a" : "1px solid #3a3a3a",
    background: active ? "#3a3226" : "#232323",
    color: disabled ? "#5a5650" : "#e8e4dc",
    fontSize: 13, cursor: disabled ? "not-allowed" : "pointer", opacity: disabled ? 0.6 : 1,
  };
}
const iconBtnStyle = { background: "none", border: "none", color: "#e8e4dc", fontSize: 20, cursor: "pointer", padding: 4 };
const inputStyle = { width: "100%", padding: "10px 12px", borderRadius: 8, border: "1px solid #3a3a3a", background: "#232323", color: "#e8e4dc", fontSize: 14 };
const labelStyle = { display: "flex", justifyContent: "space-between", fontSize: 11, color: "#b0aca2", marginBottom: 2 };
const monoStyle = { fontFamily: "ui-monospace, monospace", color: "#e8e4dc" };

ReactDOM.createRoot(document.getElementById("root")).render(<App />);
