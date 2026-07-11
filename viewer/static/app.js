const $ = (selector) => document.querySelector(selector);
const SVG_NS = "http://www.w3.org/2000/svg";

const elements = {
  dataset: $("#dataset-select"), description: $("#dataset-description"),
  splitGroup: $("#split-group"), splitNote: $("#split-note"),
  modeGroup: $("#mode-group"), modeSection: $("#mode-section"),
  sequenceOptions: $("#sequence-options"), clipLength: $("#clip-length"),
  stride: $("#frame-stride"), showMask: $("#show-mask"),
  showBBox: $("#show-bbox"), showConfidence: $("#show-confidence"),
  confidenceRow: $("#confidence-row"), legend: $("#joint-legend"),
  context: $("#sample-context"), key: $("#sample-key"),
  badges: $("#heading-badges"), stats: $("#stats-row"),
  stage: $("#viewer-stage"), error: $("#error-panel"),
  loading: $("#loading-layer"), status: $("#topbar-status"),
  previous: $("#previous-button"), next: $("#next-button"),
  random: $("#random-button"), index: $("#index-input"), total: $("#total-label"),
};

const state = {
  catalog: null, spec: null, dataset: "climbing_videos", split: "train",
  mode: "frame", index: 0, sample: null, request: null,
};

function pretty(value) {
  const names = {trainval: "Train + val", frame: "Frame", sequence: "Sequence", all: "All"};
  return names[value] || value.charAt(0).toUpperCase() + value.slice(1);
}

function segment(container, values, active, onChange) {
  container.replaceChildren();
  for (const value of values) {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = pretty(value);
    button.className = value === active ? "active" : "";
    button.setAttribute("aria-pressed", String(value === active));
    button.addEventListener("click", () => onChange(value));
    container.append(button);
  }
}

function syncControls() {
  state.spec = state.catalog.datasets.find((d) => d.id === state.dataset);
  elements.description.textContent = state.spec.description;
  segment(elements.splitGroup, state.spec.splits, state.split, (split) => {
    state.split = split; state.index = 0; syncControls(); loadSample();
  });
  segment(elements.modeGroup, state.spec.modes, state.mode, (mode) => {
    state.mode = mode; state.index = 0; syncControls(); loadSample();
  });
  elements.modeSection.classList.toggle("hidden", state.spec.modes.length === 1);
  elements.sequenceOptions.classList.toggle("hidden", state.mode !== "sequence");
  const isVideo = state.spec.target === "joint";
  elements.confidenceRow.classList.toggle("hidden", !isVideo);
  elements.legend.classList.toggle("hidden", !isVideo);
  elements.splitNote.textContent = state.dataset === "climbing_videos"
    ? "Physical dataset split; browsing is deterministic and does not jitter windows."
    : "Split provided by the source dataset.";
}

function setBusy(busy) {
  elements.loading.classList.toggle("hidden", !busy);
  elements.status.classList.toggle("busy", busy);
  elements.status.lastElementChild.textContent = busy ? "Reading data" : "Ready";
}

function showError(message) {
  elements.error.textContent = message;
  elements.error.classList.remove("hidden");
  elements.stage.replaceChildren();
}

function query() {
  return new URLSearchParams({
    dataset: state.dataset, split: state.split, mode: state.mode,
    clip_length: elements.clipLength.value, stride: elements.stride.value,
    index: state.index, show_mask: elements.showMask.checked,
    show_bbox: elements.showBBox.checked,
  });
}

async function loadSample() {
  if (state.request) state.request.abort();
  state.request = new AbortController();
  setBusy(true);
  elements.error.classList.add("hidden");
  try {
    const response = await fetch(`/api/sample?${query()}`, {signal: state.request.signal});
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || `Request failed (${response.status})`);
    state.sample = payload;
    state.index = payload.index;
    render();
  } catch (error) {
    if (error.name !== "AbortError") showError(error.message);
  } finally {
    setBusy(false);
  }
}

function badge(text, kind = "") {
  const el = document.createElement("span");
  el.className = `badge ${kind}`;
  el.textContent = text;
  return el;
}

function stat(value, label) {
  const el = document.createElement("div");
  el.className = "stat-card";
  const strong = document.createElement("strong"); strong.textContent = value;
  const span = document.createElement("span"); span.textContent = label;
  el.append(strong, span);
  return el;
}

function renderHeading(sample) {
  elements.context.textContent = `${sample.dataset_label} / ${pretty(sample.split)} / ${pretty(sample.mode)}`;
  elements.key.textContent = sample.key;
  elements.badges.replaceChildren(
    badge(sample.target === "joint" ? "22-joint labels" : "Vertex labels"),
    badge(`${sample.frames.length} frame${sample.frames.length === 1 ? "" : "s"}`),
  );
  if (sample.target === "joint") {
    const invalid = sample.frames.filter((f) => !f.frame_valid).length;
    elements.badges.append(badge(invalid ? `${invalid} invalid` : "Frames valid", invalid ? "warn" : "good"));
  }
  elements.index.value = sample.index;
  elements.index.max = Math.max(0, sample.total - 1);
  elements.total.textContent = `of ${sample.total.toLocaleString()}`;
}

function renderStats(sample) {
  elements.stats.replaceChildren();
  if (sample.target === "joint") {
    const contacts = sample.frames.reduce((sum, frame) => sum + frame.contact_count, 0);
    const supervised = sample.frames.reduce((sum, frame) => sum + frame.supervised_count, 0);
    const confidence = sample.frames.flatMap((frame) =>
      frame.joint_confidence.filter((_, i) => frame.joint_supervised[i]));
    const mean = confidence.length ? confidence.reduce((a, b) => a + b, 0) / confidence.length : null;
    elements.stats.append(
      stat(contacts, "Contact labels"), stat(supervised, "Supervised joints"),
      stat(mean === null ? "—" : `${Math.round(mean * 100)}%`, "Mean confidence"),
      stat(sample.frames.filter((f) => f.frame_valid).length, "Valid frames"),
    );
  } else {
    const frame = sample.frames[0];
    elements.stats.append(
      stat(frame.contact_count.toLocaleString(), "Contact vertices"),
      stat(frame.vertex_count.toLocaleString(), "Total vertices"),
      stat(frame.topology.toUpperCase(), "Topology"),
    );
  }
}

function hexToRgb(hex) {
  const n = parseInt(hex.slice(1), 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

function mixColor(from, to, amount) {
  const a = hexToRgb(from), b = hexToRgb(to);
  const rgb = a.map((value, i) => Math.round(value + (b[i] - value) * amount));
  return `rgb(${rgb.join(",")})`;
}

function jointColor(contact, confidence) {
  const gray = "#a8adb5";
  const target = contact ? "#de3d45" : "#20a66a";
  return mixColor(gray, target, elements.showConfidence.checked ? confidence : 1);
}

function makeSkeleton(frame, compact = false) {
  const schema = state.catalog.skeleton;
  const svg = document.createElementNS(SVG_NS, "svg");
  svg.setAttribute("viewBox", "0 0 100 108");
  svg.setAttribute("class", "skeleton");
  svg.setAttribute("role", "img");
  svg.setAttribute("aria-label", "Canonical SMPL-X contact skeleton");
  for (const [a, b] of schema.edges) {
    const line = document.createElementNS(SVG_NS, "line");
    line.setAttribute("x1", schema.joint_coords[a][0]); line.setAttribute("y1", schema.joint_coords[a][1]);
    line.setAttribute("x2", schema.joint_coords[b][0]); line.setAttribute("y2", schema.joint_coords[b][1]);
    line.setAttribute("class", "bone"); svg.append(line);
  }
  schema.joint_coords.forEach(([x, y], index) => {
    const circle = document.createElementNS(SVG_NS, "circle");
    circle.setAttribute("cx", x); circle.setAttribute("cy", y);
    circle.setAttribute("r", compact ? 3.1 : 3.35);
    const supervised = frame.joint_supervised[index];
    circle.setAttribute("class", `joint${supervised ? "" : " unknown"}`);
    if (supervised) circle.setAttribute("fill", jointColor(
      frame.joint_contact[index], frame.joint_confidence[index]));
    const title = document.createElementNS(SVG_NS, "title");
    title.textContent = `${schema.joint_names[index]} — ${supervised
      ? `${frame.joint_contact[index] ? "contact" : "non-contact"}, ${Math.round(frame.joint_confidence[index] * 100)}% confidence`
      : "not supervised"}`;
    circle.append(title); svg.append(circle);
  });
  return svg;
}

function panel(title, detail, content, className = "") {
  const root = document.createElement("section"); root.className = `panel ${className}`;
  const header = document.createElement("div"); header.className = "panel-header";
  const strong = document.createElement("strong"); strong.textContent = title;
  const span = document.createElement("span"); span.textContent = detail;
  header.append(strong, span);
  const wrap = document.createElement("div"); wrap.className = "image-wrap"; wrap.append(content);
  root.append(header, wrap); return root;
}

function contactPills(frame) {
  const list = document.createElement("div"); list.className = "joint-list";
  state.catalog.skeleton.joint_names.forEach((name, i) => {
    if (!frame.joint_supervised[i]) return;
    const pill = document.createElement("span");
    pill.className = `joint-pill${frame.joint_contact[i] ? " contact" : ""}`;
    pill.textContent = frame.joint_contact[i]
      ? `${name} · ${Math.round(frame.joint_confidence[i] * 100)}%` : name;
    list.append(pill);
  });
  return list;
}

function renderSingleJoint(frame) {
  const layout = document.createElement("div"); layout.className = "single-layout";
  const image = new Image(); image.src = frame.image_url; image.alt = "Dataset frame";
  const imagePanel = panel("Source frame", frame.frame_valid ? "Valid person track" : "Invalid person track", image);
  const skeletonPanel = panel("Canonical body-22 skeleton", "Hover joints for labels", makeSkeleton(frame), "skeleton-panel");
  skeletonPanel.append(contactPills(frame));
  layout.append(imagePanel, skeletonPanel); return layout;
}

function renderSingleVertex(frame) {
  const layout = document.createElement("div"); layout.className = "single-layout";
  const image = new Image(); image.src = frame.image_url; image.alt = "Dataset image";
  const mesh = new Image(); mesh.src = frame.mesh_url; mesh.alt = "Canonical contact mesh";
  layout.append(
    panel("Source image", frame.bbox ? "Person annotation available" : "Full image", image),
    panel("Canonical contact surface", `${frame.contact_count.toLocaleString()} contact vertices`, mesh),
  );
  return layout;
}

function renderSequence(frames) {
  const grid = document.createElement("div"); grid.className = "sequence-grid";
  for (const frame of frames) {
    const card = document.createElement("article");
    card.className = `frame-card${frame.frame_valid ? "" : " invalid"}`;
    const head = document.createElement("div"); head.className = "frame-head";
    const title = document.createElement("strong"); title.textContent = `Frame ${frame.frame_position}`;
    const meta = document.createElement("span");
    meta.textContent = `${frame.time_sec.toFixed(3)} s · ${frame.contact_count} contact`;
    head.append(title, meta);
    const content = document.createElement("div"); content.className = "frame-content";
    const image = new Image(); image.className = "frame-image"; image.src = frame.image_url; image.alt = frame.key;
    const skeleton = document.createElement("div"); skeleton.className = "mini-skeleton";
    const stats = document.createElement("div"); stats.className = "mini-stats";
    stats.textContent = frame.frame_valid
      ? `${frame.supervised_count}/22 supervised` : "Invalid frame · labels masked";
    skeleton.append(makeSkeleton(frame, true), stats);
    content.append(image, skeleton); card.append(head, content); grid.append(card);
  }
  return grid;
}

function render() {
  const sample = state.sample;
  renderHeading(sample); renderStats(sample);
  elements.confidenceRow.classList.toggle("disabled", sample.target !== "joint" || !sample.confidence_available);
  elements.showConfidence.disabled = sample.target !== "joint" || !sample.confidence_available;
  elements.stage.replaceChildren(
    sample.target === "vertex" ? renderSingleVertex(sample.frames[0])
      : sample.mode === "sequence" ? renderSequence(sample.frames)
      : renderSingleJoint(sample.frames[0]),
  );
}

function navigate(delta) {
  if (!state.sample) return;
  state.index = (state.index + delta + state.sample.total) % state.sample.total;
  loadSample();
}

elements.previous.addEventListener("click", () => navigate(-1));
elements.next.addEventListener("click", () => navigate(1));
elements.random.addEventListener("click", () => {
  if (!state.sample) return;
  state.index = Math.floor(Math.random() * state.sample.total); loadSample();
});
elements.index.addEventListener("change", () => {
  if (!state.sample) return;
  state.index = Math.max(0, Math.min(state.sample.total - 1, Number(elements.index.value) || 0));
  loadSample();
});
elements.dataset.addEventListener("change", () => {
  state.dataset = elements.dataset.value;
  state.spec = state.catalog.datasets.find((d) => d.id === state.dataset);
  state.split = state.spec.default_split; state.mode = state.spec.modes[0]; state.index = 0;
  syncControls(); loadSample();
});
elements.clipLength.addEventListener("change", () => { state.index = 0; loadSample(); });
elements.stride.addEventListener("change", () => { state.index = 0; loadSample(); });
elements.showMask.addEventListener("change", loadSample);
elements.showBBox.addEventListener("change", loadSample);
elements.showConfidence.addEventListener("change", () => { if (state.sample) render(); });
document.addEventListener("keydown", (event) => {
  if (["INPUT", "SELECT"].includes(event.target.tagName)) return;
  if (event.key === "ArrowLeft") navigate(-1);
  if (event.key === "ArrowRight") navigate(1);
  if (event.key.toLowerCase() === "r") elements.random.click();
});

async function initialize() {
  try {
    const response = await fetch("/api/catalog");
    state.catalog = await response.json();
    for (const dataset of state.catalog.datasets) {
      const option = document.createElement("option"); option.value = dataset.id; option.textContent = dataset.label;
      elements.dataset.append(option);
    }
    elements.dataset.value = state.dataset;
    state.spec = state.catalog.datasets.find((d) => d.id === state.dataset);
    state.split = state.spec.default_split; state.mode = state.spec.modes[0];
    syncControls(); await loadSample();
  } catch (error) {
    showError(`Viewer initialization failed: ${error.message}`);
  }
}

initialize();
