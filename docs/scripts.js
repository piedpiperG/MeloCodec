const sampleIds = [1, 2, 3, 4, 5];

const codecs = [
  {
    key: "original",
    label: "Original",
    badge: "Reference",
    rates: ["source"],
    path: (sample) => `audio/original/${sample}.wav`,
  },
  {
    key: "bwc",
    label: "MeloCodec",
    badge: "Ours",
    rates: ["low", "high"],
    path: (sample, rate) => `audio/bwc/${rate}/${sample}.wav`,
    ours: true,
  },
  {
    key: "dac",
    label: "DAC",
    badge: "Baseline",
    rates: ["low", "high"],
    path: (sample, rate) => `audio/dac/${rate}/${sample}.wav`,
  },
  {
    key: "encodec",
    label: "Encodec",
    badge: "Baseline",
    rates: ["low", "high"],
    path: (sample, rate) => `audio/encodec/${rate}/${sample}.wav`,
  },
  {
    key: "xcodec",
    label: "X-Codec",
    badge: "Baseline",
    rates: ["low", "high"],
    path: (sample, rate) => `audio/xcodec/${rate}/${sample}.wav`,
  },
  {
    key: "mucodec",
    label: "MuCodec",
    badge: "Low only",
    rates: ["low"],
    path: (sample) => `audio/mucodec/low/${sample}.wav`,
  },
];

const pitchCases = [1, 2, 3];
const pitchVariants = [
  { label: "Original", path: (sample) => `audio/original/${sample}.wav` },
  { label: "Melody-shift set 2", path: (sample) => `audio/bwc/pitch_2/${sample}.wav` },
  { label: "Melody-shift set 5", path: (sample) => `audio/bwc/pitch_5/${sample}.wav` },
];

const state = {
  sample: sampleIds[0],
  rate: "low",
};

function audioMarkup(src, label) {
  return `<audio controls preload="metadata" src="${src}" aria-label="${label}"></audio>`;
}

function renderSampleTabs() {
  const tabs = document.querySelector("#sample-tabs");
  tabs.innerHTML = sampleIds
    .map(
      (id) => `
        <button
          type="button"
          class="sample-tab${id === state.sample ? " is-active" : ""}"
          data-sample="${id}"
          role="tab"
          aria-selected="${id === state.sample ? "true" : "false"}"
        >
          Case ${String(id).padStart(2, "0")}
        </button>
      `,
    )
    .join("");

  tabs.querySelectorAll(".sample-tab").forEach((button) => {
    button.addEventListener("click", () => {
      state.sample = Number(button.dataset.sample);
      renderAll();
    });
  });
}

function renderRateToggle() {
  document.querySelectorAll(".rate-button").forEach((button) => {
    const active = button.dataset.rate === state.rate;
    button.classList.toggle("is-active", active);
    button.setAttribute("aria-pressed", active ? "true" : "false");
  });
}

function setupRateToggle() {
  document.querySelectorAll(".rate-button").forEach((button) => {
    button.addEventListener("click", () => {
      state.rate = button.dataset.rate;
      renderAll();
    });
  });
}

function renderComparison() {
  const grid = document.querySelector("#comparison-grid");
  const visibleCodecs = codecs.filter(
    (codec) => codec.key === "original" || codec.rates.includes(state.rate),
  );

  grid.innerHTML = visibleCodecs
    .map((codec) => {
      const rateLabel =
        codec.key === "original"
          ? "Ground-truth sample"
          : state.rate === "low"
            ? "Low bitrate reconstruction"
            : "High bitrate reconstruction";
      const src =
        codec.key === "original" ? codec.path(state.sample) : codec.path(state.sample, state.rate);

      return `
        <article class="audio-card${codec.ours ? " is-ours" : ""}">
          <div class="audio-card-header">
            <div>
              <h3>${codec.label}</h3>
              <p>${rateLabel}</p>
            </div>
            <span class="codec-badge">${codec.badge}</span>
          </div>
          ${audioMarkup(src, `${codec.label} case ${state.sample}`)}
        </article>
      `;
    })
    .join("");
}

function renderPitchCases() {
  const grid = document.querySelector("#pitch-grid");
  grid.innerHTML = pitchCases
    .map(
      (sample) => `
        <article class="pitch-card">
          <h3>Case ${String(sample).padStart(2, "0")}</h3>
          <div class="pitch-stack">
            ${pitchVariants
              .map(
                (variant) => `
                  <div class="pitch-item">
                    <span>${variant.label}</span>
                    ${audioMarkup(variant.path(sample), `${variant.label} case ${sample}`)}
                  </div>
                `,
              )
              .join("")}
          </div>
        </article>
      `,
    )
    .join("");
}

function renderAll() {
  renderSampleTabs();
  renderRateToggle();
  renderComparison();
}

setupRateToggle();
renderAll();
renderPitchCases();
