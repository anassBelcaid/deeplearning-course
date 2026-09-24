(() => {
  const dims = [2, 4, 8, 16, 32, 64, 128];
  const classes = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
  ];
  const testMse = {
    2: 0.0291551, 4: 0.0205394, 8: 0.0168457, 16: 0.0162394,
    32: 0.0163097, 64: 0.0159695, 128: 0.0158035
  };
  const sampleMse = {
    2: [0.0180263,0.0146013,0.0473109,0.0144527,0.0368099,0.0164535,0.0246208,0.0159228,0.0255397,0.0173357],
    4: [0.0122615,0.0090376,0.0367833,0.0084687,0.0358718,0.0098598,0.019511,0.013322,0.0189298,0.0088883],
    8: [0.0089956,0.0084804,0.032013,0.0072346,0.0294254,0.0101954,0.0141211,0.0116623,0.0118121,0.0084368],
    16:[0.0087265,0.0072697,0.0280126,0.0050771,0.0238681,0.0101602,0.0137141,0.0116728,0.0146716,0.0081553],
    32:[0.0091545,0.0082627,0.0311819,0.0052869,0.0290529,0.0097935,0.0143068,0.0121765,0.0144731,0.0089464],
    64:[0.0087501,0.0078936,0.0262615,0.0056423,0.0259885,0.0099278,0.0155973,0.0113775,0.0127279,0.0080804],
    128:[0.0081528,0.0065242,0.0281552,0.0059381,0.029357,0.0095218,0.0139349,0.0125159,0.0141013,0.0085454]
  };
  const root = "assets/latent-sweep";

  function updateExplorer(explorer) {
    const d = Number(explorer.dataset.dimension || 8);
    const sample = Number(explorer.dataset.sample || 2);
    explorer.querySelector("[data-role='input']").src = `${root}/sample-${sample}-input.png`;
    explorer.querySelector("[data-role='reconstruction']").src = `${root}/d-${d}-sample-${sample}-reconstruction.png`;
    explorer.querySelector("[data-role='error']").src = `${root}/d-${d}-sample-${sample}-error.png`;
    explorer.querySelector("[data-role='class']").textContent = classes[sample];
    explorer.querySelectorAll("[data-role='dimension']").forEach(node => { node.textContent = d; });
    explorer.querySelector("[data-role='ratio']").textContent = `${(100 * d / 784).toFixed(1)}%`;
    explorer.querySelector("[data-role='sample-mse']").textContent = sampleMse[d][sample].toFixed(4);
    explorer.querySelector("[data-role='test-mse']").textContent = testMse[d].toFixed(4);
    explorer.querySelectorAll("[data-d]").forEach(button => {
      button.classList.toggle("active", Number(button.dataset.d) === d);
    });
    explorer.querySelectorAll("[data-sample]").forEach(button => {
      button.classList.toggle("active", Number(button.dataset.sample) === sample);
    });
  }

  function initialiseExplorer(explorer) {
    explorer.dataset.dimension ||= "8";
    explorer.dataset.sample ||= "2";
    explorer.querySelectorAll("[data-d]").forEach(button => {
      button.addEventListener("click", event => {
        event.stopPropagation();
        explorer.dataset.dimension = button.dataset.d;
        updateExplorer(explorer);
      });
    });
    explorer.querySelectorAll("[data-sample]").forEach(button => {
      button.addEventListener("click", event => {
        event.stopPropagation();
        explorer.dataset.sample = button.dataset.sample;
        updateExplorer(explorer);
      });
    });
    updateExplorer(explorer);
  }

  function drawCurve(container) {
    const width = 1080, height = 430;
    const left = 95, right = 40, top = 35, bottom = 70;
    const minY = 0.014, maxY = 0.031;
    const x = index => left + index * (width - left - right) / (dims.length - 1);
    const y = value => top + (maxY - value) * (height - top - bottom) / (maxY - minY);
    const points = dims.map((d, index) => `${x(index)},${y(testMse[d])}`).join(" ");
    const horizontal = [0.015, 0.020, 0.025, 0.030].map(value => `
      <line x1="${left}" y1="${y(value)}" x2="${width-right}" y2="${y(value)}" class="curve-grid"/>
      <text x="${left-18}" y="${y(value)+6}" text-anchor="end" class="curve-tick">${value.toFixed(3)}</text>`).join("");
    const marks = dims.map((d, index) => `
      <circle cx="${x(index)}" cy="${y(testMse[d])}" r="9" class="curve-point ${d === 8 ? "elbow" : ""}"/>
      <text x="${x(index)}" y="${height-34}" text-anchor="middle" class="curve-d">${d}</text>
      <text x="${x(index)}" y="${y(testMse[d])-18}" text-anchor="middle" class="curve-value">${testMse[d].toFixed(4)}</text>`).join("");
    container.innerHTML = `<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="Test reconstruction MSE falls sharply from latent dimension 2 to 8, then improves only slightly and unevenly through dimension 128.">
      ${horizontal}
      <line x1="${left}" y1="${top}" x2="${left}" y2="${height-bottom}" class="curve-axis"/>
      <line x1="${left}" y1="${height-bottom}" x2="${width-right}" y2="${height-bottom}" class="curve-axis"/>
      <polyline points="${points}" class="curve-line"/>
      ${marks}
      <text x="${(left+width-right)/2}" y="${height-4}" text-anchor="middle" class="curve-label">latent dimension d</text>
      <text transform="translate(25 ${(top+height-bottom)/2}) rotate(-90)" text-anchor="middle" class="curve-label">test MSE ↓</text>
      <path d="M ${x(2)+14} ${y(testMse[8])-35} C ${x(2)+90} ${y(testMse[8])-115}, ${x(2)+180} ${y(testMse[8])-115}, ${x(2)+230} ${y(testMse[8])-72}" class="curve-annotation-line"/>
      <text x="${x(2)+240}" y="${y(testMse[8])-78}" class="curve-annotation">most of the measured gain</text>
      <text x="${x(2)+240}" y="${y(testMse[8])-54}" class="curve-annotation">has arrived by d = 8</text>
    </svg>`;
  }

  function initialise() {
    document.querySelectorAll(".latent-explorer").forEach(initialiseExplorer);
    document.querySelectorAll(".latent-curve").forEach(drawCurve);
  }
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initialise);
  } else {
    initialise();
  }
})();
