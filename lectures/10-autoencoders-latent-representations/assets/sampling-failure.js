(async () => {
  const canvases = [...document.querySelectorAll("canvas[data-role='sampling-map']")];
  if (!canvases.length) return;
  const response = await fetch("assets/sampling-failure/manifest.json");
  const data = await response.json();
  const draw = (canvas) => {
    const ctx = canvas.getContext("2d");
    const { width, height } = canvas;
    const pad = 34;
    ctx.fillStyle = "#fbfaf6";
    ctx.fillRect(0, 0, width, height);
    ctx.strokeStyle = "#d0d9dc";
    ctx.lineWidth = 2;
    ctx.strokeRect(pad, pad, width - 2 * pad, height - 2 * pad);
    for (const point of data.encoded_points) {
      ctx.beginPath();
      ctx.arc(pad + point.x * (width - 2 * pad), height - pad - point.y * (height - 2 * pad), 2.4, 0, Math.PI * 2);
      ctx.fillStyle = "rgba(64,83,94,.22)";
      ctx.fill();
    }
    for (const point of data.random_points) {
      const danger = Math.min(point.distance / 4, 1);
      ctx.beginPath();
      ctx.arc(pad + point.x * (width - 2 * pad), height - pad - point.y * (height - 2 * pad), 4.2, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(${Math.round(0 + 239 * danger)},${Math.round(142 - 51 * danger)},${Math.round(155 - 86 * danger)},.85)`;
      ctx.fill();
    }
    ctx.font = "700 15px DM Mono, monospace";
    ctx.fillStyle = "#637681";
    ctx.fillText("z₂", pad + 8, pad + 22);
    ctx.fillText("z₁", width - pad - 28, height - pad - 10);
  };
  canvases.forEach(draw);
})();
