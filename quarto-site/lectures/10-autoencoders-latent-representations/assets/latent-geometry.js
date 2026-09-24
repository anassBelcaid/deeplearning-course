(() => {
  const root = "assets/latent-geometry";
  const palette = ["#2878c8", "#ef5b45", "#008e9b", "#7655ad", "#d58b16", "#3c8d63", "#c44f86", "#586a78", "#9a6b35", "#00a3b4"];
  let cache;

  async function loadData() {
    if (!cache) {
      cache = Promise.all([
        fetch(`${root}/manifest.json`).then(response => response.json()),
        loadImage(`${root}/inputs-sprite.png`),
        loadImage(`${root}/reconstructions-sprite.png`),
      ]).then(([manifest, inputs, reconstructions]) => ({manifest, inputs, reconstructions}));
    }
    return cache;
  }

  function loadImage(src) {
    return new Promise((resolve, reject) => {
      const image = new Image();
      image.onload = () => resolve(image);
      image.onerror = reject;
      image.src = src;
    });
  }

  function pointPosition(point, canvas) {
    const pad = 28;
    return {
      x: pad + point.x * (canvas.width - 2 * pad),
      y: canvas.height - pad - point.y * (canvas.height - 2 * pad),
    };
  }

  function drawScatter(widget, selected = 200) {
    const {manifest} = widget._geometry;
    const canvas = widget.querySelector("canvas[data-role='scatter']");
    const context = canvas.getContext("2d");
    const coloured = widget.dataset.coloured === "true";
    context.clearRect(0, 0, canvas.width, canvas.height);
    context.fillStyle = "#fffdf7";
    context.fillRect(0, 0, canvas.width, canvas.height);
    context.strokeStyle = "#d9e0e1";
    context.lineWidth = 1;
    for (let i = 1; i < 5; i += 1) {
      const x = i * canvas.width / 5;
      const y = i * canvas.height / 5;
      context.beginPath(); context.moveTo(x, 0); context.lineTo(x, canvas.height); context.stroke();
      context.beginPath(); context.moveTo(0, y); context.lineTo(canvas.width, y); context.stroke();
    }
    manifest.points.forEach(point => {
      const position = pointPosition(point, canvas);
      context.beginPath();
      context.arc(position.x, position.y, coloured ? 4.4 : 3.7, 0, Math.PI * 2);
      context.fillStyle = coloured ? palette[point.label] : "rgba(35,51,61,.48)";
      context.fill();
    });
    const point = manifest.points[selected];
    const position = pointPosition(point, canvas);
    context.beginPath();
    context.arc(position.x, position.y, 10, 0, Math.PI * 2);
    context.strokeStyle = "#ef5b45";
    context.lineWidth = 4;
    context.stroke();
  }

  function drawSprite(source, id, canvas) {
    const context = canvas.getContext("2d");
    const columns = 40;
    const size = 28;
    const row = Math.floor(id / columns);
    const column = id % columns;
    context.imageSmoothingEnabled = false;
    context.clearRect(0, 0, canvas.width, canvas.height);
    context.fillStyle = "#111";
    context.fillRect(0, 0, canvas.width, canvas.height);
    context.drawImage(source, column * size, row * size, size, size, 0, 0, canvas.width, canvas.height);
  }

  function updateSelection(widget, id) {
    const {manifest, inputs, reconstructions} = widget._geometry;
    const point = manifest.points[id];
    widget.dataset.selected = String(id);
    drawScatter(widget, id);
    widget.querySelectorAll("canvas[data-role='input']").forEach(canvas => drawSprite(inputs, id, canvas));
    widget.querySelectorAll("canvas[data-role='reconstruction']").forEach(canvas => drawSprite(reconstructions, id, canvas));
    widget.querySelectorAll("[data-role='class']").forEach(node => { node.textContent = point.class; });
    widget.querySelectorAll("[data-role='coords']").forEach(node => { node.textContent = `(${point.z[0].toFixed(2)}, ${point.z[1].toFixed(2)})`; });
    widget.querySelectorAll("[data-role='mse']").forEach(node => { node.textContent = point.mse.toFixed(4); });
    updateNeighbours(widget, id);
  }

  function nearestPoints(points, anchorId, count) {
    const anchor = points[anchorId];
    return points
      .filter(point => point.id !== anchorId)
      .map(point => ({point, distance: Math.hypot(point.z[0] - anchor.z[0], point.z[1] - anchor.z[1])}))
      .sort((a, b) => a.distance - b.distance)
      .slice(0, count);
  }

  function updateNeighbours(widget, anchorId) {
    const strip = widget.querySelector("[data-role='neighbours']");
    if (!strip) return;
    const {manifest, inputs, reconstructions} = widget._geometry;
    const neighbours = nearestPoints(manifest.points, anchorId, 7);
    strip.querySelectorAll(".neighbour-item").forEach((item, index) => {
      const neighbour = neighbours[index];
      drawSprite(inputs, neighbour.point.id, item.querySelector("canvas[data-kind='input']"));
      drawSprite(reconstructions, neighbour.point.id, item.querySelector("canvas[data-kind='reconstruction']"));
      item.querySelector("small").textContent = `Δz ${neighbour.distance.toFixed(2)}`;
      item.querySelector("b").textContent = neighbour.point.class;
    });
  }

  async function initialiseWidget(widget) {
    widget._geometry = await loadData();
    const canvas = widget.querySelector("canvas[data-role='scatter']");
    canvas.addEventListener("click", event => {
      event.stopPropagation();
      const rectangle = canvas.getBoundingClientRect();
      const click = {
        x: (event.clientX - rectangle.left) * canvas.width / rectangle.width,
        y: (event.clientY - rectangle.top) * canvas.height / rectangle.height,
      };
      let closest = 0;
      let distance = Infinity;
      widget._geometry.manifest.points.forEach(point => {
        const position = pointPosition(point, canvas);
        const candidate = Math.hypot(position.x - click.x, position.y - click.y);
        if (candidate < distance) { distance = candidate; closest = point.id; }
      });
      updateSelection(widget, closest);
    });
    updateSelection(widget, Number(widget.dataset.selected || 200));
  }

  function initialise() {
    document.querySelectorAll(".latent-map-widget").forEach(initialiseWidget);
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", initialise);
  else initialise();
})();
