const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const deleteBtn = document.getElementById('deleteBtn');
const eraseBtn = document.getElementById('eraseBtn');
const predictBtn = document.getElementById('predictBtn');
const uploadBtn = document.getElementById('uploadBtn');
const imageUpload = document.getElementById('imageUpload');
const uploadStatus = document.getElementById('uploadStatus');
const revealItems = document.querySelectorAll('.reveal');

let isDrawing = false;
let isErasing = false;
let debounceTimer;
let visualTimers = [];
let lastFeatureUpdate = 0;
let explainerZoom = 1;
let fitMode = true;

const EXPLAINER_BASE_WIDTH = 1680;
const EXPLAINER_BASE_HEIGHT = 640;

const stageOrder = ['input', 'conv1', 'relu1', 'pool1', 'conv2', 'relu2', 'pool2', 'dense', 'output'];
const mapKinds = ['vertical', 'horizontal', 'diagonal', 'anti', 'edge', 'blur', 'vertical', 'horizontal', 'diagonal', 'anti'];

const layerDefinitions = [
    { stage: 'input', label: 'input', shape: '(28, 28, 1)', type: 'input' },
    { stage: 'conv1', label: 'conv_1', shape: '(26, 26, 10)', maps: 10, signed: true },
    { stage: 'relu1', label: 'relu_1', shape: '(26, 26, 10)', maps: 10, relu: true },
    { stage: 'pool1', label: 'max_pool_1', shape: '(13, 13, 10)', maps: 10, relu: true, pooled: true, compact: true },
    { stage: 'conv2', label: 'conv_2', shape: '(11, 11, 16)', maps: 12, signed: true, deep: true },
    { stage: 'relu2', label: 'relu_2', shape: '(11, 11, 16)', maps: 12, relu: true, deep: true },
    { stage: 'pool2', label: 'max_pool_2', shape: '(5, 5, 16)', maps: 12, relu: true, pooled: true, deep: true, compact: true },
    { stage: 'dense', label: 'flatten + dense', shape: '(400 -> 128)', type: 'dense' },
    { stage: 'output', label: 'output', shape: '(10)', type: 'output' }
];

const cnnStepContent = {
    input: {
        title: 'Input image',
        text: 'Your drawn image is cropped to the stroke bounding box, centered, and normalized into a 28x28x1 tensor.',
        input: 'canvas or uploaded image',
        filter: 'crop + center + resize',
        weighted: 'pixel intensity',
        output: '28x28 grayscale'
    },
    conv1: {
        title: 'Convolution 1',
        text: '3x3 kernels sweep over the image to detect edges, corners, curves, and intersections very locally.',
        input: 'patch 3x3',
        filter: '10 learned kernels',
        weighted: 'sum(pixel * weight) + bias',
        output: '10 feature map 26x26'
    },
    relu1: {
        title: 'ReLU 1',
        text: 'ReLU keeps positive responses and blocks negative signals, highlighting the activated regions in the feature map.',
        input: 'conv_1 activations',
        filter: 'max(0, x)',
        weighted: 'filter negative signals',
        output: 'non-negative activation map'
    },
    pool1: {
        title: 'Max Pooling 1',
        text: 'Pooling takes the strongest signal in each small window, reducing map size and making the model robust to slight translations.',
        input: 'map 26x26',
        filter: '2x2 window',
        weighted: 'max activation',
        output: 'map 13x13'
    },
    conv2: {
        title: 'Convolution 2',
        text: 'Deeper convolutional layers combine small strokes into larger patterns like the loop of a 9, stem of a 1, or corner of a 4.',
        input: '10 pooled maps',
        filter: '16 deep kernels',
        weighted: 'combine multiple channels',
        output: '16 feature map 11x11'
    },
    relu2: {
        title: 'ReLU 2',
        text: 'Strong features are preserved, while irrelevant areas fade before final compression.',
        input: 'conv_2 activations',
        filter: 'max(0, x)',
        weighted: 'keep strong signals',
        output: 'deep activation map'
    },
    pool2: {
        title: 'Max Pooling 2',
        text: 'A second pooling compresses features into smaller matrices to prepare the vector for the Dense layer.',
        input: 'map 11x11',
        filter: '2x2 window',
        weighted: 'max activation',
        output: 'map 5x5'
    },
    dense: {
        title: 'Flatten + Dense',
        text: 'Maps are flattened into vectors. The Dense layer learns to combine curves, spaces, and intersections into scores for each digit.',
        input: '16 map 5x5',
        filter: 'weight matrix',
        weighted: 'vector dot product',
        output: '10 logits'
    },
    output: {
        title: 'Softmax output',
        text: 'Softmax converts logits to probabilities from 0-9. The longest bar is the digit the model is most confident in.',
        input: '10 logits',
        filter: 'softmax',
        weighted: 'exp(logit) / sum',
        output: '10 probabilities'
    }
};

ctx.lineWidth = 20;
ctx.lineCap = 'round';
ctx.lineJoin = 'round';
clearCanvas();
ctx.strokeStyle = 'black';

buildExplainerBoard();
buildConnectionWeb();

const cnnVisualizer = document.querySelector('.mnist-explainer');
const cnnStepButtons = document.querySelectorAll('[data-cnn-step]');
const cnnLayerNodes = document.querySelectorAll('[data-explainer-stage]');
const cnnStepTitle = document.getElementById('cnnStepTitle');
const cnnStepText = document.getElementById('cnnStepText');
const cnnCalcInput = document.getElementById('cnnCalcInput');
const cnnCalcFilter = document.getElementById('cnnCalcFilter');
const cnnCalcWeighted = document.getElementById('cnnCalcWeighted');
const cnnCalcOutput = document.getElementById('cnnCalcOutput');
const cnnInputPreview = document.getElementById('cnnInputPreview');
const featureTiles = document.querySelectorAll('.feature-tile');
const outputRows = document.querySelectorAll('.mnist-output-row');
const detailToggle = document.getElementById('detailToggle');
const detailPanel = document.getElementById('explainerDetailPanel');
const explainerScroll = document.getElementById('explainerScroll');
const explainerCanvas = document.getElementById('explainerCanvas');
const explainerStage = document.getElementById('explainerStage');
const zoomOutBtn = document.getElementById('zoomOutBtn');
const zoomFitBtn = document.getElementById('zoomFitBtn');
const zoomInBtn = document.getElementById('zoomInBtn');
const zoomValue = document.getElementById('zoomValue');

canvas.addEventListener('mousedown', startDrawing);
canvas.addEventListener('mousemove', draw);
canvas.addEventListener('mouseup', stopDrawing);
canvas.addEventListener('mouseout', stopDrawing);

canvas.addEventListener('touchstart', (event) => {
    event.preventDefault();
    startDrawing(event.touches[0]);
});

canvas.addEventListener('touchmove', (event) => {
    event.preventDefault();
    draw(event.touches[0]);
});

canvas.addEventListener('touchend', stopDrawing);

eraseBtn.addEventListener('click', () => {
    isErasing = !isErasing;
    ctx.strokeStyle = isErasing ? 'white' : 'black';
    eraseBtn.textContent = isErasing ? 'Draw' : 'Erase';
    eraseBtn.style.backgroundColor = isErasing ? '#e85d4f' : '';
    resetDebounceTimer();
});

deleteBtn.addEventListener('click', () => {
    clearCanvas();
    clearPrediction();
    resetDrawMode();
    resetLiveCnnPreview();
    uploadStatus.textContent = '';
});

uploadBtn.addEventListener('click', () => {
    imageUpload.click();
});

imageUpload.addEventListener('change', async () => {
    const file = imageUpload.files && imageUpload.files[0];
    if (!file) return;

    if (!file.type.startsWith('image/')) {
        uploadStatus.textContent = 'Please select an image file.';
        imageUpload.value = '';
        return;
    }

    uploadStatus.textContent = 'Loading image...';

    try {
        await uploadOriginalImage(file);
        await drawUploadedImage(file);
        uploadStatus.textContent = file.name;
        updateLiveCnnPreview('input');
        sendPrediction();
    } catch (error) {
        console.error('Upload error:', error);
        uploadStatus.textContent = 'Upload failed.';
        alert('Error uploading image');
    } finally {
        imageUpload.value = '';
    }
});

if (predictBtn) {
    predictBtn.addEventListener('click', sendPrediction);
}

if (detailToggle && detailPanel) {
    detailToggle.addEventListener('click', () => {
        const hidden = detailPanel.classList.toggle('is-hidden');
        detailToggle.textContent = hidden ? 'Show details' : 'Hide details';
    });
}

zoomOutBtn?.addEventListener('click', () => {
    fitMode = false;
    setExplainerZoom(explainerZoom - 0.1);
});

zoomInBtn?.addEventListener('click', () => {
    fitMode = false;
    setExplainerZoom(explainerZoom + 0.1);
});

zoomFitBtn?.addEventListener('click', () => {
    fitMode = true;
    fitExplainerToViewport();
});

window.addEventListener('resize', () => {
    if (fitMode) fitExplainerToViewport();
});

cnnStepButtons.forEach((button) => {
    button.addEventListener('click', () => setCnnStep(button.dataset.cnnStep));
});

const probChart = window.Chart ? new Chart(document.getElementById('probChart'), {
    type: 'bar',
    data: {
        labels: ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
        datasets: [{
            label: 'Probability (%)',
            data: Array(10).fill(0),
            backgroundColor: [
                'rgba(15, 118, 110, 0.72)',
                'rgba(232, 93, 79, 0.72)',
                'rgba(240, 180, 41, 0.78)',
                'rgba(109, 91, 208, 0.72)',
                'rgba(79, 159, 95, 0.72)',
                'rgba(23, 23, 23, 0.72)',
                'rgba(15, 118, 110, 0.52)',
                'rgba(232, 93, 79, 0.52)',
                'rgba(240, 180, 41, 0.58)',
                'rgba(109, 91, 208, 0.52)'
            ],
            borderColor: 'rgba(23, 23, 23, 0.12)',
            borderWidth: 1
        }]
    },
    options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
            y: {
                beginAtZero: true,
                max: 100,
                title: { display: true, text: 'Probability (%)' },
                ticks: { callback: value => `${value}%` }
            },
            x: { title: { display: true, text: 'Digit' } }
        },
        plugins: { legend: { display: false } }
    }
}) : createFallbackChart(document.getElementById('probChart'));

function setExplainerZoom(value) {
    if (!explainerCanvas || !explainerStage) return;

    const nextZoom = Math.max(0.45, Math.min(1.45, Number(value) || 1));
    explainerZoom = nextZoom;
    explainerCanvas.style.width = `${EXPLAINER_BASE_WIDTH * nextZoom}px`;
    explainerCanvas.style.minWidth = `${EXPLAINER_BASE_WIDTH * nextZoom}px`;
    explainerCanvas.style.height = `${EXPLAINER_BASE_HEIGHT * nextZoom}px`;
    explainerStage.style.transform = `scale(${nextZoom})`;
    explainerStage.style.transformOrigin = 'top left';

    if (zoomValue) {
        zoomValue.textContent = `${Math.round(nextZoom * 100)}%`;
    }
}

function fitExplainerToViewport() {
    if (!explainerScroll) {
        setExplainerZoom(1);
        return;
    }

    const availableWidth = Math.max(320, explainerScroll.clientWidth - 2);
    const fitZoom = Math.min(1, availableWidth / EXPLAINER_BASE_WIDTH);
    setExplainerZoom(fitZoom);
    explainerScroll.scrollLeft = 0;
}

function buildExplainerBoard() {
    const board = document.getElementById('explainerBoard');
    if (!board) return;

    board.innerHTML = layerDefinitions.map((layer) => {
        if (layer.type === 'input') {
            return `
                <section class="explainer-column input-column" data-explainer-stage="${layer.stage}">
                    <header><span>${layer.label}</span><small>${layer.shape}</small></header>
                    <div class="input-preview digit-nine" id="cnnInputPreview" aria-label="28x28 input image"></div>
                    <p class="channel-label">grayscale channel</p>
                </section>
            `;
        }

        if (layer.type === 'dense') {
            return `
                <section class="explainer-column dense-column" data-explainer-stage="${layer.stage}">
                    <header><span>${layer.label}</span><small>${layer.shape}</small></header>
                    <div class="dense-web" aria-hidden="true">
                        <span></span><span></span><span></span><span></span><span></span>
                        <b></b><b></b><b></b><b></b>
                    </div>
                    <p class="dense-note">Feature vector capturing curves, intersections, and spaces of the digit.</p>
                </section>
            `;
        }

        if (layer.type === 'output') {
            return `
                <section class="explainer-column output-column" data-explainer-stage="${layer.stage}">
                    <header><span>${layer.label}</span><small>${layer.shape}</small></header>
                    <div class="mnist-output-list" id="mnistOutputList" aria-label="Probabilities of each digit">
                        ${Array.from({ length: 10 }, (_, digit) => `
                            <div class="mnist-output-row" data-digit="${digit}" style="--p: 10%;">
                                <span>${digit}</span><i></i><b>10.0%</b>
                            </div>
                        `).join('')}
                    </div>
                </section>
            `;
        }

        const tiles = Array.from({ length: layer.maps }, (_, index) => {
            const kind = mapKinds[index % mapKinds.length];
            return `<i class="feature-tile" data-map-source="${layer.stage}" data-map-kind="${kind}"></i>`;
        }).join('');

        return `
            <section class="explainer-column" data-explainer-stage="${layer.stage}">
                <header><span>${layer.label}</span><small>${layer.shape}</small></header>
                <div class="feature-grid tall ${layer.compact ? 'compact' : ''}">${tiles}</div>
                <div class="heat-scale ${layer.signed ? '' : 'positive'}"><span>${layer.signed ? '-1' : '0'}</span><i></i><span>${layer.signed ? '1' : '1'}</span></div>
            </section>
        `;
    }).join('');
}

function buildConnectionWeb() {
    const svg = document.querySelector('.connection-web');
    if (!svg) return;

    const ns = 'http://www.w3.org/2000/svg';
    const xs = [120, 300, 480, 660, 840, 1020, 1200, 1380, 1540];
    const ys = [90, 145, 200, 255, 310, 365, 420, 475];

    svg.innerHTML = '';
    for (let i = 0; i < xs.length - 1; i++) {
        const group = document.createElementNS(ns, 'g');
        const fromYs = i === 0 ? [255] : ys;
        const toYs = i === xs.length - 2 ? [95, 140, 185, 230, 275, 320, 365, 410, 455, 500] : ys;

        fromYs.forEach((fromY, fromIndex) => {
            toYs.forEach((toY, toIndex) => {
                if (fromYs.length > 1 && Math.abs(fromIndex - toIndex) > 2 && toIndex % 3 !== 0) return;
                const path = document.createElementNS(ns, 'path');
                const x1 = xs[i];
                const x2 = xs[i + 1];
                const c1 = x1 + 70;
                const c2 = x2 - 70;
                path.setAttribute('d', `M${x1} ${fromY} C${c1} ${fromY}, ${c2} ${toY}, ${x2} ${toY}`);
                group.appendChild(path);
            });
        });
        svg.appendChild(group);
    }
}

function getCanvasPoint(event) {
    const rect = canvas.getBoundingClientRect();
    return {
        x: (event.clientX - rect.left) * (canvas.width / rect.width),
        y: (event.clientY - rect.top) * (canvas.height / rect.height)
    };
}

function startDrawing(event) {
    isDrawing = true;
    const point = getCanvasPoint(event);
    ctx.beginPath();
    ctx.moveTo(point.x, point.y);
    resetDebounceTimer();
}

function draw(event) {
    if (!isDrawing) return;

    const point = getCanvasPoint(event);
    ctx.lineTo(point.x, point.y);
    ctx.stroke();
    updateLiveCnnPreview('input', null, true);
    resetDebounceTimer();
}

function stopDrawing() {
    if (!isDrawing) return;
    isDrawing = false;
    ctx.beginPath();
    updateLiveCnnPreview('conv1');
    resetDebounceTimer();
}

async function uploadOriginalImage(file) {
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch('/upload', {
        method: 'POST',
        body: formData
    });

    const data = await response.json();
    if (!response.ok || data.error) {
        throw new Error(data.error || 'Upload failed');
    }
}

function drawUploadedImage(file) {
    return new Promise((resolve, reject) => {
        const image = new Image();
        const objectUrl = URL.createObjectURL(file);

        image.onload = () => {
            URL.revokeObjectURL(objectUrl);
            clearCanvas();

            const scale = Math.min(canvas.width / image.width, canvas.height / image.height);
            const drawWidth = image.width * scale;
            const drawHeight = image.height * scale;
            const drawX = (canvas.width - drawWidth) / 2;
            const drawY = (canvas.height - drawHeight) / 2;

            ctx.drawImage(image, drawX, drawY, drawWidth, drawHeight);
            resetDrawMode();
            resolve();
        };

        image.onerror = () => {
            URL.revokeObjectURL(objectUrl);
            reject(new Error('Unable to load image'));
        };

        image.src = objectUrl;
    });
}

function clearCanvas() {
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
}

function clearPrediction() {
    document.getElementById('predictedDigit').textContent = 'No prediction yet';
    probChart.data.datasets[0].data = Array(10).fill(0);
    probChart.update();
    updateCnnOutputBars(Array(10).fill(0.1));
}

function resetDrawMode() {
    isErasing = false;
    ctx.strokeStyle = 'black';
    eraseBtn.textContent = 'Erase';
    eraseBtn.style.backgroundColor = '';
}

function createFallbackChart(canvasElement) {
    const bars = document.createElement('div');
    bars.className = 'prob-fallback';
    canvasElement.replaceWith(bars);

    const chartState = {
        data: { datasets: [{ data: Array(10).fill(0) }] },
        update() {
            bars.innerHTML = chartState.data.datasets[0].data.map((value, digit) => {
                const percent = Math.max(0, Math.min(100, Number(value) || 0));
                return `
                    <div class="prob-row">
                        <span>${digit}</span>
                        <div class="prob-track"><i style="width: ${percent}%"></i></div>
                        <strong>${percent.toFixed(1)}%</strong>
                    </div>
                `;
            }).join('');
        }
    };

    chartState.update();
    return chartState;
}

function buildCenteredDigitCanvas() {
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const { data, width, height } = imageData;

    let minX = width;
    let minY = height;
    let maxX = 0;
    let maxY = 0;

    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            const index = (y * width + x) * 4;
            const alpha = data[index + 3];
            const isNotWhite = data[index] < 245 || data[index + 1] < 245 || data[index + 2] < 245;

            if (isNotWhite && alpha > 0) {
                if (x < minX) minX = x;
                if (x > maxX) maxX = x;
                if (y < minY) minY = y;
                if (y > maxY) maxY = y;
            }
        }
    }

    if (minX > maxX || minY > maxY) {
        return { isEmpty: true, canvas: null };
    }

    const croppedWidth = maxX - minX + 1;
    const croppedHeight = maxY - minY + 1;
    const croppedImageData = ctx.getImageData(minX, minY, croppedWidth, croppedHeight);

    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = canvas.width;
    tempCanvas.height = canvas.height;
    const tempCtx = tempCanvas.getContext('2d');

    tempCtx.fillStyle = 'white';
    tempCtx.fillRect(0, 0, tempCanvas.width, tempCanvas.height);

    const offsetX = Math.floor((canvas.width - croppedWidth) / 2);
    const offsetY = Math.floor((canvas.height - croppedHeight) / 2);
    tempCtx.putImageData(croppedImageData, offsetX, offsetY);

    return { isEmpty: false, canvas: tempCanvas };
}

function sendPrediction() {
    const centered = buildCenteredDigitCanvas();

    if (centered.isEmpty) {
        clearPrediction();
        resetLiveCnnPreview();
        return;
    }

    updateLiveCnnPreview('conv1', centered.canvas);
    const dataURL = centered.canvas.toDataURL('image/png');

    fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image: dataURL })
    })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                alert(data.error);
                return;
            }

            document.getElementById('predictedDigit').textContent = `Predicted digit: ${data.digit}`;
            const percentages = data.probabilities.map(p => p * 100);
            probChart.data.datasets[0].data = percentages;
            probChart.update();
            runEndToEndVisualizer(centered.canvas, data.probabilities);
        })
        .catch(error => {
            console.error('Error:', error);
            alert('Error sending prediction request');
        });
}

function resetDebounceTimer() {
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(sendPrediction, 500);
}

function setCnnStep(step) {
    if (!cnnVisualizer || !cnnStepContent[step]) return;

    const content = cnnStepContent[step];
    cnnVisualizer.dataset.activeStage = step;

    cnnStepButtons.forEach((button) => {
        button.classList.toggle('is-active', button.dataset.cnnStep === step);
    });

    cnnLayerNodes.forEach((node) => {
        node.classList.toggle('is-focused', node.dataset.explainerStage === step);
        node.classList.toggle('is-before-focus', stageOrder.indexOf(node.dataset.explainerStage) < stageOrder.indexOf(step));
    });

    cnnStepTitle.textContent = content.title;
    cnnStepText.textContent = content.text;
    cnnCalcInput.textContent = content.input;
    cnnCalcFilter.textContent = content.filter;
    cnnCalcWeighted.textContent = content.weighted;
    cnnCalcOutput.textContent = content.output;
}

function updateLiveCnnPreview(step = 'input', sourceCanvas = null, lightOnly = false) {
    if (!cnnVisualizer) return;

    const centered = sourceCanvas ? { isEmpty: false, canvas: sourceCanvas } : buildCenteredDigitCanvas();
    if (centered.isEmpty) return;

    updateCnnInputPreview(centered.canvas);

    const now = performance.now();
    if (!lightOnly || now - lastFeatureUpdate > 160) {
        updateCnnFeatureMaps(centered.canvas);
        lastFeatureUpdate = now;
    }

    setCnnStep(step);
}

function runEndToEndVisualizer(sourceCanvas, probabilities) {
    updateCnnInputPreview(sourceCanvas);
    updateCnnFeatureMaps(sourceCanvas);
    updateCnnOutputBars(probabilities);

    visualTimers.forEach(clearTimeout);
    visualTimers = stageOrder.map((stage, index) => {
        return setTimeout(() => setCnnStep(stage), index * 180);
    });
}

function resetLiveCnnPreview() {
    visualTimers.forEach(clearTimeout);
    visualTimers = [];

    if (cnnInputPreview) {
        cnnInputPreview.classList.remove('has-live-input');
        cnnInputPreview.style.backgroundImage = '';
    }

    featureTiles.forEach((map) => {
        map.classList.remove('is-live-map');
        map.style.backgroundImage = '';
    });

    updateCnnOutputBars(Array(10).fill(0.1));
    setCnnStep('input');
    paintDefaultNine();
}

function updateCnnInputPreview(sourceCanvas) {
    if (!cnnInputPreview) return;
    const preview = document.createElement('canvas');
    preview.width = 64;
    preview.height = 64;
    const previewCtx = preview.getContext('2d');
    previewCtx.fillStyle = '#fff';
    previewCtx.fillRect(0, 0, 64, 64);
    previewCtx.drawImage(sourceCanvas, 0, 0, 64, 64);
    cnnInputPreview.classList.add('has-live-input');
    cnnInputPreview.style.backgroundImage = `url(${preview.toDataURL('image/png')})`;
}

function updateCnnFeatureMaps(sourceCanvas) {
    const cache = new Map();

    featureTiles.forEach((tile, index) => {
        const source = tile.dataset.mapSource || 'conv1';
        const kind = tile.dataset.mapKind || mapKinds[index % mapKinds.length];
        const key = `${source}:${kind}`;

        if (!cache.has(key)) {
            cache.set(key, createFeatureMapUrl(sourceCanvas, kind, {
                pooled: source.includes('pool'),
                relu: source.includes('relu') || source.includes('pool'),
                deep: source.includes('2'),
                signed: source.includes('conv')
            }));
        }

        tile.classList.add('is-live-map');
        tile.style.backgroundImage = `url(${cache.get(key)})`;
    });
}

function createFeatureMapUrl(sourceCanvas, type, options = {}) {
    const size = options.pooled ? 18 : 28;
    const base = document.createElement('canvas');
    base.width = 28;
    base.height = 28;
    const baseCtx = base.getContext('2d');
    baseCtx.drawImage(sourceCanvas, 0, 0, 28, 28);
    const input = baseCtx.getImageData(0, 0, 28, 28).data;
    const values = new Float32Array(28 * 28);

    for (let y = 0; y < 28; y++) {
        for (let x = 0; x < 28; x++) {
            const i = (y * 28 + x) * 4;
            values[y * 28 + x] = 1 - ((input[i] + input[i + 1] + input[i + 2]) / 3 / 255);
        }
    }

    const kernels = {
        vertical: [-1, 0, 1, -2, 0, 2, -1, 0, 1],
        horizontal: [-1, -2, -1, 0, 0, 0, 1, 2, 1],
        diagonal: [0, 1, 2, -1, 0, 1, -2, -1, 0],
        anti: [2, 1, 0, 1, 0, -1, 0, -1, -2],
        edge: [-1, -1, -1, -1, 8, -1, -1, -1, -1],
        blur: [1, 1, 1, 1, 1, 1, 1, 1, 1]
    };

    const kernel = kernels[type] || kernels.vertical;
    const conv = new Float32Array(28 * 28);
    let minValue = Infinity;
    let maxValue = -Infinity;

    for (let y = 1; y < 27; y++) {
        for (let x = 1; x < 27; x++) {
            let sum = 0;
            let k = 0;
            for (let ky = -1; ky <= 1; ky++) {
                for (let kx = -1; kx <= 1; kx++) {
                    sum += values[(y + ky) * 28 + (x + kx)] * kernel[k++];
                }
            }

            if (type === 'blur') sum /= 9;
            if (options.deep) sum = Math.sin(sum * 1.8 + (x - y) * 0.04) * Math.abs(sum);
            if (options.relu) sum = Math.max(0, sum);

            conv[y * 28 + x] = sum;
            minValue = Math.min(minValue, sum);
            maxValue = Math.max(maxValue, sum);
        }
    }

    if (!Number.isFinite(minValue) || !Number.isFinite(maxValue) || Math.abs(maxValue - minValue) < 0.001) {
        minValue = 0;
        maxValue = 1;
    }

    const output = document.createElement('canvas');
    output.width = size;
    output.height = size;
    const outputCtx = output.getContext('2d');
    const image = outputCtx.createImageData(size, size);

    for (let y = 0; y < size; y++) {
        for (let x = 0; x < size; x++) {
            let value;
            if (options.pooled) {
                const sx = Math.min(26, Math.floor(x * 28 / size));
                const sy = Math.min(26, Math.floor(y * 28 / size));
                value = Math.max(
                    conv[sy * 28 + sx],
                    conv[sy * 28 + sx + 1],
                    conv[(sy + 1) * 28 + sx],
                    conv[(sy + 1) * 28 + sx + 1]
                );
            } else {
                value = conv[Math.floor(y * 28 / size) * 28 + Math.floor(x * 28 / size)];
            }

            const [r, g, b] = heatColor(value, minValue, maxValue, options.signed && !options.relu);
            const idx = (y * size + x) * 4;
            image.data[idx] = r;
            image.data[idx + 1] = g;
            image.data[idx + 2] = b;
            image.data[idx + 3] = 255;
        }
    }

    outputCtx.putImageData(image, 0, 0);
    return output.toDataURL('image/png');
}

function heatColor(value, minValue, maxValue, signed) {
    if (signed) {
        const limit = Math.max(Math.abs(minValue), Math.abs(maxValue), 0.001);
        const t = Math.max(-1, Math.min(1, value / limit));
        if (t >= 0) {
            const mix = t;
            return [Math.round(242 - 205 * mix), Math.round(247 - 125 * mix), Math.round(250 - 62 * mix)];
        }
        const mix = Math.abs(t);
        return [Math.round(255 - 42 * mix), Math.round(245 - 144 * mix), Math.round(238 - 160 * mix)];
    }

    const t = Math.max(0, Math.min(1, (value - minValue) / (maxValue - minValue || 1)));
    return [Math.round(247 - 219 * t), Math.round(250 - 125 * t), Math.round(252 - 72 * t)];
}

function updateCnnOutputBars(probabilities) {
    const values = probabilities.length ? probabilities : Array(10).fill(0.1);
    const maxIndex = values.reduce((best, value, index) => value > values[best] ? index : best, 0);

    outputRows.forEach((row, index) => {
        const probability = values[index] ?? 0;
        const percent = Math.max(2, Math.min(100, probability * 100));
        row.style.setProperty('--p', `${percent}%`);
        const label = row.querySelector('b');
        if (label) label.textContent = `${(probability * 100).toFixed(1)}%`;
        row.classList.toggle('is-top', index === maxIndex);
    });
}

function paintDefaultNine() {
    const demo = document.createElement('canvas');
    demo.width = 280;
    demo.height = 280;
    const demoCtx = demo.getContext('2d');
    demoCtx.fillStyle = 'white';
    demoCtx.fillRect(0, 0, demo.width, demo.height);
    demoCtx.fillStyle = 'black';
    demoCtx.font = '210px "Segoe Print", "Comic Sans MS", cursive';
    demoCtx.textAlign = 'center';
    demoCtx.textBaseline = 'middle';
    demoCtx.translate(140, 144);
    demoCtx.rotate(-0.08);
    demoCtx.fillText('9', 0, 10);
    demoCtx.setTransform(1, 0, 0, 1, 0, 0);
    updateCnnInputPreview(demo);
    updateCnnFeatureMaps(demo);
}

// Reveal animation
if ('IntersectionObserver' in window) {
    const revealObserver = new IntersectionObserver((entries) => {
        entries.forEach((entry) => {
            if (entry.isIntersecting) {
                entry.target.classList.add('is-visible');
                revealObserver.unobserve(entry.target);
            }
        });
    }, { threshold: 0.18 });

    revealItems.forEach((item) => revealObserver.observe(item));
} else {
    revealItems.forEach((item) => item.classList.add('is-visible'));
}

// Hugging Face model loading modal
const hfModal = document.getElementById('hfModal');
const updateModelBtn = document.getElementById('updateModelBtn');
const closeModalBtn = document.getElementById('closeModalBtn');
const confirmUpdateBtn = document.getElementById('confirmUpdateBtn');
const repoIdInput = document.getElementById('repoIdInput');
const updateStatusText = document.getElementById('updateStatus');

if (updateModelBtn && hfModal) {
    updateModelBtn.addEventListener('click', () => {
        hfModal.style.display = 'flex';
        updateStatusText.textContent = '';
        updateStatusText.style.color = 'var(--coral)';
    });

    closeModalBtn.addEventListener('click', () => {
        hfModal.style.display = 'none';
    });

    window.addEventListener('click', (event) => {
        if (event.target === hfModal) {
            hfModal.style.display = 'none';
        }
    });

    confirmUpdateBtn.addEventListener('click', () => {
        const repoId = repoIdInput.value.trim();
        if (!repoId) {
            updateStatusText.textContent = 'Please enter a Repo ID.';
            return;
        }

        updateStatusText.style.color = 'var(--teal)';
        updateStatusText.textContent = 'Loading model, please wait...';
        confirmUpdateBtn.disabled = true;

        fetch('/update_model', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ repo_id: repoId })
        })
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    updateStatusText.style.color = 'var(--coral)';
                    updateStatusText.textContent = data.error;
                } else {
                    updateStatusText.style.color = 'var(--teal)';
                    updateStatusText.textContent = data.message;
                    setTimeout(() => {
                        hfModal.style.display = 'none';
                        sendPrediction();
                    }, 2000);
                }
            })
            .catch(() => {
                updateStatusText.style.color = 'var(--coral)';
                updateStatusText.textContent = 'Error connecting to server.';
            })
            .finally(() => {
                confirmUpdateBtn.disabled = false;
            });
    });
}

resetLiveCnnPreview();
requestAnimationFrame(() => fitExplainerToViewport());
