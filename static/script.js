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

// Cấu hình canvas
ctx.lineWidth = 20;
ctx.lineCap = 'round';
ctx.lineJoin = 'round';
ctx.fillStyle = 'white';
ctx.fillRect(0, 0, canvas.width, canvas.height);
ctx.strokeStyle = 'black';

// Xử lý sự kiện chuột
canvas.addEventListener('mousedown', startDrawing);
canvas.addEventListener('mousemove', draw);
canvas.addEventListener('mouseup', stopDrawing);
canvas.addEventListener('mouseout', stopDrawing);

// Xử lý sự kiện cảm ứng
canvas.addEventListener('touchstart', (e) => {
    e.preventDefault();
    startDrawing(e.touches[0]);
});
canvas.addEventListener('touchmove', (e) => {
    e.preventDefault();
    draw(e.touches[0]);
});
canvas.addEventListener('touchend', stopDrawing);

function getCanvasPoint(e) {
    const rect = canvas.getBoundingClientRect();
    return {
        x: (e.clientX - rect.left) * (canvas.width / rect.width),
        y: (e.clientY - rect.top) * (canvas.height / rect.height)
    };
}

function startDrawing(e) {
    isDrawing = true;
    const point = getCanvasPoint(e);
    ctx.beginPath();
    ctx.moveTo(point.x, point.y);
    resetDebounceTimer();
}

function draw(e) {
    if (!isDrawing) return;
    const point = getCanvasPoint(e);
    ctx.lineTo(point.x, point.y);
    ctx.stroke();
    resetDebounceTimer();
}

function stopDrawing() {
    isDrawing = false;
    ctx.beginPath();
    resetDebounceTimer();
}

// Chuyển đổi chế độ vẽ/tẩy
eraseBtn.addEventListener('click', () => {
    isErasing = !isErasing;
    ctx.strokeStyle = isErasing ? 'white' : 'black';
    eraseBtn.textContent = isErasing ? 'Draw' : 'Erase';
    eraseBtn.style.backgroundColor = isErasing ? '#e74c3c' : '#1abc9c';
    resetDebounceTimer();
});

// Xóa toàn bộ canvas
deleteBtn.addEventListener('click', () => {
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    document.getElementById('predictedDigit').textContent = 'Chưa có dự đoán';
    const probabilities = document.getElementById('probabilities');
    if (probabilities) {
        probabilities.innerHTML = '';
    }
    probChart.data.datasets[0].data = Array(10).fill(0);
    probChart.update();
    // probChart.erase();
    isErasing = false;
    ctx.strokeStyle = 'black';
    eraseBtn.textContent = 'Erase';
    eraseBtn.style.backgroundColor = '#1abc9c';
    uploadStatus.textContent = '';
    resetDebounceTimer();
});

// Khởi tạo biểu đồ xác suất
uploadBtn.addEventListener('click', () => {
    imageUpload.click();
});

imageUpload.addEventListener('change', async () => {
    const file = imageUpload.files && imageUpload.files[0];
    if (!file) return;

    if (!file.type.startsWith('image/')) {
        uploadStatus.textContent = 'Please choose an image file.';
        imageUpload.value = '';
        return;
    }

    uploadStatus.textContent = 'Uploading...';

    try {
        await uploadOriginalImage(file);
        await drawUploadedImage(file);
        uploadStatus.textContent = file.name;
        sendPrediction();
    } catch (error) {
        console.error('Upload error:', error);
        uploadStatus.textContent = 'Upload failed.';
        alert('Loi khi tai anh len');
    } finally {
        imageUpload.value = '';
    }
});

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
            ctx.fillStyle = 'white';
            ctx.fillRect(0, 0, canvas.width, canvas.height);

            const scale = Math.min(canvas.width / image.width, canvas.height / image.height);
            const drawWidth = image.width * scale;
            const drawHeight = image.height * scale;
            const drawX = (canvas.width - drawWidth) / 2;
            const drawY = (canvas.height - drawHeight) / 2;

            ctx.drawImage(image, drawX, drawY, drawWidth, drawHeight);
            isErasing = false;
            ctx.strokeStyle = 'black';
            eraseBtn.textContent = 'Erase';
            eraseBtn.style.backgroundColor = '#1abc9c';
            resolve();
        };

        image.onerror = () => {
            URL.revokeObjectURL(objectUrl);
            reject(new Error('Unable to load image'));
        };

        image.src = objectUrl;
    });
}

const probChart = window.Chart ? new Chart(document.getElementById('probChart'), {
    type: 'bar',
    data: {
        labels: ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
        datasets: [{
            label: 'Xác suất (%)',
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
        scales: {
            y: { beginAtZero: true, max: 100, title: { display: true, text: 'Xác suất (%)' } },
            x: { title: { display: true, text: 'Chữ số' } }
        },
        plugins: { legend: { display: false } }
    }
}) : createFallbackChart(document.getElementById('probChart'));

function createFallbackChart(canvasElement) {
    const bars = document.createElement('div');
    bars.className = 'prob-fallback';
    canvasElement.replaceWith(bars);

    const chartState = {
        data: {
            datasets: [{
                data: Array(10).fill(0)
            }]
        },
        update() {
            bars.innerHTML = chartState.data.datasets[0].data.map((value, digit) => {
                const percent = Math.max(0, Math.min(100, Number(value) || 0));
                return `
                    <div class="prob-row">
                        <span>${digit}</span>
                        <div class="prob-track">
                            <i style="width: ${percent}%"></i>
                        </div>
                        <strong>${percent.toFixed(1)}%</strong>
                    </div>
                `;
            }).join('');
        }
    };

    chartState.update();
    return chartState;
}

// // Hàm gửi dự đoán
// function sendPrediction() {
//     const imgData = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
//     const isEmpty = Array.from(imgData).every(pixel => pixel === 255);
//     if (isEmpty) {
//         // Không gửi nếu canvas trống
//         document.getElementById('predictedDigit').textContent = 'Chưa có dự đoán';
//         probChart.data.datasets[0].data = Array(10).fill(0);
//         probChart.update();
//         return;
//     }

//     const dataURL = canvas.toDataURL('image/png');
//     fetch('/predict', {
//         method: 'POST',
//         headers: { 'Content-Type': 'application/json' },
//         body: JSON.stringify({ image: dataURL })
//     })
//     .then(response => response.json())
//     .then(data => {
//         if (data.error) {
//             alert(data.error);
//             return;
//         }
//         document.getElementById('predictedDigit').textContent = `Chữ số dự đoán: ${data.digit}`;
//         const probsList = document.getElementById('probabilities');
//         // probsList.innerHTML = data.probabilities.map((p, i) => `<li>Chữ số ${i}: ${(p * 100).toFixed(2)}%</li>`).join('');
//         probChart.data.datasets[0].data = data.probabilities.map(p => p * 100);
//         probChart.update();
//     })
//     .catch(error => {
//         console.error('Error:', error);
//         alert('Lỗi khi gửi yêu cầu dự đoán');
//     });
// }

function sendPrediction() {
    // Lấy dữ liệu gốc
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const { data, width, height } = imageData;

    // Tìm bounding box
    let minX = width, minY = height, maxX = 0, maxY = 0;
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            const index = (y * width + x) * 4;
            const alpha = data[index + 3];
            const isNotWhite = data[index] !== 255 || data[index + 1] !== 255 || data[index + 2] !== 255;

            if (isNotWhite && alpha > 0) {
                if (x < minX) minX = x;
                if (x > maxX) maxX = x;
                if (y < minY) minY = y;
                if (y > maxY) maxY = y;
            }
        }
    }

    // Nếu không có nét vẽ -> không dự đoán
    if (minX > maxX || minY > maxY) {
        document.getElementById('predictedDigit').textContent = 'Chưa có dự đoán';
        probChart.data.datasets[0].data = Array(10).fill(0);
        probChart.update();
        return;
    }

    // Cắt ảnh theo bounding box
    const croppedWidth = maxX - minX + 1;
    const croppedHeight = maxY - minY + 1;
    const croppedImageData = ctx.getImageData(minX, minY, croppedWidth, croppedHeight);

    // Tạo canvas tạm 28x28 (hoặc canvas.width x canvas.height)
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = canvas.width;
    tempCanvas.height = canvas.height;
    const tempCtx = tempCanvas.getContext('2d');

    // Fill trắng
    tempCtx.fillStyle = 'white';
    tempCtx.fillRect(0, 0, tempCanvas.width, tempCanvas.height);

    // Tính vị trí để vẽ vào giữa
    const offsetX = Math.floor((canvas.width - croppedWidth) / 2);
    const offsetY = Math.floor((canvas.height - croppedHeight) / 2);

    // Vẽ phần đã cắt vào giữa
    tempCtx.putImageData(croppedImageData, offsetX, offsetY);

    // Lấy dataURL từ canvas đã căn giữa
    const dataURL = tempCanvas.toDataURL('image/png');

    // Gửi ảnh đi như cũ
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
        document.getElementById('predictedDigit').textContent = `Chữ số dự đoán: ${data.digit}`;
        probChart.data.datasets[0].data = data.probabilities.map(p => p * 100);
        probChart.update();
    })
    .catch(error => {
        console.error('Error:', error);
        alert('Lỗi khi gửi yêu cầu dự đoán');
    });
}


// Hàm reset timer cho dự đoán tự động
function resetDebounceTimer() {
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(sendPrediction, 500); // Chờ 1 giây
}

// Dự đoán khi nhấn nút Predict
if (predictBtn) {
    predictBtn.addEventListener('click', sendPrediction);
}
deleteBtn.addEventListener('click', () => {
    resetDebounceTimer();
});

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
