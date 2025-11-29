// Global state
let state = {
    totalFrames: 0,
    currentFrame: 0,
    fps: 10,
    isPlaying: false,
    playInterval: null,
    annotationMode: null,
    regions: {
        positive: [],
        negative: []
    },
    selectedRegion: null,
    maskRadius: 5,
    videoWidth: 0,
    videoHeight: 0,
    filename: '',
    chart: null
};

// Canvas references
let videoCanvas, annotationCanvas, videoCtx, annotationCtx;

// Initialize when page loads
document.addEventListener('DOMContentLoaded', () => {
    videoCanvas = document.getElementById('videoCanvas');
    annotationCanvas = document.getElementById('annotationCanvas');
    videoCtx = videoCanvas.getContext('2d');
    annotationCtx = annotationCanvas.getContext('2d');
    
    // Setup canvas click handler
    annotationCanvas.addEventListener('click', handleCanvasClick);
});

// Upload TIFF file
async function uploadFile() {
    const fileInput = document.getElementById('fileInput');
    const file = fileInput.files[0];
    
    if (!file) return;
    
    const statusDiv = document.getElementById('uploadStatus');
    statusDiv.innerHTML = '<div class="loading"></div> Uploading and processing TIFF...';
    
    const formData = new FormData();
    formData.append('file', file);
    
    try {
        const response = await fetch('/upload_tiff', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (data.error) {
            statusDiv.innerHTML = `<span style="color: red;">Error: ${data.error}</span>`;
            return;
        }
        
        // Update state
        state.totalFrames = data.total_frames;
        state.videoWidth = data.width;
        state.videoHeight = data.height;
        state.filename = data.filename;
        state.currentFrame = 0;
        
        // Setup canvas
        videoCanvas.width = Math.min(data.width, 800);
        videoCanvas.height = (videoCanvas.width / data.width) * data.height;
        annotationCanvas.width = videoCanvas.width;
        annotationCanvas.height = videoCanvas.height;
        
        // Update UI
        statusDiv.innerHTML = `<span style="color: green;">✓ Loaded: ${data.filename} (${data.total_frames} frames)</span>`;
        document.getElementById('frameSlider').max = data.total_frames - 1;
        document.getElementById('statFrames').textContent = data.total_frames;
        
        // Show sections
        document.getElementById('playerSection').classList.remove('hidden');
        document.getElementById('annotationSection').classList.remove('hidden');
        document.getElementById('statsSection').classList.remove('hidden');
        
        // Load first frame
        await loadFrame(0);
        updateFrameInfo();
        
    } catch (error) {
        statusDiv.innerHTML = `<span style="color: red;">Error: ${error.message}</span>`;
    }
}

// Load and display a specific frame
async function loadFrame(frameNum) {
    try {
        const response = await fetch(`/get_frame/${frameNum}`);
        const data = await response.json();
        
        if (data.error) {
            console.error('Error loading frame:', data.error);
            return;
        }
        
        const img = new Image();
        img.onload = () => {
            videoCtx.clearRect(0, 0, videoCanvas.width, videoCanvas.height);
            videoCtx.drawImage(img, 0, 0, videoCanvas.width, videoCanvas.height);
            drawRegions();
        };
        img.src = data.image;
        
    } catch (error) {
        console.error('Error loading frame:', error);
    }
}

// Play/Pause functionality
function togglePlay() {
    state.isPlaying = !state.isPlaying;
    
    const playIcon = document.getElementById('playIcon');
    const playText = document.getElementById('playText');
    
    if (state.isPlaying) {
        playIcon.textContent = '⏸';
        playText.textContent = 'Pause';
        startPlayback();
    } else {
        playIcon.textContent = '▶';
        playText.textContent = 'Play';
        stopPlayback();
    }
}

function startPlayback() {
    state.playInterval = setInterval(async () => {
        state.currentFrame++;
        if (state.currentFrame >= state.totalFrames) {
            state.currentFrame = state.totalFrames - 1;
            togglePlay();
            return;
        }
        
        await loadFrame(state.currentFrame);
        document.getElementById('frameSlider').value = state.currentFrame;
        updateFrameInfo();
    }, 1000 / state.fps);
}

function stopPlayback() {
    if (state.playInterval) {
        clearInterval(state.playInterval);
        state.playInterval = null;
    }
}

function seekFrame(frameNum) {
    state.currentFrame = parseInt(frameNum);
    loadFrame(state.currentFrame);
    updateFrameInfo();
}

function updateFPS() {
    state.fps = parseInt(document.getElementById('fpsInput').value) || 10;
    if (state.isPlaying) {
        stopPlayback();
        startPlayback();
    }
}

function updateFrameInfo() {
    document.getElementById('frameInfo').textContent = 
        `Frame ${state.currentFrame + 1} / ${state.totalFrames}`;
}

// Annotation mode
function setMode(mode) {
    state.annotationMode = mode;
    
    const positiveBtn = document.getElementById('positiveBtn');
    const negativeBtn = document.getElementById('negativeBtn');
    const modeInfo = document.getElementById('modeInfo');
    
    positiveBtn.classList.remove('active');
    negativeBtn.classList.remove('active');
    
    if (mode === 'positive') {
        positiveBtn.classList.add('active');
        modeInfo.textContent = 'Click on the center of good smTIRF events (green circles).';
    } else if (mode === 'negative') {
        negativeBtn.classList.add('active');
        modeInfo.textContent = 'Click on areas that are NOT smTIRF events (red circles).';
    } else {
        modeInfo.textContent = 'Select a mode above to start clicking on events.';
    }
}

// Handle canvas click
function handleCanvasClick(event) {
    if (!state.annotationMode) return;
    
    const rect = annotationCanvas.getBoundingClientRect();
    const scaleX = state.videoWidth / videoCanvas.width;
    const scaleY = state.videoHeight / videoCanvas.height;
    
    const x = (event.clientX - rect.left) * scaleX;
    const y = (event.clientY - rect.top) * scaleY;
    
    // Create new region
    const region = {
        id: Date.now(),
        type: state.annotationMode,
        center: { x, y },
        radius: state.maskRadius,
        createdAtFrame: state.currentFrame
    };
    
    state.regions[state.annotationMode].push(region);
    
    // Update UI
    updateRegionCounts();
    drawRegions();
    updateRegionList();
    
    // Calculate and show intensity graph
    calculateIntensity(region);
}

// Draw all regions on canvas
function drawRegions() {
    annotationCtx.clearRect(0, 0, annotationCanvas.width, annotationCanvas.height);
    
    const scaleX = annotationCanvas.width / state.videoWidth;
    const scaleY = annotationCanvas.height / state.videoHeight;
    
    // Draw positive regions
    state.regions.positive.forEach(region => {
        annotationCtx.strokeStyle = '#4CAF50';
        annotationCtx.fillStyle = 'rgba(76, 175, 80, 0.2)';
        annotationCtx.lineWidth = 2;
        
        annotationCtx.beginPath();
        annotationCtx.arc(
            region.center.x * scaleX,
            region.center.y * scaleY,
            region.radius * scaleX,
            0,
            2 * Math.PI
        );
        annotationCtx.fill();
        annotationCtx.stroke();
        
        // Draw center dot
        annotationCtx.fillStyle = '#4CAF50';
        annotationCtx.beginPath();
        annotationCtx.arc(
            region.center.x * scaleX,
            region.center.y * scaleY,
            3,
            0,
            2 * Math.PI
        );
        annotationCtx.fill();
    });
    
    // Draw negative regions
    state.regions.negative.forEach(region => {
        annotationCtx.strokeStyle = '#f44336';
        annotationCtx.fillStyle = 'rgba(244, 67, 54, 0.2)';
        annotationCtx.lineWidth = 2;
        
        annotationCtx.beginPath();
        annotationCtx.arc(
            region.center.x * scaleX,
            region.center.y * scaleY,
            region.radius * scaleX,
            0,
            2 * Math.PI
        );
        annotationCtx.fill();
        annotationCtx.stroke();
        
        // Draw center dot
        annotationCtx.fillStyle = '#f44336';
        annotationCtx.beginPath();
        annotationCtx.arc(
            region.center.x * scaleX,
            region.center.y * scaleY,
            3,
            0,
            2 * Math.PI
        );
        annotationCtx.fill();
    });
    
    // Highlight selected region
    if (state.selectedRegion) {
        const allRegions = [...state.regions.positive, ...state.regions.negative];
        const region = allRegions.find(r => r.id === state.selectedRegion.id);
        
        if (region) {
            annotationCtx.strokeStyle = '#FFD700';
            annotationCtx.lineWidth = 3;
            
            annotationCtx.beginPath();
            annotationCtx.arc(
                region.center.x * scaleX,
                region.center.y * scaleY,
                region.radius * scaleX,
                0,
                2 * Math.PI
            );
            annotationCtx.stroke();
        }
    }
}

// Update region counts
function updateRegionCounts() {
    document.getElementById('positiveCount').textContent = state.regions.positive.length;
    document.getElementById('negativeCount').textContent = state.regions.negative.length;
    document.getElementById('statPositive').textContent = state.regions.positive.length;
    document.getElementById('statNegative').textContent = state.regions.negative.length;
}

// Update region list
function updateRegionList() {
    const listDiv = document.getElementById('regionList');
    listDiv.innerHTML = '';
    
    if (state.regions.positive.length === 0 && state.regions.negative.length === 0) {
        document.getElementById('regionListSection').classList.add('hidden');
        return;
    }
    
    document.getElementById('regionListSection').classList.remove('hidden');
    
    // Add positive regions
    state.regions.positive.forEach((region, idx) => {
        const div = document.createElement('div');
        div.className = 'region-item positive';
        if (state.selectedRegion && state.selectedRegion.id === region.id) {
            div.classList.add('selected');
        }
        div.innerHTML = `
            <strong>Good Event ${idx + 1}</strong><br>
            <small>Frame ${region.createdAtFrame} (${(region.createdAtFrame / state.fps).toFixed(1)}s)</small>
        `;
        div.onclick = () => selectRegion(region);
        listDiv.appendChild(div);
    });
    
    // Add negative regions
    state.regions.negative.forEach((region, idx) => {
        const div = document.createElement('div');
        div.className = 'region-item negative';
        if (state.selectedRegion && state.selectedRegion.id === region.id) {
            div.classList.add('selected');
        }
        div.innerHTML = `
            <strong>Bad Example ${idx + 1}</strong><br>
            <small>Frame ${region.createdAtFrame} (${(region.createdAtFrame / state.fps).toFixed(1)}s)</small>
        `;
        div.onclick = () => selectRegion(region);
        listDiv.appendChild(div);
    });
}

// Select a region and show its intensity graph
function selectRegion(region) {
    state.selectedRegion = region;
    updateRegionList();
    drawRegions();
    calculateIntensity(region);
}

// Calculate intensity time series
async function calculateIntensity(region) {
    try {
        const response = await fetch('/calculate_intensity', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                center_x: region.center.x,
                center_y: region.center.y,
                radius: region.radius,
                click_frame: region.createdAtFrame,
                fps: state.fps
            })
        });
        
        const data = await response.json();
        
        if (data.error) {
            console.error('Error calculating intensity:', data.error);
            return;
        }
        
        // Show chart section
        document.getElementById('chartSection').classList.remove('hidden');
        
        // Draw chart
        drawIntensityChart(data, region);
        
    } catch (error) {
        console.error('Error calculating intensity:', error);
    }
}

// Draw intensity chart using Chart.js
function drawIntensityChart(data, region) {
    const canvas = document.getElementById('intensityChart');
    const ctx = canvas.getContext('2d');
    
    // Destroy existing chart
    if (state.chart) {
        state.chart.destroy();
    }
    
    const color = region.type === 'positive' ? '#4CAF50' : '#f44336';
    
    state.chart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: data.map(d => d.time.toFixed(2)),
            datasets: [{
                label: 'Average Intensity',
                data: data.map(d => d.intensity),
                borderColor: color,
                backgroundColor: color + '33',
                borderWidth: 2,
                pointRadius: 0,
                tension: 0.1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            aspectRatio: 3,
            plugins: {
                title: {
                    display: true,
                    text: `Region at Frame ${region.createdAtFrame} - ${region.type === 'positive' ? 'Good Event' : 'Bad Example'}`,
                    font: {
                        size: 16
                    }
                },
                legend: {
                    display: true
                },
                tooltip: {
                    callbacks: {
                        title: function(context) {
                            const index = context[0].dataIndex;
                            return `Time: ${data[index].time.toFixed(2)}s (Frame ${data[index].frame})`;
                        }
                    }
                }
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Time from click (seconds)'
                    },
                    ticks: {
                        maxTicksLimit: 10
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Average Intensity'
                    },
                    beginAtZero: false
                }
            }
        }
    });
}

// Delete last region
function deleteLastRegion() {
    if (!state.annotationMode) {
        alert('Please select positive or negative mode first');
        return;
    }
    
    if (state.regions[state.annotationMode].length === 0) {
        alert('No regions to delete');
        return;
    }
    
    const deleted = state.regions[state.annotationMode].pop();
    
    // Clear selection if deleted region was selected
    if (state.selectedRegion && state.selectedRegion.id === deleted.id) {
        state.selectedRegion = null;
        document.getElementById('chartSection').classList.add('hidden');
    }
    
    updateRegionCounts();
    drawRegions();
    updateRegionList();
}

// Clear all regions
function clearAllRegions() {
    if (!confirm('Are you sure you want to clear all regions?')) {
        return;
    }
    
    state.regions = {
        positive: [],
        negative: []
    };
    state.selectedRegion = null;
    
    updateRegionCounts();
    drawRegions();
    updateRegionList();
    document.getElementById('chartSection').classList.add('hidden');
}

// Export data
async function exportData() {
    if (state.regions.positive.length === 0 && state.regions.negative.length === 0) {
        alert('Please annotate at least one region first');
        return;
    }
    
    const btn = event.target;
    const originalText = btn.innerHTML;
    btn.innerHTML = '<div class="loading"></div> Extracting data...';
    btn.disabled = true;
    
    try {
        const response = await fetch('/export_data', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                regions: state.regions,
                fps: state.fps,
                radius: state.maskRadius
            })
        });
        
        const data = await response.json();
        
        if (data.error) {
            alert('Error: ' + data.error);
            return;
        }
        
        // Download positive data
        if (data.positive.length > 0) {
            const blob = new Blob([JSON.stringify(data.positive, null, 2)], { type: 'application/json' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `${state.filename}_positive_timeseries.json`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
        }
        
        // Download negative data
        if (data.negative.length > 0) {
            const blob = new Blob([JSON.stringify(data.negative, null, 2)], { type: 'application/json' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `${state.filename}_negative_timeseries.json`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
        }
        
        alert(`Successfully exported time series data!\n\nPositive regions: ${data.positive.length}\nNegative regions: ${data.negative.length}`);
        
    } catch (error) {
        alert('Error exporting data: ' + error.message);
    } finally {
        btn.innerHTML = originalText;
        btn.disabled = false;
    }
}