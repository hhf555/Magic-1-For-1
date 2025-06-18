class Magic141Interface {
    constructor() {
        this.currentMode = 'text-to-video';
        this.isGenerating = false;
        this.startTime = null;
        this.progressInterval = null;
        this.websocket = null;
        this.galleryItems = [];
        
        this.initializeElements();
        this.bindEvents();
        this.loadGallery();
        this.connectWebSocket();
    }

    initializeElements() {
        // Mode buttons
        this.modeBtns = document.querySelectorAll('.mode-btn');
        
        // Form elements
        this.promptInput = document.getElementById('prompt');
        this.imageUploadSection = document.querySelector('.image-upload-section');
        this.imageUpload = document.getElementById('imageUpload');
        this.imageInput = document.getElementById('imageInput');
        this.imagePreview = document.getElementById('imagePreview');
        this.previewImg = document.getElementById('previewImg');
        this.removeImageBtn = document.getElementById('removeImage');
        
        // Settings
        this.settingsToggle = document.getElementById('settingsToggle');
        this.settingsContent = document.getElementById('settingsContent');
        this.videoLengthSelect = document.getElementById('videoLength');
        this.guidanceScaleRange = document.getElementById('guidanceScale');
        this.inferenceStepsSelect = document.getElementById('inferenceSteps');
        this.seedInput = document.getElementById('seed');
        this.quantizationCheckbox = document.getElementById('quantization');
        this.lowMemoryCheckbox = document.getElementById('lowMemory');
        
        // Generate button
        this.generateBtn = document.getElementById('generateBtn');
        this.btnText = this.generateBtn.querySelector('.btn-text');
        this.btnLoader = this.generateBtn.querySelector('.btn-loader');
        
        // Progress section
        this.progressSection = document.getElementById('progressSection');
        this.progressTime = document.getElementById('progressTime');
        this.progressFill = document.getElementById('progressFill');
        this.progressStatus = document.getElementById('progressStatus');
        this.progressLogs = document.getElementById('progressLogs');
        
        // Results section
        this.resultsSection = document.getElementById('resultsSection');
        this.resultVideo = document.getElementById('resultVideo');
        this.downloadBtn = document.getElementById('downloadBtn');
        this.shareBtn = document.getElementById('shareBtn');
        this.regenerateBtn = document.getElementById('regenerateBtn');
        this.generationTime = document.getElementById('generationTime');
        this.videoResolution = document.getElementById('videoResolution');
        
        // Gallery
        this.galleryGrid = document.getElementById('galleryGrid');
    }

    bindEvents() {
        // Mode selection
        this.modeBtns.forEach(btn => {
            btn.addEventListener('click', () => this.switchMode(btn.dataset.mode));
        });

        // Image upload
        this.imageUpload.addEventListener('click', () => this.imageInput.click());
        this.imageUpload.addEventListener('dragover', (e) => this.handleDragOver(e));
        this.imageUpload.addEventListener('drop', (e) => this.handleDrop(e));
        this.imageInput.addEventListener('change', (e) => this.handleImageSelect(e));
        this.removeImageBtn.addEventListener('click', () => this.removeImage());

        // Settings toggle
        this.settingsToggle.addEventListener('click', () => this.toggleSettings());

        // Range input updates
        this.guidanceScaleRange.addEventListener('input', (e) => {
            document.querySelector('.range-value').textContent = e.target.value;
        });

        // Generate button
        this.generateBtn.addEventListener('click', () => this.generateVideo());

        // Result actions
        this.downloadBtn.addEventListener('click', () => this.downloadVideo());
        this.shareBtn.addEventListener('click', () => this.shareVideo());
        this.regenerateBtn.addEventListener('click', () => this.regenerateVideo());

        // Form validation
        this.promptInput.addEventListener('input', () => this.validateForm());
        this.imageInput.addEventListener('change', () => this.validateForm());
    }

    switchMode(mode) {
        this.currentMode = mode;
        
        // Update button states
        this.modeBtns.forEach(btn => {
            btn.classList.toggle('active', btn.dataset.mode === mode);
        });

        // Show/hide image upload section
        if (mode === 'image-to-video') {
            this.imageUploadSection.style.display = 'block';
        } else {
            this.imageUploadSection.style.display = 'none';
        }

        // Update prompt placeholder
        if (mode === 'text-to-video') {
            this.promptInput.placeholder = 'Describe the video you want to generate... (e.g., A cat playing piano in a cozy living room)';
        } else {
            this.promptInput.placeholder = 'Describe how you want the image to move... (e.g., The cat starts playing the piano keys)';
        }

        this.validateForm();
    }

    handleDragOver(e) {
        e.preventDefault();
        this.imageUpload.style.borderColor = 'var(--primary-color)';
        this.imageUpload.style.background = 'rgba(99, 102, 241, 0.02)';
    }

    handleDrop(e) {
        e.preventDefault();
        this.imageUpload.style.borderColor = 'var(--border-color)';
        this.imageUpload.style.background = 'var(--surface)';
        
        const files = e.dataTransfer.files;
        if (files.length > 0 && files[0].type.startsWith('image/')) {
            this.processImage(files[0]);
        }
    }

    handleImageSelect(e) {
        const file = e.target.files[0];
        if (file && file.type.startsWith('image/')) {
            this.processImage(file);
        }
    }

    processImage(file) {
        const reader = new FileReader();
        reader.onload = (e) => {
            this.previewImg.src = e.target.result;
            this.imageUpload.style.display = 'none';
            this.imagePreview.style.display = 'block';
            this.validateForm();
        };
        reader.readAsDataURL(file);
    }

    removeImage() {
        this.imageInput.value = '';
        this.previewImg.src = '';
        this.imageUpload.style.display = 'block';
        this.imagePreview.style.display = 'none';
        this.validateForm();
    }

    toggleSettings() {
        const isActive = this.settingsToggle.classList.toggle('active');
        this.settingsContent.classList.toggle('active', isActive);
    }

    validateForm() {
        const hasPrompt = this.promptInput.value.trim().length > 0;
        const hasImage = this.currentMode === 'text-to-video' || this.imageInput.files.length > 0;
        
        const isValid = hasPrompt && hasImage && !this.isGenerating;
        this.generateBtn.disabled = !isValid;
    }

    async generateVideo() {
        if (this.isGenerating) return;

        this.isGenerating = true;
        this.startTime = Date.now();
        
        // Update UI
        this.btnText.style.display = 'none';
        this.btnLoader.style.display = 'flex';
        this.generateBtn.disabled = true;
        
        // Show progress section
        this.progressSection.style.display = 'block';
        this.progressSection.scrollIntoView({ behavior: 'smooth' });
        this.resultsSection.style.display = 'none';
        
        // Start progress timer
        this.startProgressTimer();
        
        // Prepare form data
        const formData = new FormData();
        formData.append('mode', this.currentMode);
        formData.append('prompt', this.promptInput.value.trim());
        formData.append('video_length', this.videoLengthSelect.value);
        formData.append('guidance_scale', this.guidanceScaleRange.value);
        formData.append('inference_steps', this.inferenceStepsSelect.value);
        formData.append('quantization', this.quantizationCheckbox.checked);
        formData.append('low_memory', this.lowMemoryCheckbox.checked);
        
        if (this.seedInput.value.trim()) {
            formData.append('seed', this.seedInput.value.trim());
        }
        
        if (this.currentMode === 'image-to-video' && this.imageInput.files[0]) {
            formData.append('image', this.imageInput.files[0]);
        }

        try {
            const response = await fetch('/api/generate', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const result = await response.json();
            
            if (result.success) {
                this.showResult(result);
            } else {
                throw new Error(result.error || 'Generation failed');
            }
        } catch (error) {
            console.error('Generation error:', error);
            this.showError(error.message);
        } finally {
            this.resetGenerationState();
        }
    }

    startProgressTimer() {
        this.progressInterval = setInterval(() => {
            if (this.startTime) {
                const elapsed = Date.now() - this.startTime;
                const minutes = Math.floor(elapsed / 60000);
                const seconds = Math.floor((elapsed % 60000) / 1000);
                this.progressTime.textContent = `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
            }
        }, 1000);
    }

    updateProgress(progress, status, logs) {
        this.progressFill.style.width = `${progress}%`;
        this.progressStatus.textContent = status;
        
        if (logs) {
            this.progressLogs.innerHTML += `<div class="slide-in">${logs}</div>`;
            this.progressLogs.scrollTop = this.progressLogs.scrollHeight;
        }
    }

    showResult(result) {
        const endTime = Date.now();
        const totalTime = Math.round((endTime - this.startTime) / 1000);
        
        // Hide progress, show results
        this.progressSection.style.display = 'none';
        this.resultsSection.style.display = 'block';
        this.resultsSection.classList.add('fade-in');
        this.resultsSection.scrollIntoView({ behavior: 'smooth' });
        
        // Set video source
        this.resultVideo.src = result.video_url;
        this.resultVideo.load();
        
        // Update generation info
        this.generationTime.textContent = `${totalTime}s`;
        this.videoResolution.textContent = result.resolution || '960x540';
        
        // Add to gallery
        this.addToGallery({
            video_url: result.video_url,
            prompt: this.promptInput.value.trim(),
            timestamp: new Date().toISOString(),
            generation_time: totalTime,
            mode: this.currentMode
        });
    }

    showError(message) {
        this.progressStatus.textContent = `Error: ${message}`;
        this.progressStatus.style.color = 'var(--danger-color)';
        
        // Add error to logs
        this.progressLogs.innerHTML += `<div style="color: var(--danger-color);">❌ ${message}</div>`;
        this.progressLogs.scrollTop = this.progressLogs.scrollHeight;
    }

    resetGenerationState() {
        this.isGenerating = false;
        this.startTime = null;
        
        // Reset button
        this.btnText.style.display = 'block';
        this.btnLoader.style.display = 'none';
        this.generateBtn.disabled = false;
        
        // Clear progress timer
        if (this.progressInterval) {
            clearInterval(this.progressInterval);
            this.progressInterval = null;
        }
        
        // Reset progress status color
        this.progressStatus.style.color = 'var(--text-secondary)';
        
        this.validateForm();
    }

    downloadVideo() {
        if (this.resultVideo.src) {
            const a = document.createElement('a');
            a.href = this.resultVideo.src;
            a.download = `magic141_video_${Date.now()}.mp4`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
        }
    }

    async shareVideo() {
        if (navigator.share && this.resultVideo.src) {
            try {
                await navigator.share({
                    title: 'Magic 1-For-1 Generated Video',
                    text: this.promptInput.value.trim(),
                    url: this.resultVideo.src
                });
            } catch (error) {
                console.log('Share failed:', error);
                this.copyToClipboard(this.resultVideo.src);
            }
        } else {
            this.copyToClipboard(this.resultVideo.src);
        }
    }

    copyToClipboard(text) {
        navigator.clipboard.writeText(text).then(() => {
            // Show temporary feedback
            const originalText = this.shareBtn.innerHTML;
            this.shareBtn.innerHTML = '<span>✓</span>Copied!';
            setTimeout(() => {
                this.shareBtn.innerHTML = originalText;
            }, 2000);
        });
    }

    regenerateVideo() {
        // Scroll back to form and trigger generation
        this.generateBtn.scrollIntoView({ behavior: 'smooth' });
        setTimeout(() => this.generateVideo(), 500);
    }

    addToGallery(item) {
        this.galleryItems.unshift(item);
        
        // Keep only last 12 items
        if (this.galleryItems.length > 12) {
            this.galleryItems = this.galleryItems.slice(0, 12);
        }
        
        this.saveGallery();
        this.renderGallery();
    }

    renderGallery() {
        this.galleryGrid.innerHTML = '';
        
        if (this.galleryItems.length === 0) {
            this.galleryGrid.innerHTML = `
                <div style="grid-column: 1 / -1; text-align: center; padding: 3rem; color: var(--text-muted);">
                    <div style="font-size: 3rem; margin-bottom: 1rem;">🎬</div>
                    <p>Your generated videos will appear here</p>
                </div>
            `;
            return;
        }
        
        this.galleryItems.forEach((item, index) => {
            const galleryItem = document.createElement('div');
            galleryItem.className = 'gallery-item';
            galleryItem.innerHTML = `
                <video muted>
                    <source src="${item.video_url}" type="video/mp4">
                </video>
                <div class="gallery-item-info">
                    <div class="gallery-item-prompt">${item.prompt}</div>
                    <div class="gallery-item-meta">
                        <span>${item.mode === 'text-to-video' ? '📝' : '🖼️'} ${item.mode}</span>
                        <span>${new Date(item.timestamp).toLocaleDateString()}</span>
                    </div>
                </div>
            `;
            
            galleryItem.addEventListener('click', () => {
                this.resultVideo.src = item.video_url;
                this.resultVideo.load();
                this.resultsSection.style.display = 'block';
                this.resultsSection.scrollIntoView({ behavior: 'smooth' });
            });
            
            this.galleryGrid.appendChild(galleryItem);
        });
    }

    saveGallery() {
        localStorage.setItem('magic141_gallery', JSON.stringify(this.galleryItems));
    }

    loadGallery() {
        const saved = localStorage.getItem('magic141_gallery');
        if (saved) {
            try {
                this.galleryItems = JSON.parse(saved);
            } catch (error) {
                console.error('Failed to load gallery:', error);
                this.galleryItems = [];
            }
        }
        this.renderGallery();
    }

    connectWebSocket() {
        // Connect to WebSocket for real-time progress updates
        try {
            this.websocket = new WebSocket(`ws://${window.location.host}/ws`);
            
            this.websocket.onmessage = (event) => {
                const data = JSON.parse(event.data);
                
                if (data.type === 'progress') {
                    this.updateProgress(data.progress, data.status, data.logs);
                } else if (data.type === 'complete') {
                    this.showResult(data.result);
                } else if (data.type === 'error') {
                    this.showError(data.message);
                }
            };
            
            this.websocket.onclose = () => {
                console.log('WebSocket connection closed');
                // Attempt to reconnect after 5 seconds
                setTimeout(() => this.connectWebSocket(), 5000);
            };
            
            this.websocket.onerror = (error) => {
                console.error('WebSocket error:', error);
            };
        } catch (error) {
            console.error('Failed to connect WebSocket:', error);
        }
    }
}

// Modal functions
function showAbout() {
    document.getElementById('aboutModal').classList.add('active');
}

function showHelp() {
    document.getElementById('helpModal').classList.add('active');
}

function closeModal(modalId) {
    document.getElementById(modalId).classList.remove('active');
}

// Close modals when clicking outside
document.addEventListener('click', (e) => {
    if (e.target.classList.contains('modal')) {
        e.target.classList.remove('active');
    }
});

// Initialize the application
document.addEventListener('DOMContentLoaded', () => {
    new Magic141Interface();
});

// Add some demo data for gallery if empty
setTimeout(() => {
    const app = window.magic141App;
    if (app && app.galleryItems.length === 0) {
        // Add some demo items
        const demoItems = [
            {
                video_url: '/demo/sample1.mp4',
                prompt: 'A cat playing piano in a cozy living room',
                timestamp: new Date(Date.now() - 86400000).toISOString(),
                generation_time: 45,
                mode: 'text-to-video'
            },
            {
                video_url: '/demo/sample2.mp4',
                prompt: 'Ocean waves crashing on a rocky shore at sunset',
                timestamp: new Date(Date.now() - 172800000).toISOString(),
                generation_time: 52,
                mode: 'text-to-video'
            }
        ];
        
        app.galleryItems = demoItems;
        app.renderGallery();
    }
}, 1000);

// Store app instance globally for debugging
window.magic141App = null;
document.addEventListener('DOMContentLoaded', () => {
    window.magic141App = new Magic141Interface();
});