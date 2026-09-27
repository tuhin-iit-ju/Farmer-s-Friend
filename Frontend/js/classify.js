// Talks to the FastAPI backend's /api/predict endpoint.
//
// Expected JSON response shape (matches the updated schemas/predict.py):
// {
//   "is_paddy": true,
//   "paddy_confidence": 0.97,
//   "disease": "leaf_blast",        // machine key, or null/"normal" when healthy
//   "disease_label": "Leaf blast",  // human-readable label, or null when healthy/not paddy
//   "disease_confidence": 0.89,     // 0-1, or null when healthy/not paddy
//   "similar_images": [             // top-5 nearest images by feature-vector similarity
//     { "image_url": "/static/reference/blast_014.jpg", "label": "blast", "similarity": 0.93 },
//     ...
//   ]
// }

const API_BASE_URL = 'https://farmer-s-friend-knlm.onrender.com';
const PREDICT_ENDPOINT = `${API_BASE_URL}/api/predict`;

const dropzone = document.getElementById('dropzone');
const dropzoneContent = document.getElementById('dropzoneContent');
const fileInput = document.getElementById('fileInput');
const browseBtn = document.getElementById('browseBtn');
const previewImg = document.getElementById('previewImg');
const analyzeBtn = document.getElementById('analyzeBtn');
const resetBtn = document.getElementById('resetBtn');
const errorMessage = document.getElementById('errorMessage');
const resultPlaceholder = document.getElementById('resultPlaceholder');
const resultCard = document.getElementById('resultCard');

let selectedFile = null;

// ---------- file selection ----------

function handleFileSelected(file) {
  if (!file || !file.type.startsWith('image/')) {
    showError('Please choose an image file (JPG or PNG).');
    return;
  }
  selectedFile = file;
  hideError();

  const reader = new FileReader();
  reader.onload = (e) => {
    previewImg.src = e.target.result;
    previewImg.hidden = false;
    dropzoneContent.hidden = true;
  };
  reader.readAsDataURL(file);

  analyzeBtn.disabled = false;
  hideResult();
}

browseBtn.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', (e) => {
  if (e.target.files && e.target.files[0]) {
    handleFileSelected(e.target.files[0]);
  }
});

// drag and drop
['dragenter', 'dragover'].forEach((evt) => {
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.add('is-dragover');
  });
});
['dragleave', 'drop'].forEach((evt) => {
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.remove('is-dragover');
  });
});
dropzone.addEventListener('drop', (e) => {
  const file = e.dataTransfer.files && e.dataTransfer.files[0];
  if (file) handleFileSelected(file);
});

// ---------- reset ----------

resetBtn.addEventListener('click', resetForm);

function resetForm() {
  selectedFile = null;
  fileInput.value = '';
  previewImg.hidden = true;
  previewImg.src = '';
  dropzoneContent.hidden = false;
  analyzeBtn.disabled = true;
  analyzeBtn.hidden = false;
  analyzeBtn.textContent = 'Analyze photo';
  resetBtn.hidden = true;
  hideError();
  hideResult();
}

// ---------- analyze ----------

analyzeBtn.addEventListener('click', async () => {
  if (!selectedFile) return;

  hideError();
  analyzeBtn.disabled = true;
  analyzeBtn.textContent = 'Analyzing…';

  const formData = new FormData();
  formData.append('file', selectedFile);

  try {
    const response = await fetch(PREDICT_ENDPOINT, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      throw new Error(`Server responded with ${response.status}`);
    }

    const data = await response.json();
    renderResult(data);
    analyzeBtn.hidden = true;
    resetBtn.hidden = false;
  } catch (err) {
    showError("Couldn't reach the classifier. Check that the backend is running and try again.");
    analyzeBtn.disabled = false;
    analyzeBtn.textContent = 'Analyze photo';
  }
});

// ---------- result rendering ----------

function renderResult(data) {
  resultPlaceholder.hidden = true;
  resultCard.hidden = false;

  if (!data.is_paddy) {
    resultCard.innerHTML = `
      <span class="result-status is-not-paddy">Not paddy</span>
      <h3>This doesn't look like a paddy leaf</h3>
      <p class="result-note">Try a closer photo of a single leaf, filling most of the frame.</p>
      ${confidenceRow('Confidence', data.paddy_confidence)}
    `;
    // No disease gallery to compare against for non-paddy photos.
    return;
  }

  const isHealthy = !data.disease || data.disease === 'normal' || data.disease === 'healthy';

  if (isHealthy) {
    resultCard.innerHTML = `
      <span class="result-status is-paddy">Paddy confirmed</span>
      <span class="result-status is-healthy">Looks healthy</span>
      <h3>No disease detected</h3>
      <p class="result-note">The leaf looks healthy based on this photo.</p>
      ${confidenceRow('Paddy', data.paddy_confidence)}
      ${similarSection(data.similar_images)}
    `;
    return;
  }

  const diseaseLabel = data.disease_label || data.disease;

  resultCard.innerHTML = `
    <span class="result-status is-paddy">Paddy confirmed</span>
    <span class="result-status is-diseased">Disease found</span>
    <h3>${escapeHtml(diseaseLabel)}</h3>
    <p class="result-note">This is a model estimate — confirm with an expert before treating your crop.</p>
    ${confidenceRow('Paddy', data.paddy_confidence)}
    ${confidenceRow(escapeHtml(diseaseLabel), data.disease_confidence)}
    ${localizationSection(data.localization)}
    ${similarSection(data.similar_images)}
  `;
  attachLocalizationToggle();
}

function localizationSection(localization) {
  if (!localization || !localization.heatmap_overlay_base64) return '';

  const spotCount = (localization.spots || []).length;
  const countText = spotCount === 1 ? '1 spot detected' : `${spotCount} spots detected`;
  const heatmapSrc = `data:image/png;base64,${localization.heatmap_overlay_base64}`;
  const boxedSrc = `data:image/png;base64,${localization.annotated_image_base64}`;

  // Defaults to the heatmap view - a continuous color gradient reads
  // better than boxes when the affected area is one connected patch
  // rather than a few separate lesions. Boxes are still one tap away.
  return `
    <div class="localization-section">
      <div class="localization-header">
        <h4>Where the problem is (${countText})</h4>
        <div class="localization-toggle" role="tablist">
          <button type="button" class="toggle-btn is-active" data-view="heatmap">Heatmap</button>
          <button type="button" class="toggle-btn" data-view="boxes">Boxes</button>
        </div>
      </div>
      <img
        class="localization-img"
        id="localizationImg"
        data-heatmap-src="${heatmapSrc}"
        data-boxes-src="${boxedSrc}"
        src="${heatmapSrc}"
        alt="Leaf photo with the affected area highlighted"
      >
    </div>
  `;
}

function attachLocalizationToggle() {
  const buttons = resultCard.querySelectorAll('.localization-toggle .toggle-btn');
  const img = document.getElementById('localizationImg');
  if (!buttons.length || !img) return;

  buttons.forEach((btn) => {
    btn.addEventListener('click', () => {
      buttons.forEach((b) => b.classList.remove('is-active'));
      btn.classList.add('is-active');
      img.src = btn.dataset.view === 'boxes' ? img.dataset.boxesSrc : img.dataset.heatmapSrc;
    });
  });
}

function confidenceRow(label, value) {
  if (value === undefined || value === null) return '';
  const pct = Math.round(value * 100);
  return `
    <div class="confidence-row">
      <span class="confidence-label">${label}</span>
      <span class="confidence-track"><span class="confidence-fill" style="width:${pct}%"></span></span>
      <span class="confidence-value">${pct}%</span>
    </div>
  `;
}

function similarSection(images) {
  if (!images || images.length === 0) return '';

  const items = images
    .map((img) => {
      const pct = Math.round((img.similarity || 0) * 100);
      const src = img.image_url.startsWith('http') ? img.image_url : `${API_BASE_URL}${img.image_url}`;
      return `
        <div class="similar-item">
          <img src="${src}" alt="${escapeHtml(img.label)} reference leaf" loading="lazy">
          <div class="similar-caption">
            <strong>${escapeHtml(img.label)}</strong>
            ${pct}% match
          </div>
        </div>
      `;
    })
    .join('');

  return `
    <div class="similar-section">
      <h4>Visually similar leaves (by feature vector)</h4>
      <div class="similar-grid">${items}</div>
    </div>
  `;
}

function escapeHtml(str) {
  const div = document.createElement('div');
  div.textContent = str;
  return div.innerHTML;
}

// ---------- helpers ----------

function showError(msg) {
  errorMessage.textContent = msg;
  errorMessage.hidden = false;
}
function hideError() {
  errorMessage.hidden = true;
}
function hideResult() {
  resultCard.hidden = true;
  resultPlaceholder.hidden = false;
}
