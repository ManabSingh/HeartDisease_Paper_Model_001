// ============================================
//  HEART DISEASE PREDICTION — APP.JS
//  Connects to Flask backend for REAL model predictions
// ============================================

document.addEventListener('DOMContentLoaded', () => {

    // ==========================================
    // 1. NAVIGATION — Section Switching
    // ==========================================
    const navLinks = document.querySelectorAll('.nav-link');
    const sections = document.querySelectorAll('.section');

    navLinks.forEach(link => {
        link.addEventListener('click', () => {
            const targetId = link.dataset.section;

            // Update nav active state
            navLinks.forEach(l => l.classList.remove('active'));
            link.classList.add('active');

            // Show target section
            sections.forEach(s => s.classList.remove('active'));
            document.getElementById(targetId).classList.add('active');

            // Scroll to top
            window.scrollTo({ top: 0, behavior: 'smooth' });

            // Re-trigger scroll animations for the new section
            setTimeout(triggerScrollAnimations, 100);
        });
    });

    // ==========================================
    // 2. RANGE SLIDERS — Live Value Display
    // ==========================================
    const rangeInputs = document.querySelectorAll('.form-range');
    rangeInputs.forEach(input => {
        const valueDisplay = document.getElementById(`${input.id}-val`);
        if (valueDisplay) {
            const updateVal = () => {
                valueDisplay.textContent = parseFloat(input.value).toFixed(
                    input.step && input.step !== '1' ? 1 : 0
                );
            };
            input.addEventListener('input', updateVal);
            updateVal(); // Initialize
        }
    });

    // ==========================================
    // 3. COLLECT PATIENT DATA FROM FORM
    // ==========================================
    function getPatientData() {
        return {
            age:      parseInt(document.getElementById('age').value),
            sex:      parseInt(document.getElementById('sex').value),
            cp:       parseInt(document.getElementById('cp').value),
            trestbps: parseInt(document.getElementById('trestbps').value),
            chol:     parseInt(document.getElementById('chol').value),
            fbs:      parseInt(document.getElementById('fbs').value),
            restecg:  parseInt(document.getElementById('restecg').value),
            thalach:  parseInt(document.getElementById('thalach').value),
            exang:    parseInt(document.getElementById('exang').value),
            oldpeak:  parseFloat(document.getElementById('oldpeak').value),
            slope:    parseInt(document.getElementById('slope').value),
            ca:       parseInt(document.getElementById('ca').value),
            thal:     parseInt(document.getElementById('thal').value)
        };
    }

    // ==========================================
    // 4. REAL MODEL PREDICTION VIA FLASK API
    // ==========================================
    async function predictFromServer(data) {
        try {
            const response = await fetch('/api/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data)
            });

            const result = await response.json();

            if (result.success) {
                // Update gauge with REAL ensemble prediction
                updateGauge(result.ensemble.probability);

                // Update Board of Doctors with REAL base model predictions
                updateDoctorVotes({
                    xgb:  result.base_models.xgboost,
                    lgbm: result.base_models.lightgbm,
                    rf:   result.base_models.random_forest
                });

                return true;
            } else {
                console.error('Prediction error:', result.error);
                showServerError(result.error);
                return false;
            }
        } catch (err) {
            console.error('Server connection error:', err);
            showServerError('Cannot connect to server. Make sure server.py is running.');
            return false;
        }
    }

    // Show error message in the gauge area
    function showServerError(message) {
        const pctEl = document.getElementById('gauge-percent');
        const verdictEl = document.getElementById('verdict-text');
        const badgeEl = document.getElementById('verdict-badge');

        pctEl.textContent = '—%';
        pctEl.style.color = 'var(--accent-amber)';
        badgeEl.className = 'verdict-badge high-risk';
        badgeEl.style.borderColor = 'rgba(245, 158, 11, 0.3)';
        badgeEl.style.background = 'rgba(245, 158, 11, 0.15)';
        verdictEl.textContent = '⚠️ ' + message;
        verdictEl.style.color = 'var(--accent-amber)';
    }

    // ==========================================
    // 5. GAUGE RENDERING
    // ==========================================
    // ==========================================
    // 5. GAUGE RENDERING
    // ==========================================
    function updateGauge(probability) {
        const gaugeEl = document.getElementById('gauge-fill');
        const pctEl = document.getElementById('gauge-percent');
        const badgeEl = document.getElementById('verdict-badge');
        let verdictEl = document.getElementById('verdict-text');

        // Reset any error styles safely
        badgeEl.style.borderColor = '';
        badgeEl.style.background = '';
        if (verdictEl) {
            verdictEl.style.color = '';
        }

        // Arc length calculation (semi-circle)
        const totalLength = 283; 
        const offset = totalLength * (1 - probability);

        gaugeEl.style.strokeDashoffset = offset;

        // Color based on risk
        const isHighRisk = probability > 0.5;
        const color = isHighRisk
            ? `hsl(${Math.max(0, (1 - probability) * 60)}, 85%, 55%)`
            : `hsl(${130 - probability * 60}, 70%, 50%)`;

       gaugeEl.style.stroke = color;
        
        // Replaces the exact percentage with a simple text label
        // (If you want it completely blank, change it to: pctEl.textContent = '';)
        pctEl.textContent = isHighRisk ? 'HIGH' : 'LOW';
        
        pctEl.style.color = color;
        // Re-inject the innerHTML but PRESERVE the verdict-text ID
        if (isHighRisk) {
            badgeEl.className = 'verdict-badge high-risk';
            badgeEl.innerHTML = '<span>🚨</span> <span id="verdict-text">HIGH RISK — Heart Disease Detected</span>';
        } else {
            badgeEl.className = 'verdict-badge low-risk';
            badgeEl.innerHTML = '<span>✅</span> <span id="verdict-text">LOW RISK — No Significant Pathology</span>';
        }
    }

    function updateDoctorVotes(votes) {
        const ids = { xgb: 'vote-xgb', lgbm: 'vote-lgbm', rf: 'vote-rf' };
        for (const [key, elId] of Object.entries(ids)) {
            const el = document.getElementById(elId);
            const pct = (votes[key] * 100).toFixed(1);
            el.textContent = `${pct}%`;
            el.style.color = votes[key] > 0.5 ? 'var(--accent-red)' : 'var(--accent-green)';
        }
    }

    // ==========================================
    // 6. FORM SUBMISSION — Calls Real API
    // ==========================================
    const form = document.getElementById('prediction-form');
    const btnPredict = document.getElementById('btn-predict');

    form.addEventListener('submit', async (e) => {
        e.preventDefault();

        // Show loading state
        btnPredict.textContent = '⏳ Running Models...';
        btnPredict.disabled = true;

        const data = getPatientData();
        await predictFromServer(data);

        // Reset button
        btnPredict.textContent = '🔬 Run Diagnostic Analysis';
        btnPredict.disabled = false;
    });

    // ==========================================
    // 7. FEATURE IMPORTANCE CHART (Section 2)
    // ==========================================
    const importanceData = [
        { name: 'thalach',     label: 'Max Heart Rate',      source: 22.05, target: 22.05 },
        { name: 'age',         label: 'Age',                  source: 21.89, target: 21.88 },
        { name: 'cp_3',        label: 'Typical Angina',       source: 11.94, target: 11.94 },
        { name: 'thal_3',      label: 'Reversible Defect',    source: 9.71,  target: 9.71 },
        { name: 'ca',          label: 'Major Vessels',         source: 9.55,  target: 9.55 },
        { name: 'thal_1',      label: 'Normal Thal',          source: 9.54,  target: 9.54 },
        { name: 'oldpeak',     label: 'ST Depression',        source: 6.28,  target: 6.28 },
        { name: 'chol',        label: 'Cholesterol',          source: 5.13,  target: 5.13 }
    ];

    function buildImportanceChart() {
        const container = document.getElementById('importance-chart');
        if (!container) return;

        const maxVal = Math.max(...importanceData.map(d => Math.max(d.source, d.target)));

        container.innerHTML = importanceData.map(item => {
            const sourceWidth = (item.source / maxVal) * 100;
            const targetWidth = (item.target / maxVal) * 100;
            return `
                <div class="bar-row">
                    <div class="bar-label">${item.label}</div>
                    <div style="flex: 1; display: flex; flex-direction: column; gap: 3px;">
                        <div class="bar-track">
                            <div class="bar-fill source" style="width: ${sourceWidth}%;"></div>
                        </div>
                        <div class="bar-track">
                            <div class="bar-fill target" style="width: ${targetWidth}%;"></div>
                        </div>
                    </div>
                    <div class="bar-value">${item.source.toFixed(1)}%</div>
                </div>
            `;
        }).join('');
    }

    buildImportanceChart();

    // ==========================================
    // 8. LIGHTBOX — Image Full-Screen Viewer
    // ==========================================
    const lightbox = document.getElementById('lightbox');
    const lightboxImg = document.getElementById('lightbox-img');
    const lightboxClose = document.getElementById('lightbox-close');

    document.querySelectorAll('.gallery-item[data-img]').forEach(item => {
        item.addEventListener('click', () => {
            lightboxImg.src = item.dataset.img;
            lightbox.classList.add('active');
        });
    });

    function closeLightbox() {
        lightbox.classList.remove('active');
        lightboxImg.src = '';
    }

    lightboxClose.addEventListener('click', (e) => {
        e.stopPropagation();
        closeLightbox();
    });

    lightbox.addEventListener('click', (e) => {
        if (e.target === lightbox) closeLightbox();
    });

    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && lightbox.classList.contains('active')) {
            closeLightbox();
        }
    });

    // ==========================================
    // 9. SCROLL ANIMATIONS
    // ==========================================
    function triggerScrollAnimations() {
        const elements = document.querySelectorAll('.animate-on-scroll');
        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.classList.add('visible');
                }
            });
        }, { threshold: 0.1, rootMargin: '0px 0px -50px 0px' });

        elements.forEach(el => observer.observe(el));
    }

    triggerScrollAnimations();

    // ==========================================
    // 10. INITIAL PREDICTION ON LOAD
    // ==========================================
    // Try the real server; if unavailable, show a message
    setTimeout(async () => {
        const data = getPatientData();
        const success = await predictFromServer(data);
        if (!success) {
            // Server not running — show helpful message
            console.log('Server not detected. Run: python demo/server.py');
        }
    }, 500);

});
