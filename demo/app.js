// ============================================
//  HEART DISEASE PREDICTION — APP.JS
//  Client-side prediction simulation, UI logic
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
    // 3. PREDICTION ENGINE (Client-Side Simulation)
    // ==========================================

    // Feature importance weights from the trained ensemble model.
    // These are the normalized Ensemble_Average values from feature_importance_table.csv.
    // Used to create a weighted scoring function for client-side simulation.
    const FEATURE_WEIGHTS = {
        thalach:    0.2205,
        age:        0.2189,
        cp_3:       0.1194,
        thal_3:     0.0971,
        ca:         0.0955,
        thal_1:     0.0954,
        oldpeak:    0.0628,
        chol:       0.0513,
        slope_1:    0.0113,
        restecg_2:  0.0095,
        trestbps:   0.0085,
        cp_2:       0.0020,
        cp_1:       0.0018,
        fbs:        0.0017,
        slope_0:    0.0013,
        slope_2:    0.0011,
        restecg_0:  0.0011,
        exang:      0.0007,
        sex:        0.0002,
        thal_2:     0.0001,
        cp_0:       0.0001,
        restecg_1:  0.0000
    };

    // Normalization ranges (approximate from Cleveland dataset)
    const NORM_RANGES = {
        age:      { min: 29, max: 77 },
        trestbps: { min: 94, max: 200 },
        chol:     { min: 126, max: 564 },
        thalach:  { min: 71, max: 202 },
        oldpeak:  { min: 0, max: 6.2 },
        ca:       { min: 0, max: 4 }
    };

    function normalize(value, min, max) {
        return (value - min) / (max - min);
    }

    function sigmoid(x) {
        return 1 / (1 + Math.exp(-x));
    }

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

    function computeRiskScore(data) {
        // Build one-hot encoded feature vector
        const features = {
            age:        normalize(data.age, NORM_RANGES.age.min, NORM_RANGES.age.max),
            sex:        data.sex,
            trestbps:   normalize(data.trestbps, NORM_RANGES.trestbps.min, NORM_RANGES.trestbps.max),
            chol:       normalize(data.chol, NORM_RANGES.chol.min, NORM_RANGES.chol.max),
            fbs:        data.fbs,
            thalach:    1 - normalize(data.thalach, NORM_RANGES.thalach.min, NORM_RANGES.thalach.max), // Inverse: lower HR = higher risk
            exang:      data.exang,
            oldpeak:    normalize(data.oldpeak, NORM_RANGES.oldpeak.min, NORM_RANGES.oldpeak.max),
            ca:         normalize(data.ca, NORM_RANGES.ca.min, NORM_RANGES.ca.max),
            cp_0:       data.cp === 0 ? 1 : 0,
            cp_1:       data.cp === 1 ? 1 : 0,
            cp_2:       data.cp === 2 ? 1 : 0,
            cp_3:       data.cp === 3 ? 1 : 0,
            restecg_0:  data.restecg === 0 ? 1 : 0,
            restecg_1:  data.restecg === 1 ? 1 : 0,
            restecg_2:  data.restecg === 2 ? 1 : 0,
            slope_0:    data.slope === 0 ? 1 : 0,
            slope_1:    data.slope === 1 ? 1 : 0,
            slope_2:    data.slope === 2 ? 1 : 0,
            thal_1:     data.thal === 1 ? 1 : 0,
            thal_2:     data.thal === 2 ? 1 : 0,
            thal_3:     data.thal === 3 ? 1 : 0
        };

        // Weighted sum
        let score = -0.5; // bias
        for (const [key, weight] of Object.entries(FEATURE_WEIGHTS)) {
            score += (features[key] || 0) * weight * 4.5; // Amplify for sigmoid spread
        }

        return sigmoid(score);
    }

    // Simulate individual model opinions (slight random offsets from ensemble)
    function simulateModelVotes(ensembleScore) {
        const jitter = () => (Math.random() - 0.5) * 0.12;
        return {
            xgb:  Math.max(0, Math.min(1, ensembleScore + jitter())),
            lgbm: Math.max(0, Math.min(1, ensembleScore + jitter())),
            rf:   Math.max(0, Math.min(1, ensembleScore + jitter()))
        };
    }

    // ==========================================
    // 4. GAUGE RENDERING
    // ==========================================
    function updateGauge(probability) {
        const gaugeEl = document.getElementById('gauge-fill');
        const pctEl = document.getElementById('gauge-percent');
        const badgeEl = document.getElementById('verdict-badge');
        const verdictEl = document.getElementById('verdict-text');

        // Arc length calculation (semi-circle)
        const totalLength = 283; // approximate circumference of the semi-arc
        const offset = totalLength * (1 - probability);

        gaugeEl.style.strokeDashoffset = offset;

        // Color based on risk
        const isHighRisk = probability > 0.5;
        const color = isHighRisk
            ? `hsl(${Math.max(0, (1 - probability) * 60)}, 85%, 55%)`
            : `hsl(${130 - probability * 60}, 70%, 50%)`;

        gaugeEl.style.stroke = color;
        pctEl.textContent = `${(probability * 100).toFixed(1)}%`;
        pctEl.style.color = color;

        if (isHighRisk) {
            badgeEl.className = 'verdict-badge high-risk';
            verdictEl.textContent = '🚨 HIGH RISK — Heart Disease Detected';
        } else {
            badgeEl.className = 'verdict-badge low-risk';
            verdictEl.textContent = '✅ LOW RISK — No Significant Pathology';
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
    // 5. FORM SUBMISSION
    // ==========================================
    const form = document.getElementById('prediction-form');
    form.addEventListener('submit', (e) => {
        e.preventDefault();

        const data = getPatientData();
        const risk = computeRiskScore(data);
        const votes = simulateModelVotes(risk);

        updateGauge(risk);
        updateDoctorVotes(votes);
    });

    // ==========================================
    // 6. FEATURE IMPORTANCE CHART (Section 2)
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
    // 7. LIGHTBOX — Image Full-Screen Viewer
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
    // 8. SCROLL ANIMATIONS
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
    // 9. INITIAL DEMO PREDICTION
    // ==========================================
    // Run an initial prediction with default values so the gauge isn't empty
    setTimeout(() => {
        const data = getPatientData();
        const risk = computeRiskScore(data);
        const votes = simulateModelVotes(risk);
        updateGauge(risk);
        updateDoctorVotes(votes);
    }, 500);

});
