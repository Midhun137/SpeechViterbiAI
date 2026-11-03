/**
 * script.js - Advanced Modern Controller & Interactive Trellis Engine
 * Speech Sequence Recognition Using the Viterbi Algorithm
 */

document.addEventListener("DOMContentLoaded", () => {
    // State
    let appConfig = {
        states: ["Silent", "Vowel", "Consonant"],
        observations: ["Low", "Medium", "High"],
        initial_prob: [0.6, 0.1, 0.3],
        transition_prob: [
            [0.5, 0.1, 0.4],
            [0.2, 0.5, 0.3],
            [0.1, 0.7, 0.2]
        ],
        emission_prob: [
            [0.80, 0.18, 0.02],
            [0.10, 0.30, 0.60],
            [0.20, 0.60, 0.20]
        ]
    };

    let samplePresets = [];
    let currentTokens = ["Low", "Medium", "High", "Medium", "Low"];

    // Chart instances
    let trellisChartInstance = null;
    let emissionChartInstance = null;

    // DOM Elements
    const sampleSelect = document.getElementById("sample-select");
    const presetDescText = document.getElementById("preset-description");
    const activeTokensContainer = document.getElementById("active-tokens-container");
    const seqLengthBadge = document.getElementById("seq-length-badge");
    const manualTextInput = document.getElementById("manual-text-input");
    const applyTextSeqBtn = document.getElementById("apply-text-seq-btn");
    const clearTokensBtn = document.getElementById("clear-tokens-btn");
    const resetSampleBtn = document.getElementById("reset-sample-btn");
    const runMainBtn = document.getElementById("run-main-btn");
    const navRunBtn = document.getElementById("nav-run-btn");
    const resetHmmBtn = document.getElementById("reset-hmm-btn");
    const validateHmmBtn = document.getElementById("validate-hmm-btn");

    // Audio upload DOM
    const audioDropZone = document.getElementById("audio-drop-zone");
    const audioFileInput = document.getElementById("audio-file-input");
    const loadSyntheticAudioBtn = document.getElementById("load-synthetic-audio-btn");
    const audioStatusBanner = document.getElementById("audio-status-banner");

    // Initialize application
    init();

    async function init() {
        setupTabs();
        setupEventListeners();
        await fetchConfig();
        renderTokens();
        renderMatrices();
        // Auto-run initial recognition on launch
        executeRecognition();
    }

    /* ----------------------------------------------------
       FETCH CONFIG & PRESETS
       ---------------------------------------------------- */
    async function fetchConfig() {
        try {
            const res = await fetch("/api/config");
            const data = await res.json();
            appConfig = data;
            samplePresets = data.sample_presets || [];
            populatePresetDropdown();
        } catch (err) {
            console.error("Failed to fetch config:", err);
        }
    }

    function populatePresetDropdown() {
        sampleSelect.innerHTML = "";
        samplePresets.forEach(preset => {
            const opt = document.createElement("option");
            opt.value = preset.id;
            opt.textContent = preset.name;
            sampleSelect.appendChild(opt);
        });

        if (samplePresets.length > 0) {
            presetDescText.textContent = samplePresets[0].description;
        }
    }

    /* ----------------------------------------------------
       TAB SWITCHING
       ---------------------------------------------------- */
    function setupTabs() {
        const segBtns = document.querySelectorAll(".seg-btn");
        segBtns.forEach(btn => {
            btn.addEventListener("click", () => {
                segBtns.forEach(b => b.classList.remove("active"));
                document.querySelectorAll(".tab-pane").forEach(p => p.classList.remove("active"));

                btn.classList.add("active");
                const targetTab = document.getElementById(btn.dataset.tab);
                if (targetTab) targetTab.classList.add("active");
            });
        });
    }

    /* ----------------------------------------------------
       EVENT LISTENERS
       ---------------------------------------------------- */
    function setupEventListeners() {
        // Preset Selection Change
        sampleSelect.addEventListener("change", (e) => {
            const selected = samplePresets.find(p => p.id === e.target.value);
            if (selected) {
                currentTokens = [...selected.sequence];
                presetDescText.textContent = selected.description;
                renderTokens();
            }
        });

        // Add Token Buttons
        document.querySelectorAll(".btn-token-glow").forEach(btn => {
            btn.addEventListener("click", () => {
                const tok = btn.dataset.token;
                if (currentTokens.length >= 25) {
                    alert("Maximum sequence length for clear visual decoding is 25 steps.");
                    return;
                }
                currentTokens.push(tok);
                renderTokens();
            });
        });

        // Clear Tokens
        clearTokensBtn.addEventListener("click", () => {
            currentTokens = [];
            renderTokens();
        });

        // Reset Sample
        resetSampleBtn.addEventListener("click", () => {
            currentTokens = ["Low", "Medium", "High", "Medium", "Low"];
            renderTokens();
        });

        // Apply Comma Separated Text Input
        applyTextSeqBtn.addEventListener("click", () => {
            const raw = manualTextInput.value.trim();
            if (!raw) return;

            const parts = raw.split(",").map(s => s.trim());
            const validTokens = [];
            const allowed = ["low", "medium", "high"];

            for (let p of parts) {
                const lower = p.toLowerCase();
                if (allowed.includes(lower)) {
                    validTokens.push(lower.charAt(0).toUpperCase() + lower.slice(1));
                } else if (p.length > 0) {
                    alert(`Invalid observation symbol '${p}'. Allowed symbols: Low, Medium, High.`);
                    return;
                }
            }

            if (validTokens.length === 0) {
                alert("Please provide at least one valid observation symbol (Low, Medium, High).");
                return;
            }

            currentTokens = validTokens;
            renderTokens();
            manualTextInput.value = "";
        });

        // Run Recognition Buttons
        runMainBtn.addEventListener("click", () => {
            executeRecognition();
            scrollToSection("comparison-section");
        });

        navRunBtn.addEventListener("click", () => {
            executeRecognition();
            scrollToSection("comparison-section");
        });

        // Reset HMM Matrices
        resetHmmBtn.addEventListener("click", async () => {
            await fetchConfig();
            renderMatrices();
            executeRecognition();
        });

        // Validate HMM Matrices
        validateHmmBtn.addEventListener("click", () => {
            validateAllMatricesLive(true);
        });

        // Drag & Drop Audio Upload
        setupAudioUpload();

        // Synthetic Audio Generator
        loadSyntheticAudioBtn.addEventListener("click", async () => {
            try {
                const res = await fetch("/api/demo-audio-features");
                const data = await res.json();
                currentTokens = data.observation_sequence;
                renderTokens();
                showAudioStatus(data.message);
                executeRecognition();
                scrollToSection("input-section");
            } catch (err) {
                alert("Failed to load synthetic speech features.");
            }
        });
    }

    function scrollToSection(id) {
        const el = document.getElementById(id);
        if (el) {
            el.scrollIntoView({ behavior: "smooth" });
        }
    }

    /* ----------------------------------------------------
       AUDIO UPLOAD HANDLER
       ---------------------------------------------------- */
    function setupAudioUpload() {
        if (!audioDropZone || !audioFileInput) return;

        ["dragenter", "dragover"].forEach(evt => {
            audioDropZone.addEventListener(evt, (e) => {
                e.preventDefault();
                audioDropZone.classList.add("drag-over");
            });
        });

        ["dragleave", "drop"].forEach(evt => {
            audioDropZone.addEventListener(evt, (e) => {
                e.preventDefault();
                audioDropZone.classList.remove("drag-over");
            });
        });

        audioDropZone.addEventListener("drop", (e) => {
            if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
                handleAudioFile(e.dataTransfer.files[0]);
            }
        });

        audioFileInput.addEventListener("change", (e) => {
            if (e.target.files && e.target.files.length > 0) {
                handleAudioFile(e.target.files[0]);
            }
        });
    }

    async function handleAudioFile(file) {
        if (!file.name.toLowerCase().endsWith(".wav")) {
            alert("Please select a valid .wav audio recording.");
            return;
        }

        const formData = new FormData();
        formData.append("audio", file);
        formData.append("frames", "8");

        showAudioStatus(`Processing '${file.name}' and extracting frame acoustic energy...`);

        try {
            const res = await fetch("/api/upload-audio", {
                method: "POST",
                body: formData
            });
            const data = await res.json();
            if (data.error) {
                alert(data.error);
                return;
            }

            currentTokens = data.observation_sequence;
            renderTokens();
            showAudioStatus(data.message);
            executeRecognition();
            scrollToSection("input-section");
        } catch (err) {
            alert("Error uploading audio file: " + err.message);
        }
    }

    function showAudioStatus(msg) {
        audioStatusBanner.style.display = "block";
        audioStatusBanner.textContent = msg;
    }

    /* ----------------------------------------------------
       RENDER TOKENS WITH AUDIO EQUALIZER BARS
       ---------------------------------------------------- */
    function renderTokens() {
        activeTokensContainer.innerHTML = "";
        seqLengthBadge.textContent = `${currentTokens.length} Time Steps (T=${currentTokens.length})`;

        if (currentTokens.length === 0) {
            activeTokensContainer.innerHTML = `<span style="color: var(--text-muted); font-size: 0.9rem; font-style: italic;">No observations queued. Click '+ Low', '+ Medium', or '+ High' above to build a sequence.</span>`;
            return;
        }

        const eqIcons = {
            "Low": `<span class="eq-icon eq-low"><span></span><span></span></span>`,
            "Medium": `<span class="eq-icon eq-med"><span></span><span></span><span></span></span>`,
            "High": `<span class="eq-icon eq-high"><span></span><span></span><span></span><span></span></span>`
        };

        currentTokens.forEach((token, idx) => {
            const chip = document.createElement("div");
            chip.className = `eq-token-card token-${token.toLowerCase()}`;
            chip.innerHTML = `
                ${eqIcons[token] || ''}
                <div>
                    <div class="token-step-tag">t=${idx}</div>
                    <div class="token-label">${token}</div>
                </div>
                <button type="button" class="token-kill-btn" title="Remove token" data-index="${idx}">&times;</button>
            `;

            chip.querySelector(".token-kill-btn").addEventListener("click", (e) => {
                e.stopPropagation();
                currentTokens.splice(idx, 1);
                renderTokens();
            });

            activeTokensContainer.appendChild(chip);

            if (idx < currentTokens.length - 1) {
                const arrow = document.createElement("span");
                arrow.className = "eq-arrow-divider";
                arrow.innerHTML = "&rarr;";
                activeTokensContainer.appendChild(arrow);
            }
        });
    }

    /* ----------------------------------------------------
       RENDER HMM MATRICES (WITH HEATMAP SHADING)
       ---------------------------------------------------- */
    function renderMatrices() {
        renderInitialProbabilities();
        renderTransitionMatrix();
        renderEmissionMatrix();
        validateAllMatricesLive(false);
    }

    function getHeatmapBg(val) {
        // Subtle glow color based on probability
        const clamped = Math.max(0, Math.min(1, val));
        return `rgba(0, 242, 254, ${clamped * 0.25})`;
    }

    function renderInitialProbabilities() {
        const tbody = document.getElementById("initial-prob-body");
        tbody.innerHTML = "";

        const rationales = {
            "Silent": "Natural speech begins in silence or acoustic pause prior to vocal onset.",
            "Vowel": "Rare to begin directly on an open vowel without an onset articulation.",
            "Consonant": "Frequent consonant onsets (plosives, fricatives) at word initiation."
        };

        appConfig.states.forEach((state, i) => {
            const tr = document.createElement("tr");
            const val = appConfig.initial_prob[i];
            tr.innerHTML = `
                <td><span class="state-pill state-${state.toLowerCase()}">${state}</span></td>
                <td>
                    <input type="number" step="0.05" min="0" max="1" 
                           class="prob-glow-input initial-input" 
                           data-state-idx="${i}" 
                           style="background: ${getHeatmapBg(val)};"
                           value="${val}">
                </td>
                <td style="color: var(--text-secondary); font-size: 0.825rem;">${rationales[state] || ""}</td>
            `;
            tbody.appendChild(tr);
        });

        tbody.querySelectorAll(".initial-input").forEach(input => {
            input.addEventListener("input", (e) => {
                const idx = parseInt(e.target.dataset.stateIdx);
                const val = parseFloat(e.target.value) || 0;
                appConfig.initial_prob[idx] = val;
                e.target.style.background = getHeatmapBg(val);
                validateAllMatricesLive(false);
            });
        });
    }

    function renderTransitionMatrix() {
        const tbody = document.getElementById("transition-prob-body");
        tbody.innerHTML = "";

        appConfig.states.forEach((fromState, i) => {
            const tr = document.createElement("tr");
            let rowHtml = `<td><span class="state-pill state-${fromState.toLowerCase()}">${fromState}</span></td>`;

            appConfig.states.forEach((toState, j) => {
                const val = appConfig.transition_prob[i][j];
                rowHtml += `
                    <td>
                        <input type="number" step="0.05" min="0" max="1" 
                               class="prob-glow-input trans-input" 
                               data-from="${i}" data-to="${j}" 
                               style="background: ${getHeatmapBg(val)};"
                               value="${val}">
                    </td>
                `;
            });

            rowHtml += `<td class="font-mono trans-row-sum" data-row="${i}" style="font-weight: 700;">1.000</td>`;
            tr.innerHTML = rowHtml;
            tbody.appendChild(tr);
        });

        tbody.querySelectorAll(".trans-input").forEach(input => {
            input.addEventListener("input", (e) => {
                const f = parseInt(e.target.dataset.from);
                const t = parseInt(e.target.dataset.to);
                const val = parseFloat(e.target.value) || 0;
                appConfig.transition_prob[f][t] = val;
                e.target.style.background = getHeatmapBg(val);
                validateAllMatricesLive(false);
            });
        });
    }

    function renderEmissionMatrix() {
        const tbody = document.getElementById("emission-prob-body");
        tbody.innerHTML = "";

        appConfig.states.forEach((state, i) => {
            const tr = document.createElement("tr");
            let rowHtml = `<td><span class="state-pill state-${state.toLowerCase()}">${state}</span></td>`;

            appConfig.observations.forEach((obs, j) => {
                const val = appConfig.emission_prob[i][j];
                rowHtml += `
                    <td>
                        <input type="number" step="0.05" min="0" max="1" 
                               class="prob-glow-input emission-input" 
                               data-state="${i}" data-obs="${j}" 
                               style="background: ${getHeatmapBg(val)};"
                               value="${val}">
                    </td>
                `;
            });

            rowHtml += `<td class="font-mono emission-row-sum" data-row="${i}" style="font-weight: 700;">1.000</td>`;
            tr.innerHTML = rowHtml;
            tbody.appendChild(tr);
        });

        tbody.querySelectorAll(".emission-input").forEach(input => {
            input.addEventListener("input", (e) => {
                const s = parseInt(e.target.dataset.state);
                const o = parseInt(e.target.dataset.obs);
                const val = parseFloat(e.target.value) || 0;
                appConfig.emission_prob[s][o] = val;
                e.target.style.background = getHeatmapBg(val);
                validateAllMatricesLive(false);
            });
        });
    }

    function validateAllMatricesLive(showAlert = false) {
        let allValid = true;

        // 1. Initial
        const initSum = appConfig.initial_prob.reduce((a, b) => a + b, 0);
        const initBadge = document.getElementById("initial-sum-badge");
        if (Math.abs(initSum - 1.0) < 0.01) {
            initBadge.className = "glow-pill-badge valid";
            initBadge.innerHTML = `&Sigma; = ${initSum.toFixed(3)} &check;`;
        } else {
            allValid = false;
            initBadge.className = "glow-pill-badge invalid";
            initBadge.innerHTML = `&Sigma; = ${initSum.toFixed(3)} &times; (Must = 1.0)`;
        }

        // 2. Transition
        let transValid = true;
        document.querySelectorAll(".trans-row-sum").forEach(cell => {
            const r = parseInt(cell.dataset.row);
            const rowSum = appConfig.transition_prob[r].reduce((a, b) => a + b, 0);
            cell.textContent = rowSum.toFixed(3);
            if (Math.abs(rowSum - 1.0) < 0.01) {
                cell.style.color = "var(--neon-emerald)";
            } else {
                transValid = false;
                cell.style.color = "var(--neon-rose)";
            }
        });

        const transBadge = document.getElementById("trans-valid-badge");
        if (transValid) {
            transBadge.className = "glow-pill-badge valid";
            transBadge.innerHTML = "Rows Valid &check;";
        } else {
            allValid = false;
            transBadge.className = "glow-pill-badge invalid";
            transBadge.innerHTML = "Sum Mismatch &times;";
        }

        // 3. Emission
        let emissionValid = true;
        document.querySelectorAll(".emission-row-sum").forEach(cell => {
            const r = parseInt(cell.dataset.row);
            const rowSum = appConfig.emission_prob[r].reduce((a, b) => a + b, 0);
            cell.textContent = rowSum.toFixed(3);
            if (Math.abs(rowSum - 1.0) < 0.01) {
                cell.style.color = "var(--neon-emerald)";
            } else {
                emissionValid = false;
                cell.style.color = "var(--neon-rose)";
            }
        });

        const emissionBadge = document.getElementById("emission-valid-badge");
        if (emissionValid) {
            emissionBadge.className = "glow-pill-badge valid";
            emissionBadge.innerHTML = "Rows Valid &check;";
        } else {
            allValid = false;
            emissionBadge.className = "glow-pill-badge invalid";
            emissionBadge.innerHTML = "Sum Mismatch &times;";
        }

        if (showAlert) {
            if (allValid) {
                alert("All HMM probability distributions are mathematically valid and sum to 1.0!");
            } else {
                alert("Some rows do not sum to 1.0! Please ensure each row in the transition and emission matrices sums to 1.0.");
            }
        }

        return allValid;
    }

    /* ----------------------------------------------------
       EXECUTE RECOGNITION (POST /api/recognize)
       ---------------------------------------------------- */
    async function executeRecognition() {
        if (currentTokens.length === 0) {
            alert("Observation sequence cannot be empty! Please add tokens or select a preset.");
            return;
        }

        const payload = {
            sequence: currentTokens,
            observations: currentTokens,
            observation_symbols: appConfig.observations,
            states: appConfig.states,
            initial_prob: appConfig.initial_prob,
            transition_prob: appConfig.transition_prob,
            emission_prob: appConfig.emission_prob
        };

        try {
            const res = await fetch("/api/recognize", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload)
            });

            const data = await res.json();
            if (data.error) {
                alert("Recognition Error: " + data.error);
                return;
            }

            // Render all components
            renderComparisonCards(data);
            renderDivergenceAnalysis(data);
            renderInteractiveTrellisSVG(data);
            renderCharts(data.chart_data);
            renderBackpointerMatrix(data);
            renderViterbiScoreTable(data);
            renderStepByStepWalkthrough(data.viterbi.step_details, data.viterbi.termination, data.viterbi.best_path);

        } catch (err) {
            console.error("Recognition request failed:", err);
            alert("Network or server error during recognition.");
        }
    }

    /* ----------------------------------------------------
       RENDER COMPARISON CARDS
       ---------------------------------------------------- */
    function renderComparisonCards(data) {
        const vitTokens = document.getElementById("viterbi-path-tokens");
        const indepTokens = document.getElementById("indep-path-tokens");
        const vitLogScore = document.getElementById("viterbi-log-score");
        const vitProbScore = document.getElementById("viterbi-prob-score");

        vitTokens.innerHTML = "";
        indepTokens.innerHTML = "";

        const vitPath = data.viterbi.best_path;
        const indepPath = data.independent.path;

        vitPath.forEach((state, idx) => {
            const span = document.createElement("span");
            span.className = `state-pill state-${state.toLowerCase()}`;
            span.innerHTML = `t${idx}: ${state}`;
            vitTokens.appendChild(span);

            if (idx < vitPath.length - 1) {
                const arr = document.createElement("span");
                arr.className = "eq-arrow-divider";
                arr.innerHTML = "&rarr;";
                vitTokens.appendChild(arr);
            }
        });

        indepPath.forEach((state, idx) => {
            const isDivergent = (state !== vitPath[idx]);
            const span = document.createElement("span");
            span.className = `state-pill state-${state.toLowerCase()}`;
            if (isDivergent) {
                span.style.boxShadow = "0 0 12px rgba(255, 51, 102, 0.6)";
                span.style.border = "1.5px solid var(--neon-rose)";
            }
            span.innerHTML = `t${idx}: ${state} ${isDivergent ? '&#9888;' : ''}`;
            indepTokens.appendChild(span);

            if (idx < indepPath.length - 1) {
                const arr = document.createElement("span");
                arr.className = "eq-arrow-divider";
                arr.innerHTML = "&rarr;";
                indepTokens.appendChild(arr);
            }
        });

        vitLogScore.textContent = data.viterbi.best_log_prob.toFixed(4);
        if (data.viterbi.approx_prob > 0) {
            vitProbScore.textContent = data.viterbi.approx_prob.toExponential(3);
        } else {
            vitProbScore.textContent = "< 1.0e-300";
        }
    }

    /* ----------------------------------------------------
       RENDER DIVERGENCE ANALYSIS
       ---------------------------------------------------- */
    function renderDivergenceAnalysis(data) {
        const divergenceIndicator = document.getElementById("divergence-indicator");
        const summaryPill = document.getElementById("divergence-summary-pill");
        const overallExpText = document.getElementById("overall-explanation-text");
        const tbody = document.getElementById("comparison-table-body");

        tbody.innerHTML = "";
        const divCount = data.divergence_count;

        if (divCount > 0) {
            divergenceIndicator.innerHTML = `
                <span class="pill-diverged">
                    ${divCount} Sequence Divergence${divCount > 1 ? 's' : ''} Detected!
                </span>
            `;
            summaryPill.className = "pill-diverged";
            summaryPill.textContent = `${divCount} Difference(s)`;
            overallExpText.className = "diagnostic-narrative divergence-alert";
        } else {
            divergenceIndicator.innerHTML = `
                <span class="pill-match">
                    100% Sequence Concordance
                </span>
            `;
            summaryPill.className = "pill-match";
            summaryPill.textContent = "Complete Concordance";
            overallExpText.className = "diagnostic-narrative";
        }

        overallExpText.innerHTML = `<strong>Acoustic Insight:</strong> ${data.overall_explanation}`;

        data.comparison.forEach(item => {
            const tr = document.createElement("tr");
            tr.className = item.match ? "" : "row-diverged";

            tr.innerHTML = `
                <td class="font-mono">t=${item.time}</td>
                <td><span class="hero-obs-tag">${item.observation}</span></td>
                <td><span class="state-pill state-${item.viterbi_state.toLowerCase()}">${item.viterbi_state}</span></td>
                <td><span class="state-pill state-${item.indep_state.toLowerCase()}">${item.indep_state}</span></td>
                <td>
                    <span class="${item.match ? 'pill-match' : 'pill-diverged'}">
                        ${item.match ? 'MATCH &check;' : 'DIVERGED &times;'}
                    </span>
                </td>
                <td style="font-size: 0.85rem; line-height: 1.5; color: ${item.match ? 'var(--text-secondary)' : '#fecdd3'};">
                    ${item.explanation}
                </td>
            `;

            tbody.appendChild(tr);
        });
    }

    /* ----------------------------------------------------
       INTERACTIVE SVG TRELLIS LATTICE GRAPH (SHOWSTOPPER)
       ---------------------------------------------------- */
    function renderInteractiveTrellisSVG(data) {
        const container = document.getElementById("trellis-svg-container");
        if (!container) return;

        const T = data.observation_sequence.length;
        const states = appConfig.states;
        const N = states.length;
        const vitIndices = data.viterbi.best_path_indices;

        const width = Math.max(700, T * 140);
        const height = 260;
        const padX = 80;
        const padY = 50;
        const stepX = (width - 2 * padX) / Math.max(1, T - 1);
        const stepY = (height - 2 * padY) / (N - 1);

        const stateColors = {
            0: "#38bdf8", // Silent
            1: "#ff3366", // Vowel
            2: "#00f5a0"  // Consonant
        };

        let svgHtml = `
            <svg viewBox="0 0 ${width} ${height}" width="${width}" height="${height}" xmlns="http://www.w3.org/2000/svg" style="background: transparent;">
                <defs>
                    <filter id="glow-cyan" x="-20%" y="-20%" width="140%" height="140%">
                        <feGaussianBlur stdDeviation="4" result="blur" />
                        <feComposite in="SourceGraphic" in2="blur" operator="over" />
                    </filter>
                    <linearGradient id="grad-path" x1="0%" y1="0%" x2="100%" y2="0%">
                        <stop offset="0%" stop-color="#0284c7" />
                        <stop offset="100%" stop-color="#00f2fe" />
                    </linearGradient>
                </defs>
        `;

        // 1. Draw non-optimal candidate transition lines (subtle)
        for (let t = 0; t < T - 1; t++) {
            const x1 = padX + t * stepX;
            const x2 = padX + (t + 1) * stepX;
            for (let i = 0; i < N; i++) {
                const y1 = padY + i * stepY;
                for (let j = 0; j < N; j++) {
                    const y2 = padY + j * stepY;
                    const isWinningEdge = (vitIndices[t] === i && vitIndices[t + 1] === j);
                    if (!isWinningEdge) {
                        svgHtml += `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="#1e293b" stroke-width="1.2" stroke-dasharray="3,3" opacity="0.6" />`;
                    }
                }
            }
        }

        // 2. Draw Winning Viterbi Path (Bold glowing neon line)
        for (let t = 0; t < T - 1; t++) {
            const x1 = padX + t * stepX;
            const y1 = padY + vitIndices[t] * stepY;
            const x2 = padX + (t + 1) * stepX;
            const y2 = padY + vitIndices[t + 1] * stepY;

            svgHtml += `
                <line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="#00f2fe" stroke-width="5" filter="url(#glow-cyan)" opacity="0.95" />
                <line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="#ffffff" stroke-width="2" opacity="1" />
            `;
        }

        // 3. Draw Nodes (circles with labels)
        for (let t = 0; t < T; t++) {
            const cx = padX + t * stepX;
            const obs = data.observation_sequence[t];

            // Time step column header
            svgHtml += `
                <text x="${cx}" y="20" fill="#94a3b8" font-family="'JetBrains Mono', monospace" font-size="12" font-weight="700" text-anchor="middle">t=${t}</text>
                <text x="${cx}" y="36" fill="#38bdf8" font-family="'Plus Jakarta Sans', sans-serif" font-size="11" font-weight="700" text-anchor="middle">(${obs})</text>
            `;

            for (let j = 0; j < N; j++) {
                const cy = padY + j * stepY;
                const isWinner = (vitIndices[t] === j);
                const score = data.viterbi.raw_viterbi_matrix[t][j].toFixed(2);
                const stateColor = stateColors[j];

                if (isWinner) {
                    // Winning Node: glowing outer ring
                    svgHtml += `
                        <circle cx="${cx}" cy="${cy}" r="19" fill="none" stroke="#00f2fe" stroke-width="2" filter="url(#glow-cyan)" opacity="0.8">
                            <animate attributeName="r" values="18;22;18" dur="2s" repeatCount="indefinite" />
                            <animate attributeName="opacity" values="0.8;0.3;0.8" dur="2s" repeatCount="indefinite" />
                        </circle>
                        <circle cx="${cx}" cy="${cy}" r="15" fill="#040914" stroke="#00f2fe" stroke-width="3" />
                        <text x="${cx}" y="${cy + 4}" fill="#ffffff" font-family="'JetBrains Mono', monospace" font-size="11" font-weight="800" text-anchor="middle">${states[j][0]}</text>
                    `;
                } else {
                    // Regular Node
                    svgHtml += `
                        <circle cx="${cx}" cy="${cy}" r="12" fill="#0b1222" stroke="#334155" stroke-width="2" />
                        <text x="${cx}" y="${cy + 4}" fill="#64748b" font-family="'JetBrains Mono', monospace" font-size="10" font-weight="600" text-anchor="middle">${states[j][0]}</text>
                    `;
                }

                // Score tag beneath node
                svgHtml += `
                    <text x="${cx}" y="${cy + 25}" fill="${isWinner ? '#00f2fe' : '#475569'}" font-family="'JetBrains Mono', monospace" font-size="9" text-anchor="middle">${score}</text>
                `;
            }
        }

        // Left State Axis Labels
        for (let j = 0; j < N; j++) {
            const cy = padY + j * stepY;
            svgHtml += `
                <text x="15" y="${cy + 4}" fill="${stateColors[j]}" font-family="'Space Grotesk', sans-serif" font-size="12" font-weight="700" text-anchor="start">${states[j]}</text>
            `;
        }

        svgHtml += `</svg>`;
        container.innerHTML = svgHtml;
    }

    /* ----------------------------------------------------
       RENDER CHARTS (CHART.JS)
       ---------------------------------------------------- */
    function renderCharts(chartData) {
        renderTrellisChart(chartData);
        renderEmissionChart(chartData);
    }

    function renderTrellisChart(chartData) {
        const ctx = document.getElementById("trellisChart").getContext("2d");
        if (trellisChartInstance) {
            trellisChartInstance.destroy();
        }

        trellisChartInstance = new Chart(ctx, {
            type: "line",
            data: {
                labels: chartData.labels,
                datasets: chartData.trellis_datasets
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: { mode: "index", intersect: false },
                plugins: {
                    legend: {
                        position: "top",
                        labels: {
                            color: "#94a3b8",
                            font: { family: "'Plus Jakarta Sans', sans-serif", size: 12, weight: 600 }
                        }
                    }
                },
                scales: {
                    x: {
                        grid: { color: "rgba(255, 255, 255, 0.05)" },
                        ticks: { color: "#94a3b8", font: { family: "'JetBrains Mono', monospace", size: 11 } }
                    },
                    y: {
                        grid: { color: "rgba(255, 255, 255, 0.05)" },
                        ticks: { color: "#94a3b8", font: { family: "'JetBrains Mono', monospace", size: 11 } },
                        title: {
                            display: true,
                            text: "Log Probability ln V_t(j)",
                            color: "#64748b",
                            font: { size: 11 }
                        }
                    }
                }
            }
        });
    }

    function renderEmissionChart(chartData) {
        const ctx = document.getElementById("emissionChart").getContext("2d");
        if (emissionChartInstance) {
            emissionChartInstance.destroy();
        }

        emissionChartInstance = new Chart(ctx, {
            type: "bar",
            data: {
                labels: chartData.labels,
                datasets: chartData.emission_datasets
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: "top",
                        labels: {
                            color: "#94a3b8",
                            font: { family: "'Plus Jakarta Sans', sans-serif", size: 12, weight: 600 }
                        }
                    }
                },
                scales: {
                    x: {
                        grid: { color: "rgba(255, 255, 255, 0.05)" },
                        ticks: { color: "#94a3b8", font: { family: "'JetBrains Mono', monospace", size: 11 } }
                    },
                    y: {
                        beginAtZero: true,
                        max: 1.0,
                        grid: { color: "rgba(255, 255, 255, 0.05)" },
                        ticks: { color: "#94a3b8", font: { family: "'JetBrains Mono', monospace", size: 11 } },
                        title: {
                            display: true,
                            text: "Probability P(Obs|State)",
                            color: "#64748b",
                            font: { size: 11 }
                        }
                    }
                }
            }
        });
    }

    /* ----------------------------------------------------
       RENDER BACKPOINTER MATRIX
       ---------------------------------------------------- */
    function renderBackpointerMatrix(data) {
        const headRow = document.getElementById("backpointer-head-row");
        const tbody = document.getElementById("backpointer-body");

        headRow.innerHTML = "<th>State \\ Time Step</th>";
        tbody.innerHTML = "";

        const T = data.observation_sequence.length;
        const vitIndices = data.viterbi.best_path_indices;

        for (let t = 0; t < T; t++) {
            const th = document.createElement("th");
            th.innerHTML = `t=${t}<br><span style="color: var(--neon-cyan); font-weight: normal;">${data.observation_sequence[t]}</span>`;
            headRow.appendChild(th);
        }

        appConfig.states.forEach((state, stateIdx) => {
            const tr = document.createElement("tr");
            let rowHtml = `<td><span class="state-pill state-${state.toLowerCase()}">${state}</span></td>`;

            for (let t = 0; t < T; t++) {
                const bpVal = data.viterbi.backpointer_matrix[t][state];
                const isPathActive = (vitIndices[t] === stateIdx);
                const activeClass = isPathActive ? "cell-winner" : "";

                rowHtml += `<td class="font-mono ${activeClass}" style="text-align: center;">${bpVal}</td>`;
            }

            tr.innerHTML = rowHtml;
            tbody.appendChild(tr);
        });
    }

    /* ----------------------------------------------------
       RENDER VITERBI SCORE TABLE
       ---------------------------------------------------- */
    function renderViterbiScoreTable(data) {
        const headRow = document.getElementById("viterbi-head-row");
        const tbody = document.getElementById("viterbi-score-body");

        headRow.innerHTML = "<th>State \\ Time Step</th>";
        tbody.innerHTML = "";

        const T = data.observation_sequence.length;
        const vitIndices = data.viterbi.best_path_indices;

        for (let t = 0; t < T; t++) {
            const th = document.createElement("th");
            th.innerHTML = `t=${t}<br><span style="color: var(--neon-cyan); font-weight: normal;">${data.observation_sequence[t]}</span>`;
            headRow.appendChild(th);
        }

        appConfig.states.forEach((state, stateIdx) => {
            const tr = document.createElement("tr");
            let rowHtml = `<td><span class="state-pill state-${state.toLowerCase()}">${state}</span></td>`;

            for (let t = 0; t < T; t++) {
                const scoreVal = data.viterbi.viterbi_matrix[t][state];
                const isPathActive = (vitIndices[t] === stateIdx);
                const activeClass = isPathActive ? "cell-winner" : "";

                rowHtml += `<td class="font-mono ${activeClass}" style="text-align: right;">${scoreVal.toFixed(3)}</td>`;
            }

            tr.innerHTML = rowHtml;
            tbody.appendChild(tr);
        });
    }

    /* ----------------------------------------------------
       RENDER STEP-BY-STEP MATHEMATICAL TRACE
       ---------------------------------------------------- */
    function renderStepByStepWalkthrough(stepDetails, termination, bestPath) {
        const container = document.getElementById("stepper-container");
        container.innerHTML = "";

        stepDetails.forEach((step, idx) => {
            const stepCard = document.createElement("div");
            stepCard.className = "stepper-card-box";

            let calcsHtml = "";
            step.state_calculations.forEach(calc => {
                let candidatesHtml = "";
                if (calc.candidates) {
                    candidatesHtml = `<ul class="cand-list">`;
                    calc.candidates.forEach(cand => {
                        const isWin = (cand.from_state === calc.best_prev_state);
                        candidatesHtml += `
                            <li class="${isWin ? 'winner-cand' : ''}">
                                from <strong>${cand.from_state}</strong>: V_{${step.step-1}}(${cand.prev_viterbi}) + ln(a_{${cand.from_state}&rarr;${calc.state}}) (${cand.trans_log}) = <strong>${cand.combined}</strong> ${isWin ? '&#9733; [MAX]' : ''}
                            </li>
                        `;
                    });
                    candidatesHtml += `</ul>`;
                }

                calcsHtml += `
                    <div class="state-calc-pod">
                        <div class="state-calc-pod-head">
                            <div>
                                <span class="state-pill state-${calc.state.toLowerCase()}">${calc.state}</span>
                                <span style="font-size: 0.8rem; color: var(--text-muted); margin-left: 0.6rem;">V_{${step.step}}(${calc.state})</span>
                            </div>
                            <span class="font-mono" style="font-size: 0.95rem; font-weight: 800; color: var(--neon-cyan);">
                                Log Score: ${calc.log_score}
                            </span>
                        </div>
                        <div class="calc-formula-text">${calc.formula}</div>
                        ${candidatesHtml}
                        <div style="font-size: 0.825rem; color: var(--text-secondary); margin-top: 0.4rem;">
                            ${step.step > 0 ? `Selected Backpointer &psi;_{${step.step}}(${calc.state}) = <strong>${calc.best_prev_state}</strong> | Added Emission ln(b_${calc.state}('${step.observation}')) = ${calc.emission_log}` : `Initial &pi; = ${calc.raw_prob}`}
                        </div>
                    </div>
                `;
            });

            stepCard.innerHTML = `
                <div class="stepper-head">
                    <div class="stepper-head-left">
                        <span class="stepper-badge-circle">${step.step}</span>
                        <div>
                            <span class="stepper-heading-text">${step.phase}: Observation O_{${step.step}} = "${step.observation}"</span>
                            <div style="font-size: 0.8rem; color: var(--text-muted);">Winning State on Optimal Path: <strong style="color: var(--neon-cyan);">${bestPath[idx]}</strong></div>
                        </div>
                    </div>
                    <span style="font-size: 0.8rem; color: var(--text-muted);">&#9662; Details</span>
                </div>
                <div class="stepper-body-content">
                    ${calcsHtml}
                </div>
            `;

            container.appendChild(stepCard);
        });

        // Termination Card
        const termCard = document.createElement("div");
        termCard.className = "stepper-card-box";
        termCard.innerHTML = `
            <div class="stepper-head" style="background: rgba(0, 245, 160, 0.15); border-bottom: 1px solid rgba(0, 245, 160, 0.3);">
                <div class="stepper-head-left">
                    <span class="stepper-badge-circle" style="background: var(--neon-emerald);">T</span>
                    <div>
                        <span class="stepper-heading-text">Termination & Global Backtracking Chain</span>
                        <div style="font-size: 0.8rem; color: #a7f3d0;">Reconstructed Maximum A-Posteriori (MAP) State Sequence</div>
                    </div>
                </div>
            </div>
            <div class="stepper-body-content">
                <div class="state-calc-pod" style="border-color: rgba(0, 242, 254, 0.4);">
                    <div style="font-size: 0.95rem; margin-bottom: 0.6rem; color: var(--text-pure);">
                        <strong>1. Termination:</strong> &hat;S_T = argmax_j [ V_{${stepDetails.length - 1}}(j) ] = <span class="state-pill state-${termination.final_state.toLowerCase()}">${termination.final_state}</span> (Log Score = ${termination.final_score})
                    </div>
                    <div style="font-size: 0.9rem; margin-bottom: 0.6rem; color: var(--text-secondary);">
                        <strong>2. Backtracking Chain:</strong> Tracing backwards via stored &psi;_t(j) backpointers from t=${stepDetails.length - 1} down to t=0 yields:
                    </div>
                    <div style="font-family: var(--font-mono); font-size: 1.1rem; color: var(--neon-cyan); padding: 0.85rem 1.25rem; background: #040813; border-radius: 8px; border: 1px solid rgba(0, 242, 254, 0.3);">
                        ${bestPath.join(" &rarr; ")}
                    </div>
                </div>
            </div>
        `;
        container.appendChild(termCard);
    }

});
