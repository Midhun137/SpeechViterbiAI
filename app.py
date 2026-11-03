"""
app.py
Flask Web Application for Speech Sequence Recognition Using the Viterbi Algorithm.
"""

import os
from flask import Flask, render_template, request, jsonify
from viterbi import SpeechHMM, extract_audio_features_from_wav

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max upload

# Predefined demonstration samples
SAMPLE_PRESETS = [
    {
        "id": "sample-1",
        "name": "Sample 1: Sustained Vowel Word (Standard Demo)",
        "description": "Notice how at t=3 (Medium), independent selection greedily picks Consonant, but Viterbi correctly maintains Vowel to preserve natural speech continuity.",
        "sequence": ["Low", "Medium", "High", "Medium", "Low"]
    },
    {
        "id": "sample-2",
        "name": "Sample 2: Extended Monosyllabic Utterance",
        "description": "Silent onset leading to an onset consonant, sustained high-energy vowel, coda consonant, and decay into silence.",
        "sequence": ["Low", "Low", "Medium", "High", "High", "Medium", "Low"]
    },
    {
        "id": "sample-3",
        "name": "Sample 3: Two-Syllable Rhythmic Pulse",
        "description": "Simulates two rapid vowel beats separated by a consonant transition: 'ba-ba'.",
        "sequence": ["Low", "Medium", "High", "Medium", "High", "Medium", "Low"]
    },
    {
        "id": "sample-4",
        "name": "Sample 4: Noise Perturbation in Silence",
        "description": "Demonstrates noise filtering: a momentary acoustic burst in silence is smoothed out by transition probabilities.",
        "sequence": ["Low", "Low", "Medium", "Low", "Low"]
    },
    {
        "id": "sample-5",
        "name": "Sample 5: High Energy Vocalic Exclamation",
        "description": "Immediate high energy speech ending with acoustic decay.",
        "sequence": ["High", "High", "High", "Medium", "Low"]
    }
]


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/config', methods=['GET'])
def get_config():
    """Returns the default HMM configuration and sample presets."""
    return jsonify({
        "states": SpeechHMM.DEFAULT_STATES,
        "observations": SpeechHMM.DEFAULT_OBSERVATIONS,
        "initial_prob": SpeechHMM.DEFAULT_INITIAL,
        "transition_prob": SpeechHMM.DEFAULT_TRANSITION,
        "emission_prob": SpeechHMM.DEFAULT_EMISSION,
        "sample_presets": SAMPLE_PRESETS
    })


@app.route('/api/validate', methods=['POST'])
def validate_config():
    """Validates user-customized HMM matrices."""
    data = request.get_json() or {}
    try:
        hmm = SpeechHMM(
            states=data.get('states'),
            observations=data.get('observations'),
            initial_prob=data.get('initial_prob'),
            transition_prob=data.get('transition_prob'),
            emission_prob=data.get('emission_prob')
        )
        is_valid, msg = hmm.validate()
        return jsonify({"valid": is_valid, "message": msg})
    except Exception as e:
        return jsonify({"valid": False, "message": str(e)}), 400


@app.route('/api/recognize', methods=['POST'])
def recognize():
    """
    Main speech sequence recognition endpoint.
    Runs both Viterbi and Independent Most Likely algorithms.
    """
    data = request.get_json() or {}
    # Can be passed as 'sequence' or 'observations'
    obs_sequence = data.get('sequence') or data.get('observations', [])

    if not obs_sequence or not isinstance(obs_sequence, list):
        return jsonify({"error": "Observation sequence must be a non-empty list of symbols."}), 400

    try:
        # Build HMM (use custom matrices if provided, else defaults)
        # Note: 'observation_symbols' is the alphabet (Low, Medium, High)
        alphabet = data.get('observation_symbols') or SpeechHMM.DEFAULT_OBSERVATIONS

        hmm = SpeechHMM(
            states=data.get('states'),
            observations=alphabet,
            initial_prob=data.get('initial_prob'),
            transition_prob=data.get('transition_prob'),
            emission_prob=data.get('emission_prob')
        )

        is_valid, msg = hmm.validate()
        if not is_valid:
            return jsonify({"error": f"Invalid HMM Configuration: {msg}"}), 400

        # Execute comparison
        results = hmm.compare_methods(obs_sequence)


        # Prepare Chart.js friendly format
        chart_labels = [f"t={t}\n({obs})" for t, obs in enumerate(obs_sequence)]

        # Trellis dataset (Viterbi log score per state over time)
        trellis_datasets = []
        colors = {
            "Silent": {"border": "#38bdf8", "bg": "rgba(56, 189, 248, 0.2)"},
            "Vowel": {"border": "#f43f5e", "bg": "rgba(244, 63, 94, 0.2)"},
            "Consonant": {"border": "#10b981", "bg": "rgba(16, 185, 129, 0.2)"}
        }

        raw_vit = results["viterbi"]["raw_viterbi_matrix"]
        for j, state in enumerate(hmm.states):
            state_scores = [round(raw_vit[t][j], 3) for t in range(len(obs_sequence))]
            color = colors.get(state, {"border": "#a855f7", "bg": "rgba(168, 85, 247, 0.2)"})
            trellis_datasets.append({
                "label": f"Viterbi Score ({state})",
                "data": state_scores,
                "borderColor": color["border"],
                "backgroundColor": color["bg"],
                "fill": False,
                "tension": 0.25,
                "borderWidth": 2.5
            })

        # Emission distribution per step
        emission_datasets = []
        for j, state in enumerate(hmm.states):
            em_values = [
                round(hmm.emission_prob[j, hmm.obs_to_idx[obs]], 3)
                for obs in obs_sequence
            ]
            color = colors.get(state, {"border": "#a855f7", "bg": "rgba(168, 85, 247, 0.6)"})
            emission_datasets.append({
                "label": f"Emission P(Obs|{state})",
                "data": em_values,
                "backgroundColor": color["bg"],
                "borderColor": color["border"],
                "borderWidth": 1.5
            })

        results["chart_data"] = {
            "labels": chart_labels,
            "trellis_datasets": trellis_datasets,
            "emission_datasets": emission_datasets
        }

        return jsonify(results)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/upload-audio', methods=['POST'])
def upload_audio():
    """
    Handles audio file upload, extracts frame energy,
    and quantizes into Low, Medium, High observation symbols.
    """
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided in request."}), 400

    file = request.files['audio']
    if file.filename == '':
        return jsonify({"error": "No selected file."}), 400

    try:
        num_frames = int(request.form.get('frames', 8))
        num_frames = max(3, min(num_frames, 20))  # limit to 3-20 frames for clear visual teaching

        quantized_seq, energies = extract_audio_features_from_wav(file, num_frames=num_frames)

        return jsonify({
            "filename": file.filename,
            "observation_sequence": quantized_seq,
            "frame_energies": [round(float(e), 4) for e in energies],
            "num_frames": len(quantized_seq),
            "message": f"Successfully extracted {len(quantized_seq)} observation tokens from '{file.filename}'."
        })
    except Exception as e:
        return jsonify({"error": f"Audio processing failed: {str(e)}"}), 500


@app.route('/api/demo-audio-features', methods=['GET'])
def demo_audio_features():
    """
    Generates synthetic speech acoustic features (RMS energy profile of a word like 'HELLO')
    and extracts the corresponding Low/Medium/High observation tokens.
    """
    import numpy as np
    # Synthesize 1.5 seconds of simulated speech envelope:
    # 0.0 - 0.2s: Silence (Low)
    # 0.2 - 0.5s: Consonant 'H' (Medium burst)
    # 0.5 - 1.0s: Vowel 'EH-LL-OH' (High sustained resonance)
    # 1.0 - 1.3s: Decay Consonant (Medium)
    # 1.3 - 1.5s: Silence (Low)
    energies = [0.05, 0.08, 0.42, 0.88, 0.95, 0.82, 0.51, 0.09]
    tokens = ["Low", "Low", "Medium", "High", "High", "High", "Medium", "Low"]

    return jsonify({
        "filename": "synthetic_speech_word.wav",
        "observation_sequence": tokens,
        "frame_energies": energies,
        "num_frames": len(tokens),
        "message": "Extracted acoustic features from synthetic speech utterance ('HELLO')."
    })



if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"Starting Speech Viterbi Web App on http://127.0.0.1:{port}")
    app.run(host='127.0.0.1', port=port, debug=True)
