# Speech Sequence Recognition Using the Viterbi Algorithm

A full-stack, beginner-friendly, and mathematically rigorous web application demonstrating speech sequence decoding using a **Hidden Markov Model (HMM)** and the **Viterbi Algorithm**.

The application decodes hidden acoustic speech states (**Silent**, **Vowel**, **Consonant**) from observed acoustic energy features (**Low**, **Medium**, **High**) and provides a direct, side-by-side comparison between **Dynamic Programming (Viterbi)** and **Instantaneous Independent State Selection**.

---

## 1. Project Overview & Problem Statement

### Problem Statement
In automatic speech recognition (ASR), the true phonetic units (hidden articulatory states) cannot be observed directly from raw audio waveforms. Instead, digital signal processing extracts observable features (e.g., energy levels, zero-crossing rates, or MFCCs) over discrete time windows.

The goal is to determine the most likely sequence of hidden states $S = (s_1, s_2, \dots, s_T)$ given an observed feature sequence $O = (o_1, o_2, \dots, o_T)$.

### Key Educational Demonstration
This project contrasts two fundamental classification philosophies:
1. **Method 1: Viterbi Algorithm (Dynamic Programming)**:
   - Maximizes joint likelihood $P(S_1 \dots S_T, O_1 \dots O_T)$.
   - Enforces Markov transition rules ($a_{ij}$) between consecutive states.
   - Finds the **globally optimal** phonetic trajectory.
2. **Method 2: Individual Most Likely State (Instantaneous Emission Maximization)**:
   - Selects $\arg\max_j P(O_t \mid S_j)$ locally at each time step.
   - Ignores temporal continuity ($a_{ij}$), often selecting phonetically implausible or illegal transitions.

---

## 2. Technology Stack

- **Backend**: Python 3.10+, Flask
- **Algorithm Engine**: NumPy, SciPy (Logarithmic Viterbi Decoding & Acoustic Feature Extraction)
- **Frontend**: Modern Responsive HTML5, CSS3, Vanilla JavaScript (ES6+)
- **Visualizations**: Chart.js 4.x (Interactive Trellis evolution line chart & acoustic emission bar chart)
- **Theme**: Dark Navy Blue (`#070b14`, `#0f172a`) and Light Cyan / Blue (`#38bdf8`, `#818cf8`)

---

## 3. Project Directory Structure

```
speech-recognition-viterbi/
│
├── app.py                     # Flask web server and REST API endpoints
├── viterbi.py                 # Core HMM and Viterbi dynamic programming engine
├── requirements.txt           # Python package dependencies
├── sample_speech.wav          # Synthetic speech WAV for file upload demonstration
├── README.md                  # Comprehensive documentation and viva presentation guide
│
├── templates/
│   └── index.html             # Main dashboard UI with formulas and viva guide
│
└── static/
    ├── css/
    │   └── style.css          # Dark navy blue theme, glassmorphism, responsive styles
    └── js/
        └── script.js          # Interactive controller, Chart.js integrations, live validation
```

---

## 4. Mathematical Foundation

### Hidden Markov Model Parameters ($\lambda = (\pi, A, B)$)
- **Hidden States ($N=3$)**: $S = \{\text{Silent (0)}, \text{Vowel (1)}, \text{Consonant (2)}\}$
- **Observation Symbols ($M=3$)**: $V = \{\text{Low (0)}, \text{Medium (1)}, \text{High (2)}\}$
- **Initial Probabilities ($\pi$)**:
  $$\pi = [\pi_{\text{Silent}}=0.6, \, \pi_{\text{Vowel}}=0.1, \, \pi_{\text{Consonant}}=0.3]$$
- **Transition Probability Matrix ($A = [a_{ij}]$)**:
  $$A = \begin{bmatrix}
  0.5 & 0.1 & 0.4 \\
  0.2 & 0.5 & 0.3 \\
  0.1 & 0.7 & 0.2
  \end{bmatrix}$$
  *Row 0 (Silent) $\to$ [Silent: 0.5, Vowel: 0.1, Consonant: 0.4]*
  *Row 1 (Vowel) $\to$ [Silent: 0.2, Vowel: 0.5, Consonant: 0.3]*
  *Row 2 (Consonant) $\to$ [Silent: 0.1, Vowel: 0.7, Consonant: 0.2]*

- **Emission Probability Matrix ($B = [b_j(k)]$ where $b_j(k) = P(O_t=k \mid S_t=j)$)**:
  $$B = \begin{bmatrix}
  0.80 & 0.18 & 0.02 \\
  0.10 & 0.30 & 0.60 \\
  0.20 & 0.60 & 0.20
  \end{bmatrix}$$
  *Silent: Low=0.80, Medium=0.18, High=0.02*
  *Vowel: Low=0.10, Medium=0.30, High=0.60*
  *Consonant: Low=0.20, Medium=0.60, High=0.20*

### The Viterbi Dynamic Programming Recurrence
Standard product form:
$$V_t(j) = \max_{1 \le i \le N} \left[ V_{t-1}(i) \cdot a_{ij} \right] \cdot b_j(O_t)$$

Backpointer:
$$\psi_t(j) = \arg\max_{1 \le i \le N} \left[ V_{t-1}(i) \cdot a_{ij} \right]$$

### Logarithmic Transformation (Numerical Underflow Protection)
Multiplying numerous probabilities $< 1$ leads to IEEE 754 floating-point underflow ($P \to 0.0$). We transform multiplications into additions using natural logarithms:
$$\ln V_t(j) = \max_{1 \le i \le N} \left[ \ln V_{t-1}(i) + \ln a_{ij} \right] + \ln b_j(O_t)$$

---

## 5. Installation & Setup Instructions

### Prerequisites
- Python 3.8, 3.9, 3.10, 3.11, or 3.12 installed on your machine.

### Step 1: Clone or Navigate to the Project Directory
```bash
cd speech-recognition-viterbi
```

### Step 2: (Optional but Recommended) Create a Virtual Environment
**On Windows:**
```powershell
python -m venv venv
.\venv\Scripts\activate
```

**On macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Required Dependencies
```bash
pip install -r requirements.txt
```

---

## 6. Running the Application

### Start the Flask Server
```bash
python app.py
```

You will see output similar to:
```
Starting Speech Viterbi Web App on http://127.0.0.1:5000
 * Serving Flask app 'app'
 * Debug mode: on
 * Running on http://127.0.0.1:5000
```

### Open in Your Browser
Open your browser and navigate to:
```
http://127.0.0.1:5000
```

---

## 7. Sample Demonstration & Output

### Input Observation Sequence
$$\text{Low} \longrightarrow \text{Medium} \longrightarrow \text{High} \longrightarrow \text{Medium} \longrightarrow \text{Low}$$

### Execution Results
- **Method 1 (Viterbi Dynamic Programming)**:
  $$\text{Silent} \longrightarrow \text{Consonant} \longrightarrow \text{Vowel} \longrightarrow \mathbf{Vowel} \longrightarrow \text{Silent}$$
  - **Final Log-Probability Score**: `-6.7583`
  - **Joint Likelihood**: $\approx 1.161 \times 10^{-3}$

- **Method 2 (Independent Most Likely State)**:
  $$\text{Silent} \longrightarrow \text{Consonant} \longrightarrow \text{Vowel} \longrightarrow \mathbf{Consonant} \longrightarrow \text{Silent}$$

### Divergence Analysis (Why Do They Differ?)
At time step $t=3$ (Observation = **Medium**):
- **Independent Method** selects **Consonant** because:
  $$P(\text{Medium} \mid \text{Consonant}) = 0.60 > P(\text{Medium} \mid \text{Vowel}) = 0.30$$
  This is a greedy, myopic choice.
- **Viterbi Algorithm** selects **Vowel** because:
  1. Transition continuity: $P(\text{Vowel} \to \text{Vowel}) = 0.50$, whereas $P(\text{Vowel} \to \text{Consonant}) = 0.30$.
  2. Future transition feasibility: Transitioning from **Vowel** to **Silent** at $t=4$ has probability $0.20$, whereas transitioning from **Consonant** to **Silent** at $t=4$ is heavily penalized ($0.10$).
  3. Total path likelihood:
     $$\text{Vowel} \to \text{Vowel} \to \text{Silent}: 0.50 \times 0.30 \times 0.20 \times 0.80 = \mathbf{0.0240}$$
     $$\text{Vowel} \to \text{Consonant} \to \text{Silent}: 0.30 \times 0.60 \times 0.10 \times 0.80 = \mathbf{0.0144}$$
- **Conclusion**: Viterbi correctly maintains the sustained vowel, matching natural human speech phonetics!

---

## 8. College Faculty Demonstration & Viva Guide

When presenting this project to a faculty member or examiner, follow this 4-step demonstration flow:

### 1. The High-Level Overview
> *"Sir/Ma'am, in automatic speech recognition, sound waves contain noise and formants. Our system models speech as a Hidden Markov Model where articulatory states (Silent, Vowel, Consonant) are hidden, and observable acoustic energy levels (Low, Medium, High) are measured from audio."*

### 2. Live Demo of Divergence
1. In the web dashboard, select **Sample 1: Sustained Vowel Word**.
2. Click **Run Recognition**.
3. Point to the **Comparison Cards**:
   > *"Notice that at step $t=3$ (Observation 'Medium'), the Independent method chose Consonant, but Viterbi chose Vowel."*
4. Explain the difference:
   > *"Independent state selection is myopic—it looks only at the instantaneous emission table where Medium has a 0.60 probability for Consonant. However, in speech, a vowel naturally sustains over time. Viterbi considers the Markov transition probability $a_{ij}$ and calculates that sustaining the vowel yields a higher joint probability for the entire word."*

### 3. Mathematical Rigor & Underflow Protection
1. Scroll down to the **Step-by-Step Viterbi Mathematical Trace**.
2. Show the expansion:
   > *"We use natural logarithms $\ln V_t(j) = \max_i [\ln V_{t-1}(i) + \ln a_{ij}] + \ln b_j(O_t)$ because multiplying fractional probabilities over long utterances causes arithmetic underflow to zero."*
3. Show the **Backpointer Matrix** table and how backtracking reconstructs the path from $t=T-1$ down to $t=0$.

### 4. Interactive Audio File Feature Extraction
1. Switch to the **Real/Synthetic Audio** tab.
2. Click **Synthesize Speech Envelope** or upload `sample_speech.wav`.
3. Show how the Python backend computes RMS acoustic energy windows and quantizes them into observation tokens automatically.

---

## 9. Common Viva Questions & Answers

1. **Q: Why is an HMM called 'Hidden'?**
   - **A**: Because the underlying state sequence (phonemes/articulatory positions) cannot be directly seen; only the resulting acoustic features (energy, frequency spectrum) are observed.

2. **Q: What is the time complexity of the Viterbi Algorithm vs. Brute Force?**
   - **A**: Brute force evaluating all paths takes $\mathcal{O}(N^T)$ exponential time. Viterbi uses dynamic programming and takes $\mathcal{O}(T \times N^2)$ polynomial time. For $N=3$ states and $T=10$ steps, brute force is $3^{10} = 59,049$ evaluations, whereas Viterbi requires only $10 \times 9 = 90$ operations.

3. **Q: Can the Independent method produce an impossible sequence?**
   - **A**: Yes! If transition probability $a_{ij} = 0$ between two states, the Independent method can still choose them sequentially if their individual emissions are high. Viterbi will multiply by $\ln(0) = -\infty$, strictly preventing illegal transitions.

