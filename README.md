# 📚 CDPR Research: Coupled Dynamic Modeling and Hysteresis-Aware Tension Optimization

**Repository for the conference paper:** *"Coupled Dynamic Modeling and Hysteresis-Aware Tension Optimization for Cable-Driven Parallel Robots"*

[![Paper](https://img.shields.io/badge/Paper-Conference%20Publication-blue)](https://doi.org/your-doi-here)
[![IEEE](https://img.shields.io/badge/IEEE%20Format-Ready-red)](https://www.ieee.org)
[![License](https://img.shields.io/badge/License-Academic%20Use-green)](LICENSE)

## 🎯 Overview

This repository presents our research on high-precision control of Cable-Driven Parallel Robots (CDPRs). Our work presents a novel framework that integrates Bouc-Wen cable hysteresis modeling with LuGre actuator friction dynamics and implements a hysteresis-aware quadratic programming tension optimizer for real-time control.

**Key contributions:**
- ✅ Unified dynamic model combining platform flexibility, Bouc-Wen hysteresis, and LuGre friction
- ✅ Hysteresis-aware quadratic programming tension optimizer
- ✅ 73% reduction in RMS tracking error
- ✅ 72% reduction in platform vibration
- ✅ 61% smoother cable tension distribution
- ✅ 95% real-time success at 150 Hz control rates

## 📊 Performance Highlights

| Metric | Baseline | Proposed | Improvement |
|--------|----------|----------|-------------|
| **RMS Tracking Error** | 19.19 mm | 5.11 mm | **73.4% ↓** |
| **Vibration RMS** | 10.82 mm | 2.90 mm | **73.2% ↓** |
| **Tension Std Dev** | 24.27 N | 9.39 N | **61.3% ↓** |
| **Avg Loop Time** | 7.91 ms | 4.43 ms | **44.0% ↓** |
| **RT Success @150Hz** | 65% | 95% | **+30%** |

## 🧮 Mathematical Framework

### Unified Dynamic Model

Our framework integrates three key components:

#### 1. Platform Dynamics (Newton-Euler equations)

```
M(x) * ẍ + C(x, ẋ) * ẋ + G(x) = -Jᵀ(x) * τ
```

Where:
- **M(x)** = Mass matrix (6×6)
- **C(x, ẋ)** = Coriolis and centrifugal terms
- **G(x)** = Gravitational forces
- **J(x)** = Structure Jacobian
- **τ** = Cable tension vector

#### 2. Bouc-Wen Cable Hysteresis

Cable tension combines elastic and hysteretic components:

```
T_i = k_i * ε_i * L_i⁰ + α_i * z_i
```

Hysteresis state evolution:

```
ż_i = A_i * ε̇_i - β_i * |ε̇_i| * |z_i|^(n_i-1) * z_i - γ_i * ε̇_i * |z_i|^n_i
```

Where:
- **T_i** = Cable i tension
- **k_i** = Stiffness coefficient
- **ε_i** = Cable strain ((L_i - L_i⁰)/L_i⁰)
- **z_i** = Hysteresis state
- **A_i, β_i, γ_i, n_i** = Bouc-Wen parameters

#### 3. LuGre Actuator Friction

Friction torque model:

```
τ_f = σ₀ * z_f + σ₁ * ż_f + σ₂ * θ̇
```

Bristle dynamics:

```
ż_f = θ̇ - (σ₀ * |θ̇| * z_f) / g(θ̇)
```

Stribeck function:

```
g(θ̇) = F_c + (F_s - F_c) * exp(-(θ̇/v_s)²)
```

Where:
- **τ_f** = Friction torque
- **z_f** = Bristle deflection state
- **σ₀, σ₁, σ₂** = Stiffness, damping, viscous coefficients
- **F_s, F_c, v_s** = Static, Coulomb friction, Stribeck velocity

### Hysteresis-Aware QP Optimization

Cost function incorporating hysteresis effects:

```
J(τ) = ½ * τᵀ * W_τ * τ + ½ * zᵀ * W_z * z + ½ * τ̇ᵀ * W_r * τ̇
```

Subject to constraints:

```
Minimize J(τ) subject to:
  1) Jᵀ * τ = w_d    (Wrench balance)
  2) τ_min ≤ τ ≤ τ_max  (Tension limits)
```

**Note:** For properly rendered equations with LaTeX formatting, please see the [paper PDF](paper/cdpr_paper_ieee.pdf) or [LaTeX source](paper/cdpr_paper_ieee.tex).


## 🚀 Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/cdpr-research.git
cd cdpr-research
```

### 2. Compile the Paper
```bash
cd paper
pdflatex cdpr_paper_ieee.tex
bibtex cdpr_paper_ieee
pdflatex cdpr_paper_ieee.tex
pdflatex cdpr_paper_ieee.tex
```

### 3. Generate Figures (Optional)
```bash
cd code
python3 generate_cdpr_figures.py
```

### 4. Run Simulations (MATLAB)
```matlab
cd code
cdpr_simulator
```

## 🎨 Figure Integration Guide

### For LaTeX Papers
1. Copy PDF figures to your project:
   ```bash
   cp figures/*.pdf your-paper-folder/figures/
   ```

2. Use the LaTeX integration code:
   ```latex
   \begin{figure}[htbp]
       \centering
       \includegraphics[width=0.9\columnwidth]{figures/figure4_tracking_performance}
       \caption{Trajectory tracking performance comparison showing 73\% RMS error reduction across all degrees of freedom. The proposed hysteresis-aware optimization (blue) significantly outperforms the baseline method (red), particularly during trajectory reversals at t = 2.5, 5.0, and 7.5 seconds.}
       \label{fig:tracking_performance}
   \end{figure}
   ```

3. Reference in text:
   ```latex
   As shown in Fig.~\ref{fig:tracking_performance}, our proposed method achieves a 73.4\% reduction in RMS tracking error.
   ```

### For Presentations
Use the PNG versions in PowerPoint or Keynote:
- **Figure 1:** System introduction slide
- **Figures 2-3:** Modeling methodology
- **Figure 4:** Main results highlight
- **Figures 5-7:** Detailed performance analysis
- **Figure 8:** Conclusion summary

## 📖 Paper Sections and Corresponding Figures

| Section | Recommended Figures | Key Metrics to Highlight |
|---------|-------------------|---------------------------|
| **Abstract** | Figure 8 (summary) | 73% error reduction, 95% real-time success |
| **Introduction** | Figure 1 (system) | CDPR configuration, 8-cable setup |
| **Modeling** | Figures 2-3 | Bouc-Wen parameters, LuGre friction |
| **Methodology** | Figure 7 | QP solve time, computational complexity |
| **Results** | Figures 4-6 | Tracking error, vibration, tension |
| **Discussion** | Figure 8 | Comprehensive comparison |
| **Conclusion** | Figure 8 | Overall improvements, future work |

## 📈 Results Interpretation

### Primary Finding (Figure 4)
- **73% tracking error reduction** across all degrees of freedom
- Most significant improvement during trajectory reversals
- Consistent performance across X, Y, Z axes (72-75% improvement)

### Vibration Suppression (Figure 6)
- **72.7% reduction** in vibration RMS amplitude
- **87.8% reduction** in vibration energy (0-10 Hz range)
- Effective damping of first bending mode (3.2 Hz)

### Real-Time Feasibility (Figure 7)
- Average loop time: **4.43 ms** (fits 150 Hz control)
- 95th percentile: **6.12 ms** (safety margin)
- Real-time success rate: **95%** (was 65% with baseline)

## 🛠️ Customization

### Modify System Parameters
Edit `data/parameters.json`:
```json
{
  "platform_mass": 15.0,
  "workspace": [4.0, 4.0, 3.0],
  "cable_stiffness": 1000,
  "control_frequency": 150
}
```

### Extend the Model
```matlab
% Add new nonlinear effects in bouc_wen_lugre_model.m
function dx = extended_model(t, x, u, params)
    % Existing Bouc-Wen + LuGre dynamics
    dx_bw = bouc_wen_dynamics(x, params);
    dx_lg = lugre_dynamics(x, params);
    
    % Add your custom dynamics here
    dx_custom = custom_dynamics(x, params);
    
    dx = dx_bw + dx_lg + dx_custom;
end
```

## 📝 Citation

If you use this work in your research, please cite:

```bibtex
@inproceedings{smith2025integrated,
  title={Coupled Dynamic Modeling and Hysteresis-Aware Tension Optimization for Cable-Driven Parallel Robots},
  author={Mfeuter Joseph Tachia, Alexander Maloletov},
  booktitle={Proceedings of the International Conference on Control, Automation, Robotics and Vision Engineering},
  year={2026},
  pages={1--8},
  doi={10.1109/PSUEDO.119.2026}
}
```

## 📄 License

This work is licensed under the **Academic Use License** - see the [LICENSE](LICENSE) file for details.

You are free to:
- Use the figures in your academic papers
- Modify and adapt the code for research purposes
- Share with collaborators and students

Please contact the authors for commercial use.

## 📧 Contact

For questions about this research:
- **Mfeuter Joseph Tachia**: m.tachia@innopolis.university


## 📊 Related Research

For more information on CDPRs:
- [Reinforcement Learning-Based Control for Cable Sag Compensation in Cable-Driven Parallel Robots Using Soft Actor-Critic Algorithm](https://doi.org/10.1109/CTS67336.2025.11196696) by M. J. Tachia, A. V. Maloletov
- [Applications of Dynamic Models for Cable-Driven Parallel Robots: A Comprehensive Review](https://doi.org/10.20537/nd251101) by M. J. Tachia, A. V. Maloletov


**Note:** For properly rendered mathematical equations, please view the paper PDF or compile the LaTeX source. GitHub Markdown doesn't support LaTeX rendering natively.



