# DQN for Traffic Light Control (Vietnam)

A Deep Q-Network agent to optimize traffic signal control in Vietnamese urban intersections using SUMO traffic simulation. Includes baseline comparison and comprehensive evaluation.

**Key Features:**
- DQN with Double DQN and target networks
- Vietnam-specific vehicle distribution (60% motorcycles, 30% cars, 10% buses/trucks)
- Fixed-time traffic controller baseline for comparison
- Comprehensive evaluation and visualization notebooks
- Thesis Chapter 4 evaluation and results

---

## Quick Start (30 seconds)

```bash
# 1. Install dependencies
pip install -r requirements.txt
brew install sumo  # macOS

# 2. Validate setup
./run.sh validate

# 3. Train DQN
./run.sh train

# 4. View results (see chapter4_evaluation.ipynb and chapter4_visualization.ipynb)
```

---

## Installation

### Step 1: Install SUMO

**macOS (Homebrew):**
```bash
brew tap dlr-ts/sumo
brew install sumo
```

**Linux/Other:**
Visit [SUMO Downloads](https://sumo.dlr.de/docs/Downloads.php)

**Set environment variable** (add to `~/.zshrc` or `~/.bashrc`):
```bash
export SUMO_HOME="/opt/homebrew/opt/sumo/share/sumo"  # adjust path for your system
export PATH="$SUMO_HOME/bin:$PATH"
```

Reload: `source ~/.zshrc`

### Step 2: Python Dependencies

```bash
pip install -r requirements.txt
```

Required packages:
- `torch>=2.0` - Deep learning
- `numpy>=1.24` - Numerical computing
- `pyyaml>=6.0` - Configuration
- `scipy>=1.10` - Statistics
- `traci>=1.20` - SUMO interface
- `tqdm` - Progress bars

### Step 3: Generate Scenario (Optional)

```bash
python src/utils/generate_scenario.py
```

The intersection scenario files are pre-generated in `data/scenarios/hn_sample/`.

---

## Usage

### Unified Launcher

```bash
./run.sh              # Show help menu
./run.sh train        # Train DQN agent
./run.sh validate     # Check environment
./run.sh demo         # Run demo with random actions
./run.sh dqn          # Run trained DQN in GUI
./run.sh baseline     # Run fixed-time controller
./run.sh compare 10   # Compare both strategies (10 episodes)
./run.sh docker-train # Train in Docker container
```

### Individual Scripts

**Training:**
```bash
python scripts/train.py
```
Trained weights saved to `outputs/dqn_vn_tls.pt`

**Evaluation:**
```bash
python scripts/compare_strategies.py --num-episodes 5 --model-path outputs/dqn_vn_tls.pt
```

**Validation:**
```bash
python scripts/validate.py
```

---

## Project Structure

```
.
├── 📚 Documentation
│   ├── README.md                         # This file
│   ├── config.yaml                      # Training configuration
│   └── requirements.txt                 # Dependencies
│
├── 📔 Jupyter Notebooks (Chapter 4)
│   ├── chapter4_evaluation.ipynb         # Evaluation pipeline & statistics
│   ├── chapter4_visualization.ipynb      # Performance charts & Excel reports
│   └── QuickStart.ipynb                 # Quick demo
│
├── 📁 Source Code (src/)
│   ├── dqn/
│   │   ├── model.py                    # Dueling DQN neural network
│   │   ├── agent.py                    # Double DQN agent
│   │   └── replay_buffer.py            # Experience replay
│   ├── env/
│   │   └── sumo_env.py                 # SUMO MDP environment wrapper
│   ├── baseline/
│   │   └── fixed_time_controller.py    # Fixed-time traffic control
│   └── utils/
│       ├── schedules.py                # Epsilon decay schedule
│       ├── generate_scenario.py        # Scenario generator
│       └── plotting.py                 # Visualization utilities
│
├── 🔧 Scripts (scripts/)
│   ├── train.py                        # DQN training
│   ├── compare_strategies.py           # DQN vs baseline comparison
│   ├── gui.py                          # Interactive GUI launcher
│   ├── run_demo.py                     # Demo evaluation
│   ├── validate.py                     # Environment validation
│   ├── setup_sumo.py                   # SUMO setup helper
│   └── common.py                       # Shared utilities
│
├── 🧪 Tests (tests/)
│   ├── test_baseline.py                # Fixed-time controller tests
│   ├── test_env.py                     # Environment tests
│   └── test_imports.py                 # Import validation
│
├── 📊 Data & Outputs
│   ├── data/scenarios/hn_sample/       # SUMO scenario files
│   └── outputs/                        # Generated models & results
│
└── 🚀 Launchers
    └── run.sh                          # Unified command launcher
```

---

## Configuration

Edit `config.yaml` to customize training:

### SUMO Settings
```yaml
sumo:
  sumocfg_path: data/scenarios/hn_sample/config.sumocfg
  tls_id: c
  phases: [0, 1, 2, 3]
  action_duration: 5
  max_steps: 3600
  gui: false
```

### Vietnam-Specific Weights
```yaml
vn_weights:
  motorcycle: 1.5  # Higher weight for motorcycles (VN traffic)
  car: 1.0
  bus: 2.0
  truck: 2.0
```

### Training Hyperparameters
```yaml
train:
  gamma: 0.99
  lr: 0.001
  batch_size: 64
  tau: 1.0
  target_update_interval: 1000
  double_dqn: true
  total_steps: 200000
```

See full config in [config.yaml](config.yaml)

---

## Vietnam-Specific Features

1. **Vehicle Type Weighting**: Motorcycles weighted 1.5× to reflect dominance
2. **Traffic Distribution**: 60% motorcycles, 30% cars, 10% buses/trucks
3. **Adaptive Reward**: Penalizes queue length and waiting time
4. **State Representation**: Per-lane queue, speed, waiting time, weighted counts

---

## Jupyter Notebooks

### chapter4_evaluation.ipynb
Comprehensive evaluation pipeline for thesis Chapter 4:
- Configuration reporting
- Evaluation metrics definitions
- Run both DQN and fixed-time strategies
- Statistical analysis (t-tests, Cohen's d, confidence intervals)
- CSV export for visualization

**Cells:**
1. Dependencies & configuration
2. Metrics definitions
3. Evaluation execution
4. Statistical analysis
5. Results export

### chapter4_visualization.ipynb
Publication-ready visualizations:
- **Figure 4.1** - Performance bar chart with error bars
- **Figure 4.2** - Time series tracking (4 subplots)
- **Figure 4.3** - Queue length distributions
- **Figure 4.4** - Improvement percentages
- Excel report generation with styled tables

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError: No module named 'traci'` | Ensure SUMO is installed and `SUMO_HOME` is set |
| `netconvert not found` | Add `$SUMO_HOME/bin` to your `PATH` |
| `XQuartz not available` (macOS) | Install: `brew install --cask xquartz` |
| Import errors in notebooks | Run first cell to install dependencies |
| SUMO not found | Run `./run.sh validate` to diagnose |

---

## Customization

- **Reward function**: Change `reward.type` to `"throughput"` or `"queue_delay"`
- **Network architecture**: Edit `src/dqn/model.py`
- **Exploration schedule**: Modify `explore.eps_start`, `eps_end`, `eps_steps` in config
- **Scenario**: Create custom `.nod.xml`, `.edg.xml`, `.rou.xml` files

---

## Project Information

| Metric | Value |
|--------|-------|
| Python Version | 3.9+ |
| PyTorch Version | 2.0+ |
| Code Files | 20+ |
| Lines of Code | ~3500 |
| Test Files | 3 |
| Notebooks | 3 |

---

## For Group Collaboration

### Before Sharing Code

1. **Validate setup:**
   ```bash
   ./run.sh validate
   ```

2. **Run tests:**
   ```bash
   python tests/test_imports.py
   python tests/test_baseline.py
   ```

### Best Practices

1. **Track these files:**
   - `src/` - Core algorithms
   - `scripts/` - Executable scripts
   - `tests/` - Unit tests
   - `config.yaml` - Configuration
   - `requirements.txt` - Dependencies

2. **Don't track:**
   - `outputs/` - Training results
   - `data/scenarios/hn_sample/` - Generated files (except templates)
   - `__pycache__/` - Python caches

3. **When adding code:**
   - Add docstrings to functions
   - Update config.yaml for new hyperparameters
   - Create tests for new features

---

## Support

For issues or questions:
1. Check the relevant script docstring: `python -c "import scripts.train; help(scripts.train.main)"`
2. Review test files for usage examples
3. Check notebook cells for implementation examples
4. See `run.sh` for all available commands

**Last Updated:** January 2026  
**Status:** Ready for thesis submission
