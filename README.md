# Reward Hacking in RL-Trained Code Models Using Weak and Strong Verifiers

## 1. Project Overview

This project investigates **reward hacking in reinforcement learning (RL)-trained code generation models**.

Reward hacking occurs when a model learns to maximize the reward provided by a verifier without necessarily producing genuinely correct programs. In code generation, this can happen when a model receives rewards based on a limited set of test cases and learns to pass those tests while failing more comprehensive tests.

The project studies whether combining weak-test correctness, additional challenge tests, metamorphic testing, and execution validity can improve the reliability of reward signals.

We use **Group Relative Policy Optimization (GRPO)** with **Low-Rank Adaptation (LoRA)** to train a small code language model and independently evaluate the generated programs using the HumanEval+ benchmark through EvalPlus.

## 2. Project Objectives

- Investigate reward hacking in RL-trained code generation models.
- Compare weak-verifier performance with strong-verifier performance.
- Implement GRPO-based fine-tuning using LoRA.
- Develop a composite reward function called BAGR.
- Compare weak-only, static challenge, policy-aware challenge, and full BAGR reward strategies.
- Perform ablation and sensitivity analyses.
- Evaluate generated programs using the official HumanEval+ test suite.

## 3. Model and Dataset

### Base Model

**Qwen2.5-Coder-0.5B-Instruct**

Hugging Face model: https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B-Instruct

The model is fine-tuned using LoRA adapters rather than updating all model parameters.

### Dataset

The project uses four selected tasks from the HumanEval Python programming benchmark.

| Task ID | Problem |
|---|---|
| HumanEval/22 | filter_integers |
| HumanEval/25 | factorize |
| HumanEval/44 | change_base |
| HumanEval/48 | is_palindrome |

- **Weak verifier:** Selected public HumanEval test cases.
- **Challenge verifier:** Additional task-specific inputs generated and validated against reference implementations.
- **Strong verifier:** HumanEval+ tests evaluated independently using EvalPlus.

The HumanEval+ hidden tests are reserved for final evaluation and are not used to construct training rewards.

HumanEval dataset: https://github.com/openai/human-eval

EvalPlus: https://github.com/evalplus/evalplus

## 4. Methodology

### 4.1 Model Fine-Tuning

The base Qwen2.5-Coder model is fine-tuned using:

- Reinforcement Learning: GRPO
- Parameter-efficient fine-tuning: LoRA
- Optimizer: AdamW
- Learning rate: 5e-6
- LoRA rank: 8
- LoRA alpha: 16
- Random seed: 42
- Training stages: 2
- Scheduled steps per stage: 5

Each training stage uses a fresh optimizer.

### 4.2 BAGR Reward Function

The project implements a composite reward function:

**BAGR = 0.40 Pw + 0.35 Pc + 0.15 Mw + 0.10 V**

Where:

- **Pw – Weak Correctness:** Correctness on selected public test cases.
- **Pc – Challenge Correctness:** Correctness on additional challenge inputs.
- **Mw – Metamorphic Robustness:** Satisfaction of predefined metamorphic relationships.
- **V – Execution Validity:** Proportion of checks that execute successfully and produce valid outputs.

All reward components are normalized between 0 and 1.

### 4.3 Experimental Configurations

The notebook compares the following experiments:

| Experiment | Description |
|---|---|
| E1 | Base model without RL training |
| E2 | GRPO with weak-test reward only |
| E3 | GRPO with weak and static challenge tests |
| E4 | GRPO with weak and policy-aware challenge tests |
| E5 | GRPO with the complete BAGR reward |
| E6 | Ablation experiments removing individual BAGR components |

### 4.4 Evaluation Metrics

The generated programs are evaluated using:

- Weak Pass Rate
- Strong Pass Rate
- Weak-to-Strong Gap
- Weak-Pass/Strong-Fail Count
- Hack Rate
- Training Efficiency
- Reward Component Scores

**Weak-to-Strong Gap:**

Gap = Weak Pass Rate − Strong Pass Rate

**Hack Rate:**

Hack Rate = (Weak Pass and Strong Fail programs / Weak Pass programs) × 100

A weak-pass/strong-fail program is treated as a potential reward-hacking case. It does not establish that the model intentionally cheated.

## 5. Technologies and Dependencies

### Programming Language
- Python 3

### Deep Learning and RL
- PyTorch
- Hugging Face Transformers
- TRL
- PEFT
- Accelerate
- Hugging Face Datasets

### Data Processing and Visualization
- NumPy
- Pandas
- Matplotlib
- psutil

### Evaluation
- EvalPlus
- HumanEval
- HumanEval+

### Main Package Versions

| Package | Version |
|---|---|
| transformers | 4.57.3 |
| trl | 0.24.0 |
| peft | 0.17.1 |
| accelerate | 1.11.0 |
| datasets | 4.3.0 |
| evalplus | 0.3.1 |
| PyTorch | 2.6 or later |

The notebook also installs supporting evaluation dependencies, including `tree-sitter`, `tree-sitter-python`, `tqdm`, `rich`, `fire`, and `termcolor`.

## 6. Installation and Setup

### Step 1: Clone the Repository

```bash
git clone https://github.com/msnigdha07/RL_Project.git
```

Navigate into the project directory:

```bash
cd RL_Project
```

### Step 2: Create a Virtual Environment (Optional)

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

Activate it on Linux or macOS:

```bash
source venv/bin/activate
```

### Step 3: Install Dependencies

The notebook contains dependency installation cells.

For a compatible environment, install the main packages:

```bash
pip install transformers==4.57.3
pip install trl==0.24.0
pip install peft==0.17.1
pip install accelerate==1.11.0
pip install datasets==4.3.0
pip install "protobuf==6.31.1"
pip install "pandas>=2.2,<3"
pip install "matplotlib>=3.9,<4"
pip install psutil
```

Install EvalPlus and its supporting dependencies using the installation instructions in the notebook:

```bash
pip install wget tempdir multipledispatch appdirs numpy tqdm termcolor fire "tree-sitter>=0.22.0" "tree-sitter-python>=0.21.0" rich stop-sequencer
pip install evalplus==0.3.1 --no-deps
```

**Note:** PyTorch must be installed with a build compatible with the selected hardware and CUDA version. The notebook checks the available device and configures the model accordingly.

## 7. Running the Project

### Recommended Environment

Google Colab with a GPU runtime, preferably an NVIDIA T4 GPU, is recommended for the training experiments.

### Step 1: Open the Notebook

Open `final_results.ipynb` in Google Colab or a compatible Jupyter environment.

### Step 2: Configure the Runtime

In Google Colab:

1. Open Runtime.
2. Select Change runtime type.
3. Choose GPU as the hardware accelerator.
4. Save the configuration.

### Step 3: Execute the Notebook

Run the notebook cells in their prescribed order.

The notebook performs the following operations:

1. Environment and dependency setup.
2. Device detection and reproducibility configuration.
3. Loading the official HumanEval public dataset.
4. Extracting and validating public test cases.
5. Setting up bounded execution for generated code.
6. Loading the Qwen2.5-Coder model.
7. Generating baseline code samples.
8. Building weak and challenge verifiers.
9. Implementing metamorphic checks and the BAGR reward.
10. Running GRPO training experiments with LoRA.
11. Generating code candidates from trained models.
12. Performing independent HumanEval+ evaluation.
13. Computing evaluation metrics and sensitivity analyses.
14. Generating result tables, CSV files, and plots.

**Important:** Run the notebook from the beginning in a fresh runtime. Some cells install or update packages and may require a kernel/runtime restart before execution continues.

## 8. Project Output

The notebook automatically creates a timestamped output directory named:

```text
BAGR_project_results/
```

The run directory contains the following folders:

```text
BAGR_project_results/
│
└── <run_timestamp>/
    │
    ├── data/
    │   ├── public_tasks.json
    │   ├── provenance.json
    │   ├── static_challenges.json
    │   ├── baseline_policy_challenges.json
    │   └── ...
    │
    ├── candidates/
    │   ├── E1.jsonl
    │   ├── E2.jsonl
    │   ├── E3.jsonl
    │   ├── E4.jsonl
    │   ├── E5.jsonl
    │   └── ...
    │
    ├── checkpoints/
    │   ├── E2/
    │   ├── E3/
    │   ├── E4/
    │   ├── E5/
    │   └── ...
    │
    ├── logs/
    │   ├── reward_components.csv
    │   └── ...
    │
    ├── results/
    │   ├── model_comparison.csv
    │   ├── per_program_results.csv
    │   ├── weak_coverage_summary.csv
    │   ├── weight_sensitivity.csv
    │   ├── training_efficiency.csv
    │   ├── official_humaneval_plus.json
    │   └── REPORT.md
    │
    └── figures/
        └── model_comparison.png
```

Additional JSON, CSV, and checkpoint files are generated during execution.

The exact output contents depend on the completed run.

## 9. Important Experimental Settings

| Parameter | Value |
|---|---|
| Base model | Qwen2.5-Coder-0.5B-Instruct |
| Random seed | 42 |
| Training algorithm | GRPO |
| Fine-tuning | LoRA |
| Stages | 2 |
| Steps per stage | 5 |
| Generations per prompt | 2 |
| Learning rate | 5e-6 |
| Weak test budget | 1 |
| Challenge test budget | 6 |
| Evaluation samples per task | 2 |
| Maximum generation tokens | 256 |
| BAGR weights | 0.40, 0.35, 0.15, 0.10 |

The experiments are lightweight pilot runs on four HumanEval tasks. Results should be interpreted within this limited evaluation scope.

## 10. Important Notes and Limitations

- GPU execution is recommended because model training can be computationally expensive.
- The first run requires internet access to download the model and benchmark data.
- Model checkpoints and generated outputs are stored in the timestamped output directory.
- HumanEval+ is used for independent final evaluation.
- The generated-code executor uses process isolation, timeouts, restricted built-ins, and allowed imports. It is a research helper, not a hardened security sandbox.
- Four tasks and a single random seed are insufficient to establish statistical significance or generalization to unseen programming problems.
- A reduction in the weak-to-strong gap does not necessarily indicate improved strong correctness.
- Post-hoc reward rescoring does not establish the effect of retraining with alternative reward weights.

## 11. Research Significance

This project provides an experimental framework for studying the difference between optimizing a limited verifier and producing genuinely correct code.

It explores how additional challenge-based verification, policy-aware test selection, metamorphic relationships, and execution-validity signals can be incorporated into reinforcement learning reward design.

The work also highlights the importance of evaluating RL-trained code models using independent and more comprehensive test suites rather than relying exclusively on the reward used during training.

## 12. Author and Repository

**Project:** Reward Hacking in RL-Trained Code Models (Weak vs. Strong Verifiers)

**GitHub Repository:**  
https://github.com/msnigdha07/RL_Project

**Main Notebook:** `final_results.ipynb`

---
