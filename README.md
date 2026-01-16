# MDD-Thinker

**MDD-Thinker** is a large language model (LLM)-based diagnostic framework designed for **major depressive disorder (MDD)**. The system integrates **supervised fine-tuning (SFT)** and **reinforcement learning (RL)** to enhance both **diagnostic accuracy** and **interpretability**, allowing the model to generate structured reasoning paths for clinical decision support.

---

## Overview

MDD-Thinker aims to provide a scalable and explainable solution for intelligent psychiatric assessment. Its architecture combines domain-specific reasoning with LLM capabilities to produce:

- Accurate MDD diagnosis
- Structured and clinically coherent reasoning paths
- Integration of multimodal clinical and psychological knowledge (text-based)

### Core Architecture
<img src="imgs/arch.png" width="70%">  
*Figure: Core workflow of MDD-Thinker including data processing, SFT training, and RL fine-tuning.*

---

## Data Processing

MDD-Thinker leverages two main sources of data:

1. **UK Biobank Clinical Data**  
   - Used to construct structured reasoning samples for MDD diagnosis.
   - Includes demographic, lifestyle, psychosocial, clinical, and biochemical features.
   - Extensive data filtering applied to ensure clinical representativeness and completeness.

2. **Publicly Available Psychology-Related Datasets**  
   - Provides additional mental health knowledge for general psychological reasoning.
   - Includes QA pairs, dialogues, and multiple-choice questions related to depression and mental health.

The **reasoning dataset** is generated in three steps:

1. **Feature Selection** – selecting 22 clinically and statistically relevant variables.
2. **Data Filtering** – removing low-information or inconsistent samples, including LLM-assisted consistency checks.
3. **Reasoning Path Construction** – creating structured step-by-step diagnostic explanations (symptom evidence → psychosocial context → diagnosis).

*Data processing scripts are provided in [data/](./data) folder.*

---

## Supervised Fine-Tuning (SFT)

The **SFT stage** uses the **llamafactory** framework with the default configuration, adapted to our dataset. Key points:

- Inputs: structured reasoning data generated from UK Biobank and public psychology datasets.
- Outputs: model predictions + reasoning paths.
- Objective: optimize the LLM to follow structured reasoning instructions while maintaining diagnostic accuracy.

SFT config files and training scripts can be found in [sft/](./sft) folder.

---

## Reinforcement Learning (RL) Stage

The **RL stage** employs **veRL** framework to further refine reasoning ability:

- Uses the pre-trained SFT model as the starting policy.
- Reward function designed to:
  - Encourage accurate diagnostic predictions.
  - Ensure structured and coherent reasoning paths.
  - Penalize contradictory or illogical outputs.
- The model generates multiple candidate reasoning paths, which are evaluated and used to update the policy.

RL config files and training scripts are available in [rl/](./rl) folder.

---

## Repository Structure
