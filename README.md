# MDD-Thinker

**MDD-Thinker** is a large language model (LLM)-based diagnostic framework designed for **major depressive disorder (MDD)**. The system integrates **supervised fine-tuning (SFT)** and **reinforcement learning (RL)** to enhance both **diagnostic accuracy** and **interpretability**, allowing the model to generate structured reasoning paths for clinical decision support.

---

## Overview

MDD-Thinker aims to provide a scalable and explainable solution for intelligent psychiatric assessment. Its architecture combines domain-specific reasoning with LLM capabilities to produce:

- Accurate MDD diagnosis
- Structured and clinically coherent reasoning paths
- Integration of multimodal clinical and psychological knowledge (text-based)

### Core Architecture
<img src="imgs/arch.png" width="60%">  

*Figure: Core workflow of MDD-Thinker including data processing, SFT training, and RL fine-tuning.*

---

## Main Results
We conduct extensive experiments on UKBiobank cohort. The model performance was evaluated using the *Accuracy (Acc)*, *F1-Score (F1)*, *Area Under the Curve (AUC)*, *Specificity (SPE)*, *Sensitivity (SNE)*, *Positive Predictive Value (PPV)*, and *Negative Predictive Value (NPV)*. The results are shown as follows.

| Method             | ACC    | F1     | AUC    | SPE    | SENS   | PPV    | NPV    |
|-------------------|--------|--------|--------|--------|--------|--------|--------|
| SVM               | 0.6794 | 0.6517 | 0.7463 | 0.6958 | 0.6596 | 0.6438 | 0.7106 |
| RF                | 0.6883 | 0.6341 | 0.7369 | 0.7662 | 0.5947 | 0.6791 | 0.6943 |
| LightGBM          | 0.7091 | 0.6707 | 0.7693 | 0.7572 | 0.6516 | 0.6908 | 0.7231 |
| XGBoost           | 0.7068 | 0.6704 | 0.7497 | 0.7501 | 0.6554 | 0.6858 | 0.7233 |
| CatBoost          | 0.7117 | 0.6717 | 0.7751 | 0.7642 | 0.6486 | 0.6961 | 0.7232 |
| MLP               | 0.6869 | 0.6301 | 0.7522 | 0.7692 | 0.5877 | 0.6793 | 0.6916 |
| ResNet1D          | 0.7077 | 0.6644 | 0.7654 | 0.7669 | 0.6369 | 0.6945 | 0.7173 |
| LLaMA3.1-8B       | 0.6167 | 0.5229 | 0.6387 | 0.7452 | 0.4625 | 0.6013 | 0.6249 |
| Qwen2.5-7B        | 0.6409 | 0.5852 | 0.6532 | 0.7593 | 0.5411 | 0.6182 | 0.6483 |
| MDD-LLM 8B        | 0.7919 | 0.7642 | 0.8579 | 0.8039 | 0.7763 | 0.7524 | 0.8241 |
| MDD-Thinker 7B    | 0.8268 | 0.8081 | 0.8803 | 0.8229 | 0.8291 | 0.7838 | 0.8614 |

## Requirement
The following Python packages and frameworks are required to run MDD-Thinker. Please follow the installation and environment setup instructions provided by **LlamaFactory** and **veRL** for proper configuration.

| Framework / Package       | Version / Notes                                  | Purpose                                           |
|---------------------------|-------------------------------------------------|-------------------------------------------------|
| Python                    | >= 3.9                                          | Core programming language                        |
| PyTorch                   | >= 2.0                                          | Deep learning backend                             |
| Transformers              | >= 4.30                                         | LLM support for model loading and training       |
| LlamaFactory              | see [LlamaFactory docs](#)            | Framework for supervised fine-tuning (SFT)      |
| veRL                      | see [veRL docs](#)                   | Framework for reinforcement learning (RL)       |
| numpy                     | >= 1.23                                         | Numerical computations                           |
| pandas                    | >= 1.6                                          | Data manipulation                                |
| scikit-learn              | >= 1.2                                          | Evaluation metrics and preprocessing             |
| tqdm                      | >= 4.65                                         | Progress bars                                    |
| matplotlib / seaborn       | latest                                          | Visualization of results                          |
| datasets                  | >= 2.15                                         | Loading public NLP datasets                        |
| sentencepiece             | latest                                          | Tokenization support for LLMs                     |

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

1. **Feature Selection** – selecting 22 clinically and statistically relevant variables. (**feature_selection.py**)
2. **Data Filtering** – removing low-information or inconsistent samples, including LLM-assisted consistency checks. (**data_filter.py**)
3. **Reasoning Path Construction** – creating structured step-by-step diagnostic explanations (symptom evidence → psychosocial context → diagnosis). （**reasoning_generation.py**）

*Data processing scripts are provided in [data_process/](./data_process/) folder.*

---

## Supervised Fine-Tuning (SFT)

The **SFT stage** uses the **llamafactory** framework with the default configuration, adapted to our dataset. Key points:

- Inputs: structured reasoning data generated from UK Biobank and public psychology datasets.
- Outputs: model predictions + reasoning paths.
- Objective: optimize the LLM to follow structured reasoning instructions while maintaining diagnostic accuracy.

SFT config files and training scripts can be found in [sft/](./sft) folder.

#### SFT Training Steps
1. Install LLaMA-Factory
   ```
   git clone https://github.com/hiyouga/LLaMA-Factory.git
   cd LLaMA-Factory
   pip install -r requirements.txt
   ```
2. Prepare Model and Data
* Download the base model (e.g., Qwen2.5-7B) and note its local path.
* Prepare the SFT dataset and register it in data/dataset_info.json.

3. Place the YAML Configuration
* Copy the prepared YAML file (`sft/sft_full.yaml`) into `LLaMA-Factory/examples/train_full/`.

4. Model Training
* Multi-GPU training
   `FORCE_TORCHRUN=1 llamafactory-cli train examples/train_full/ukb_mental_full_sft.yaml`

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

#### RL Training Steps
1. Install veRL
   ```
   git clone https://github.com/volcengine/verl.git
   cd verl
   pip install -e .
   ```

2. Modify Reward Function
* move the `mdd_reward.py` to `verl/workers/reward_manager/` and update
```
reward_model:
  strategy: manual
  reward_manager: 'verl.workers.reward_manager.mdd_reward.MDDReasoningRewardManager'
```

3. Data Preparation
Convert your MDD dataset into `.parquet` format.
* Prompt: Include patient descriptions and formatting instructions (e.g., "Provide Reasoning and Diagnosis").
* Label: The ground-truth diagnosis.

4. Configuration Modification
* Modify some key parameters in `run_qwen2_5_7b_grpo.sh`, such as the location of the data.

5. Model Tranining

`bash run_qwen2_5_7b_grpo.sh`
