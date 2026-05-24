## TAB: Task-adaptive Mixture of Experts with Attentional Bias for Graph Learning in LLMs

This repository provides the official implementation and cleaned code for the paper:

**TAB: Task-adaptive Mixture of Experts with Attentional Bias for Graph Learning in LLMs**

The code focuses on:
- Fine-tuning several large language models (LLMs) on graph reasoning tasks with **task-adaptive attentional bias**.
- A unified **graph sequence encoder** that turns graphs into text sequences and attention-bias tensors.
- **Task-specific evaluation** for a diverse set of graph problems.

All experimental code has been organized and stripped of debugging prints and comments to facilitate open-sourcing.

### Installation

The `./environment.yml` lists all Python libraries that TAB depends on, and you can install using:

```bash
conda env create -f environment.yml
conda activate TAB
```

If you change the environment name in `environment.yml`, use that name instead of `TAB`.

---

## Datasets & Models

All training and testing data are placed under `tasks/`. Each task directory contains:

- `train.json`
- `test.json`

TAB is evaluated on four backbone LLMs. You can download the model from Hugging Face and place it in the local path. If there is no such path locally, the code will automatically download it from Hugging Face.

- **Llama-2-7B**
  - Local Path: `finetuning/Llama-2-7b-hf`
  - Hugging Face: `meta-llama/Llama-2-7b-hf`

- **GPT-OSS-20B**
  - Local Path: `finetuning/gpt-oss-20b-hf`
  - Hugging Face: `openai/gpt-oss-20b`
  
- **Phi-4-mini**
  - Local Path: `finetuning/Phi-4-mini-instruct-hf`
  - Hugging Face: `microsoft/Phi-4-mini-instruct`
  
- **Qwen3-8B**
  - Local Path: `finetuning/Qwen3-8B-hf`
  - Hugging Face: `Qwen/Qwen3-8B`

---

## Fine-tuning 

All finetuning scripts are in `finetuning/` and implement LoRA fine-tuning. You can run the following instructions for fine-tuning. All scripts exclusively use `train.json` in each `tasks/<task>/` directory.

**Llama-2-7B**

```bash
cd finetuning
python simple_llama2_7b_finetuning.py
```

The result will be saved to `output/llama2_7b_finetuned`.

**GPT-OSS-20B**

```bash
cd finetuning
python simple_gpt_oss_20b_finetuning.py
```

The result will be saved to `output/gpt_oss_20b_finetuned`.

**Phi-4-mini**

```bash
cd finetuning
python simple_phi4_mini_finetuning.py
```

The result will be saved to `output/phi4_mini_finetuned`.

**Qwen3-8B**

```bash
cd finetuning
python simple_qwen3_8b_finetuning.py
```

The result will be saved to  `output/qwen3_8b_finetuned`.

---

## Testing

After fine-tuning, you can test model performance using the `testing/` scripts. You can run the testing scripts as follows. After the test is completed, the accuracy rate of each graph task will be printed out.

**Llama-2-7B**

```bash
cd testing
python test_llama2_7b_finetuned.py
```

Detailed task-wise outputs will be saved to `tasks/<task>/test_results_llama2_7b_*.json`.

**GPT-OSS-20B**

```bash
cd testing
python test_gpt_oss_20b_finetuned.py
```

Detailed task-wise outputs will be saved to `tasks/<task>/test_results_gpt_oss_20b_*.json`.

**Phi-4-mini**

```bash
cd testing
python test_phi4_mini_finetuned.py
```

Detailed task-wise outputs will be saved to `tasks/<task>/test_results_phi4_mini_*.json`.

**Qwen3-8B**

```bash
cd testing
python test_qwen3_8b_finetuned.py
```

Detailed task-wise outputs will be saved to `tasks/<task>/test_results_qwen3_8b_*.json`.
