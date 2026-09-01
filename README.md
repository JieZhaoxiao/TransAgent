# TransAgent

Official implementation of **TransAgent: A Plug-and-Play Agent for Finding Optimal Input Transformations in Adversarial Attacks**.

## Environment

```bash
conda env create -f environment.yml
conda activate transagent
```

Model weights are downloaded automatically by TorchVision and timm.

## Data

`data/clean` contains 1,000 ImageNet-compatible clean images, and `data/labels.csv` contains their labels. `data/adversarial/mi_resnet50` contains one TransAgent run generated with MI and a ResNet-50 surrogate. Images are numbered from 1 to 1000. ImageNet images remain subject to their original terms.

## Generate Adversarial Examples

The paper uses Qwen3.7-Plus through Alibaba Cloud Model Studio.

Linux and macOS

```bash
export DASHSCOPE_API_KEY=<your-key>
export DASHSCOPE_BASE_URL=<openai-compatible-endpoint>
python generate.py --attack mi --surrogate resnet50 --seed 0
```

Windows PowerShell

```powershell
$env:DASHSCOPE_API_KEY = "<your-key>"
$env:DASHSCOPE_BASE_URL = "<openai-compatible-endpoint>"
python generate.py --attack mi --surrogate resnet50 --seed 0
```

Attack implementations and identifiers follow the open-source [TransferAttack](https://github.com/Trustworthy-AI-Group/TransferAttack) registry. Checkpoint-based methods require their official weights.

## Evaluate Adversarial Examples

The following command evaluates the included run on the seven black-box target models used in Table 1.

```bash
python evaluate.py
```
