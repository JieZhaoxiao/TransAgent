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

The paper uses Qwen3.7-Plus through Alibaba Cloud Model Studio. Keep the API key in environment variables and never commit it to source code or configuration files.

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

Available attacks are `mi`, `pgn`, `mumodig`, `gaa`, and `foolmix`. Available surrogates are `resnet50` and `vit`. The paper configuration uses an $L_\infty$ budget of 16/255, a step size of 1.6/255, 10 attack steps, batch size 5, replanning interval 5, 7 retrieved memory records, and 2 test views per candidate program. Runtime is recorded under `runs/`.

When the API is unavailable, the documented local fallback keeps the code executable. This API-free mode is not the main paper setting.

The implementations of the configured attacks follow the open-source [TransferAttack](https://github.com/Trustworthy-AI-Group/TransferAttack) project.

## Evaluate Adversarial Examples

The following command evaluates the included run on the seven black-box target models used in Table 1. It prints ASR and runtime without writing result files.

```bash
python evaluate.py
```

## Citation

```bibtex
@article{zhang2026transagent,
  title={TransAgent: A Plug-and-Play Agent for Finding Optimal Input Transformations in Adversarial Attacks},
  author={Zhang, Yu and Yang, Xing and Zhao, Shijie and Deng, Kang and Peng, Anjie and Zeng, Hui},
  year={2026}
}
```
