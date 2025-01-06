# TaKSIE
WACV 2025

The code for the paper:

**Incorporating Task Progress Knowledge for Subgoal Generation in Robotic Manipulation through Image Edits**

[[Project Page]](https://live-robotics-uva.github.io/TaKSIE/) [[Arxiv]](https://arxiv.org/abs/2410.11013)


## Quick Start

### 1. Create the Environment

```bash
conda create -n taksie python=3.9
conda activate taksie
pip install -r requirements.txt
```

### 2. Test the Inference

Run the example inference script to test the subgoal generation:

```bash
python example_inference.py
```

## Evaluate the Trained Checkpoint on CALVIN

### 1. Install Required Modules

#### (a) CALVIN

Follow the instructions [here](https://github.com/mees/calvin) to install the CALVIN environment.

#### (b) Goal-Conditioned Policy

Install JAX and `jaxrl_m` for the goal-conditioned policy:

```bash
# Follow the instructions here to install jax and jaxrl_m:
https://github.com/rail-berkeley/bridge_data_v2
```

Download the DGBC checkpoint from the provided link. Update the `cfg/cfg.yaml` file by replacing the parent directory path with the path to the downloaded checkpoint.

#### (c) LIV for Progress Evaluator

Follow the instructions [here](https://github.com/penn-pal-lab/LIV) to install the LIV framework. Additionally, download the fine-tuned LIV checkpoint for the CALVIN dataset:

- Download the checkpoint: [Hugging Face](https://huggingface.co/ShuaKang/taksie/tree/main/liv_resnet50_CALVIN)
- Replace the default checkpoint in `~/.liv/resnet50` with the downloaded fine-tuned checkpoint.

### 2. Run the Evaluation

Run the evaluation script with the appropriate configuration file:

```bash
python evaluate_calvin.py --running_config=cfg/cfg.yaml
```

## Checkpoints

- GCBC checkpoint: [Link](https://huggingface.co/ShuaKang/taksie/tree/main/gcbc_checkpoint)
- LIV fine-tuned checkpoint: [Link](https://huggingface.co/ShuaKang/taksie/tree/main/liv_resnet50_CALVIN)
- ControlNet: [Link](https://huggingface.co/ShuaKang/TaKSIE_controlnet)
- Unet: [Link](https://huggingface.co/ShuaKang/TaKSIE_unet)
## Bibtex

```bibtex
@misc{kang2024incorporatingtaskprogressknowledge,
      title={Incorporating Task Progress Knowledge for Subgoal Generation in Robotic Manipulation through Image Edits}, 
      author={Xuhui Kang and Yen-Ling Kuo},
      year={2024},
      eprint={2410.11013},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2410.11013}, 
}
```

## Acknowledgements
Thank you for these excellent works!
- CALVIN: [https://github.com/mees/calvin](https://github.com/mees/calvin)
- LIV: [https://github.com/penn-pal-lab/LIV](https://github.com/penn-pal-lab/LIV)
- GCBC Policy: [https://github.com/rail-berkeley/bridge_data_v2](https://github.com/rail-berkeley/bridge_data_v2)
- SuSIE: [https://github.com/kvablack/susie](https://github.com/kvablack/susie)
