# CrackCue

The new version is now online, with the GitHub link: [https://hal.science/hal-05169427](https://hal.science/hal-05169427)

PyTorch implementation of "CrackCue" (to appear in Pattern Recognition, 2025).  
[Paper link](https://papers.ssrn.com/sol3/Delivery.cfm?abstractid=4957739)

BibTeX:
@article{liu2025coarse,
  title={Coarse-to-Fine Crack Cue for Robust Crack Detection},
  author={Liu, Zelong and Gu, Yuliang and Sun, Zhichao and Zhu, Huachao and Xiao, Xin and Du, Bo and Najman, Laurent and Xu, Yongchao},
  journal={Pattern Recognition},
  pages={112107},
  year={2025},
  publisher={Elsevier}
}

## Datasets

You can download the datasets from the following links:

- CrackTree260 dataset:
  https://1drv.ms/f/s!AittnGm6vRKLyiQUk3ViLu8L9Wzb

- CRKWH100 dataset:
  https://1drv.ms/f/s!AittnGm6vRKLtylBkxVXw5arGn6R

- CRKWH100 Ground Truth:
  https://1drv.ms/f/s!AittnGm6vRKLglyfiCw_C6BDeFsP

- CrackLS315 dataset:
  https://1drv.ms/f/s!AittnGm6vRKLtylBkxVXw5arGn6R

- CrackLS315 Ground Truth:
  https://1drv.ms/u/s!AittnGm6vRKLg0HrFfJNhP2Ne1L5?e=WYbPvF

- Stone331 dataset:
  https://1drv.ms/f/s!AittnGm6vRKLtylBkxVXw5arGn6R

- Stone331 Ground Truth:
  https://1drv.ms/f/s!AittnGm6vRKLwiL55f7f0xdpuD9_

- Stone331 Mask:
  https://1drv.ms/u/s!AittnGm6vRKLxmFB78iKSxTzNLRV?e=9Ph5aP

After downloading, replace `img_path` in the `path.py` script with your local path and run:

    python path.py

## Checkpoints

You can download our trained model checkpoint from the following link:

- CrackCue checkpoint:  
  https://drive.google.com/file/d/1G2SC1ez8h0Uf1KK22dlzuIDIssPiOrco/view?usp=sharing

After downloading, place the checkpoint file under the `./checkpoints` directory or modify the path in the `--checkpoint_path` argument accordingly.

## Training

To train the model, run the following command with desired arguments:

    python train_CrackCue.py --bs 4 --lr 0.0001 --epochs 700 --checkpoint_path ./checkpoints --log_path ./logs

Where:
- `--bs`: batch size
- `--lr`: learning rate
- `--epochs`: number of training epochs
- `--checkpoint_path`: path to save model checkpoints
- `--log_path`: path to save training logs

## Evaluating

To evaluate a trained model, run:

    python test_CrackCue.py --base_model_name Crack_test_0.0001_700_bs4 --checkpoint_path ./checkpoints

Where:
- `--base_model_name`: name of the checkpoint
- `--checkpoint_path`: path where the checkpoint is stored
