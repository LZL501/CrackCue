# CrackCue
PyTorch implementation of [CrackCue](https://papers.ssrn.com/sol3/Delivery.cfm?abstractid=4957739) - PR 2025:

```
@article{liu2025coarse,
  title={Coarse-to-Fine Crack Cue for Robust Crack Detection},
  author={Liu, Zelong and Gu, Yuliang and Sun, Zhichao and Zhu, Huachao and Xiao, Xin and Du, Bo and Najman, Laurent and Xu, Yongchao},
  journal={Pattern Recognition},
  pages={112107},
  year={2025},
  publisher={Elsevier}
}
```
## Datasets
```
You can download the datasets from follow links:
CrackTree260 dataset: https://1drv.ms/f/s!AittnGm6vRKLyiQUk3ViLu8L9Wzb
CRKWH100 dataset: https://1drv.ms/f/s!AittnGm6vRKLtylBkxVXw5arGn6R
CRKWH100 GT: https://1drv.ms/f/s!AittnGm6vRKLglyfiCw_C6BDeFsP

CrackLS315 dataset: https://1drv.ms/f/s!AittnGm6vRKLtylBkxVXw5arGn6R 
CrackLS315 GT: https://1drv.ms/u/s!AittnGm6vRKLg0HrFfJNhP2Ne1L5?e=WYbPvF

Stone331 dataset: https://1drv.ms/f/s!AittnGm6vRKLtylBkxVXw5arGn6R 
Stone331 GT: https://1drv.ms/f/s!AittnGm6vRKLwiL55f7f0xdpuD9_
Stone331 Mask: https://1drv.ms/u/s!AittnGm6vRKLxmFB78iKSxTzNLRV?e=9Ph5aP

Then replace the img_path in the following function with your local path and run it:
```
python path.py


## Training
```
After passing batch size (-- bs), learning rate (-- lr), epoch (-- epochs), path for storing checkpoints (-- checkpoint_path), and log storage path (-- log_path) as parameters into the script, run the following command
```
python train_CrackCue.py --bs 4 --lr 0.0001 --epochs 700 --checkpoint_path ./checkpoints --log_path ./logs


## Evaluating
```
After passing base model name (-- base_model_name), checkpoint path (-- checkpoint_path) as parameters into the script, run the following command
```
python test_CrackCue.py --base_model_name Crack_test_0.0001_700_bs4 --checkpoint_path ./checkpoints

