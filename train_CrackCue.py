import torch
from torch.utils.data import DataLoader
from torch import optim
from tensorboard_visualizer import TensorboardVisualizer
from model_unet import  DiscriminativeSubNetwork,Unet
from loss import FocalLoss, SSIM
import os
from torch import nn
from data.augmentation import augCompose, RandomBlur, RandomColorJitter, Resize,RandomFlip
from data.dataset_back import readIndex, dataReadPip, loadedDataset
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

def similarity(m1, m2):
    return 1 - torch.cosine_similarity(m1, m2, dim=1).unsqueeze(1)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def mean_flat(tensor):
    return torch.mean(tensor, dim=list(range(0, len(tensor.shape))))

def MSE_loss(img, img_rec, mask):
    return mean_flat((1-mask) * (img - img_rec).square())

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def train_on_device(args):

    if not os.path.exists(args.checkpoint_path):
        os.makedirs(args.checkpoint_path)

    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)


    run_name = 'Crack_test_'+str(args.lr)+'_'+str(args.epochs)+'_bs'+str(args.bs)
    visualizer = TensorboardVisualizer(log_dir=os.path.join(args.log_path, run_name+"/"))

    model = Unet(in_channels=3, out_channels=3)
    model.cuda()
    model.apply(weights_init)
    
    model_seg = DiscriminativeSubNetwork(in_channels=4, out_channels=1)
    model_seg.cuda()
    model_seg.apply(weights_init)

    optimizer = torch.optim.Adam([
                                    {"params": model.parameters(), "lr": args.lr},
                                    {"params": model_seg.parameters(), "lr": args.lr}])

    scheduler = optim.lr_scheduler.MultiStepLR(optimizer,[args.epochs*0.8,args.epochs*0.9],gamma=0.2, last_epoch=-1)

    mask_loss = nn.BCEWithLogitsLoss(reduction='mean',
                                              pos_weight=torch.cuda.FloatTensor([1]))
    
    data_augment_op = augCompose(transforms=[[RandomColorJitter, 0.5], [RandomBlur, 0.2],[RandomFlip,0.6],[Resize, 1]])

    train_pipline = dataReadPip(transforms=data_augment_op)


    train_dataset = loadedDataset(readIndex("data/train_example.txt", shuffle=True), preprocess=train_pipline)

    dataloader = DataLoader(train_dataset, batch_size=args.bs,
                            shuffle=True, num_workers=4)

    n_iter = 0
    for epoch in range(args.epochs):
        recon_losses = []
        seg_losses = []
        for i_batch, (gray_batch, anomaly_mask, mask_dila)in enumerate(dataloader):
            anomaly_mask = anomaly_mask.unsqueeze(1)
            mask_dila = mask_dila.unsqueeze(1)
            gray_batch = gray_batch.cuda()
            anomaly_mask = anomaly_mask.cuda()
            mask_dila = mask_dila.cuda()

            gray_rec, tensor_recon,feature2_r, feature3_r, feature4_r = model(gray_batch)

            diffrence = torch.mean((gray_rec - gray_batch).abs(), dim = 1 ,keepdim= True)
            diffrence = (diffrence - diffrence.min())/(diffrence.max()-diffrence.min())
            joined_in = torch.cat((diffrence, gray_batch), dim=1)

            out_mask = model_seg(joined_in)


            out_mask_sm = out_mask
            
            l2_loss = MSE_loss(gray_rec,gray_batch, mask_dila)

            segment_loss = mask_loss(out_mask_sm.view(-1, 1), anomaly_mask.view(-1, 1))

            loss = l2_loss + segment_loss
            seg_losses.append(segment_loss.detach().cpu().numpy())
            recon_losses.append((l2_loss ).detach().cpu().numpy())

            optimizer.zero_grad()

            loss.backward()
            optimizer.step()

            if args.visualize and n_iter % 200 == 0:
                visualizer.plot_loss(l2_loss, n_iter, loss_name='l2_loss')
                visualizer.plot_loss(segment_loss, n_iter, loss_name='segment_loss')
            if args.visualize and n_iter % 400 == 0:
                t_mask = out_mask_sm[:, 1:, :, :]
                visualizer.visualize_image_batch(gray_batch, n_iter, image_name='batch_recon_target')
                visualizer.visualize_image_batch(gray_rec, n_iter, image_name='batch_recon_out')
                visualizer.visualize_image_batch(anomaly_mask, n_iter, image_name='mask_target')
                visualizer.visualize_image_batch(t_mask, n_iter, image_name='mask_out')


            n_iter +=1

        scheduler.step()
        print("Epoch: "+str(epoch)+ "--seg_loss:" + str(np.mean(seg_losses))+ "--recon_loss:" + str(np.mean(recon_losses)))
        torch.save(model.state_dict(), os.path.join(args.checkpoint_path, run_name+".pckl"))
        torch.save(model_seg.state_dict(), os.path.join(args.checkpoint_path, run_name+"_seg.pckl"))


if __name__=="__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--bs', action='store', type=int, required=True)
    parser.add_argument('--lr', action='store', type=float, required=True)
    parser.add_argument('--epochs', action='store', type=int, required=True)
    parser.add_argument('--checkpoint_path', action='store', type=str, required=True)
    parser.add_argument('--log_path', action='store', type=str, required=True)
    parser.add_argument('--visualize', action='store_true')

    args = parser.parse_args()


    train_on_device(args)

