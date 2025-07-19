import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from model_unet import DiscriminativeSubNetwork,Unet
import os
from data.dataset_back import readIndex, dataReadPip, loadedDataset
import cv2
import torchvision
from torch import nn
from data.augmentation import augCompose,Resize


os.environ["CUDA_VISIBLE_DEVICES"] = '6'

def similarity(m1, m2):
    return 1 - torch.cosine_similarity(m1, m2, dim=1).unsqueeze(1)


def test(checkpoint_path, base_model_name):

    model = Unet(in_channels=3, out_channels=3)
    model.load_state_dict(torch.load(os.path.join(checkpoint_path,base_model_name+".pckl"), map_location='cuda:0'))
    model.cuda()
    model.eval()

    model_seg = DiscriminativeSubNetwork(in_channels=4, out_channels=1)
    model_seg.load_state_dict(torch.load(os.path.join(checkpoint_path, base_model_name+"_seg.pckl"), map_location='cuda:0'))
    model_seg.cuda()
    model_seg.eval()
    
    stone_flag = 0
    if stone_flag:
        data_augment_op = augCompose(transforms=[[Resize, 1]])

        test_pipline = dataReadPip(transforms=data_augment_op)
    else:
        data_augment_op = augCompose(transforms=[[Resize, 1]])
        test_pipline = dataReadPip(transforms=data_augment_op)
    stone_mask = "/data/arima/Stone331/Stone331_mask"
    test_list = readIndex("data/train_example.txt")
    test_dataset = loadedDataset(readIndex("data/train_example.txt"), preprocess=test_pipline)

    dataloader = DataLoader(test_dataset, batch_size=1,
                            shuffle=False, num_workers=0)

    os.makedirs("./recon", exist_ok=True)
    os.makedirs("./recon_img", exist_ok=True)
    os.makedirs("./pred2", exist_ok=True)
    os.makedirs("./compare_cos", exist_ok= True)
    os.makedirs("./compare_l1", exist_ok= True)
    iter = 0


    with torch.no_grad():
        
        for names, (gray_batch, true_mask, mask_dila) in zip(test_list, dataloader):
            
            filename = names[0]
            gray_batch = gray_batch.cuda()
            true_mask =  true_mask.unsqueeze(1)
            true_mask_bina = (mask_dila>0).float()
            true_mask =  true_mask.cuda()
            true_mask_bina = true_mask_bina.cuda()
            if stone_flag:
                for_mask = cv2.imread(os.path.join(stone_mask, os.path.basename(filename)[:-4]+".bmp"))
                for_mask = np.mean(for_mask, axis= -1)
                for_mask = for_mask/255
                for_mask_t = torch.from_numpy(for_mask).float().cuda()
            
            gray_rec,tensor_recon,feature2, feature3, feature4 = model(gray_batch)
            diffrence = torch.mean((gray_rec - gray_batch).abs(), dim = 1 ,keepdim= True)


            diffrence = (diffrence - diffrence.min())/(diffrence.max()-diffrence.min())
            joined_in = torch.cat((diffrence.detach(), gray_batch), dim=1)

            out_mask = model_seg(joined_in)

            out_mask_sm = torch.sigmoid(out_mask)

            iter = iter + 1

            if stone_flag:
                mask_pred = out_mask_sm * for_mask_t
                mask_pred = mask_pred.cpu().squeeze()
            else:
                mask_pred = out_mask_sm.cpu().squeeze()

            mask_pred = mask_pred.detach().numpy() * 255
            if mask_pred.max()< 1:
                print(filename)
                mask_pred[0][0] = 255
                
            save_name = os.path.join("./pred2", os.path.basename(filename).replace("jpg", "png"))
            mask_pred = cv2.resize(mask_pred, (512, 512))
            cv2.imwrite(save_name, mask_pred.astype(np.uint8))

if __name__=="__main__":
    import argparse

    parser = argparse.ArgumentParser()
    # parser.add_argument('--gpu_id', action='store', type=int, required=True)
    parser.add_argument('--base_model_name', action='store', type=str, required=True)

    parser.add_argument('--checkpoint_path', action='store', type=str, required=True)

    args = parser.parse_args()

    test(args.checkpoint_path, args.base_model_name)
