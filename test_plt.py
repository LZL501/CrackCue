import torch
import torch.nn.functional as F
from data_loader import MVTecDRAEMTestDataset
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from model_unet import ReconstructiveSubNetwork, DiscriminativeSubNetwork,ReconstructiveSubNetwork_back
import os
from data.dataset import readIndex, dataReadPip, loadedDataset
import cv2
import torchvision
from torch import nn
from data.augmentation import augCompose,Resize
from deep_crack import DeepCrack
from matplotlib import pyplot as plt
os.environ["CUDA_VISIBLE_DEVICES"] = '4,3'

def similarity(m1, m2):
    return 1 - torch.cosine_similarity(m1, m2, dim=1).unsqueeze(1)


def test(checkpoint_path, base_model_name):

    run_name = base_model_name+"_"+"Crack"+'_'

    model = ReconstructiveSubNetwork_back(in_channels=3, out_channels=3)
    model = torch.nn.DataParallel(model, device_ids=(0,1))
    model.load_state_dict(torch.load(os.path.join(checkpoint_path,run_name+".pckl"), map_location='cuda:0'))
    model.cuda()
    model.eval()

    model_seg = DeepCrack()
    model_seg = torch.nn.DataParallel(model_seg, device_ids=(0,1))
    model_seg.load_state_dict(torch.load(os.path.join(checkpoint_path, run_name+"_seg.pckl"), map_location='cuda:0'))
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
    mae_for_sum = 0
    mae_back_sum = 0
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

            out_mask,_,_,_,_,_ = model_seg(joined_in)

            out_mask_sm = torch.sigmoid(out_mask)

            iter = iter + 1

            compare = torch.cat((true_mask, diffrence), dim = -1)
            
            diffrence_l1 = torch.abs((gray_rec - gray_batch))
            diffrence_l1 = (diffrence_l1 - diffrence_l1.min())/(diffrence_l1.max()-diffrence_l1.min())
            diffrence_l1 = torch.mean(diffrence_l1, dim=1, keepdim= True)
            # compare_l1 = torch.cat((true_mask,diffrence_l1), dim=-1)

            final = torch.cat((out_mask_sm, true_mask), dim= -1)
            final_img = np.array(gray_rec.cpu().squeeze().detach()*255).astype(np.uint8)
            final_img = np.transpose(final_img,(1,2,0))
            plt.axis('off')  # 去坐标轴
            plt.xticks([])
            plt.yticks([])
            plt.imshow(final_img)
            plt.savefig(os.path.join("./recon_img", os.path.basename(filename).replace('jpg','png')), bbox_inches='tight', pad_inches=-0.1, dpi=300)  # 注意两个参数
            torchvision.utils.save_image(compare, os.path.join("./compare_cos", os.path.basename(filename)))
            torchvision.utils.save_image(final, os.path.join("./recon", os.path.basename(filename)))
            # torchvision.utils.save_image(final_img, os.path.join("./recon_img", os.path.basename(filename).replace('jpg','png')),dpi=(600, 600))
            torchvision.utils.save_image(diffrence_l1, os.path.join("./compare_l1", os.path.basename(filename)))
            # torchvision.utils.save_image(compare_mask, os.path.join("./compare_mask", os.path.basename(filename)))

            if stone_flag:
                mask_pred = out_mask_sm * for_mask_t
                mask_pred = mask_pred.cpu().squeeze()
            else:
                mask_pred = out_mask_sm.cpu().squeeze()
            # print(mask_pred.size())
            # print(mask_pred.max())
            # print(mask_pred.min())
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
