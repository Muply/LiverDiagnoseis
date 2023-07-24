import torch
import torch.nn as nn
from torchvision import transforms
import torch.optim as optim
from mymodel_newdata import myModel
from dataset_newdata_predict import MyDataset
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import os
from torchcam.methods import SmoothGradCAMpp
import matplotlib.pyplot as plt
from torchcam.utils import overlay_mask
from torchvision.io.image import read_image
from torchvision.transforms.functional import normalize, resize, to_pil_image


device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
model_weight_path = './train_history/model_doubleConv_newdata_train.pth'
valid_txt_path = './train_cfgs/test_1.txt'
# create model
model = myModel()
model.load_state_dict(torch.load(model_weight_path, map_location=device))
model.eval()
cam_extractor = SmoothGradCAMpp(model,target_layer='satt1', input_shape=(24,224,224))

data_transform = transforms.Compose(
    [transforms.Resize(256),
     transforms.CenterCrop(224),
     transforms.ToTensor(),
     transforms.Normalize([0.43304095, 0.29830533, 0.18766068], [0.2689185, 0.22025496, 0.15420981])])
valid_data = MyDataset(txt_path=valid_txt_path, transform=data_transform)
# fn = './dataset/301_CEUS_7/test/474415'   # fn 是一个文件夹的地址
# all_img = []
# for i in range(1,9):
#     all_img.append(fn + '/' + str(i) + '.png')
# # img_dir = os.listdir(fn)
# # for _ in img_dir:
# #     all_img.append(fn + '/' + _)
# im1 = Image.open(all_img[0]).convert('RGB')
# im1 = data_transform(im1)
# im2 = Image.open(all_img[1]).convert('RGB')
# im2 = data_transform(im2)
# im3 = Image.open(all_img[2]).convert('RGB')
# im3 = data_transform(im3)
# im4 = Image.open(all_img[3]).convert('RGB')
# im4 = data_transform(im4)
# im5 = Image.open(all_img[4]).convert('RGB')
# im5 = data_transform(im5)
# im6 = Image.open(all_img[5]).convert('RGB')
# im6 = data_transform(im6)
# im7 = Image.open(all_img[6]).convert('RGB')
# im7 = data_transform(im7)
# im8 = Image.open(all_img[7]).convert('RGB')
# im8 = data_transform(im8)
# img = torch.cat((im1,im2,im3,im4,im5,im6,im7,im8),0)
for img, lable, fn in valid_data:

    out = model(img)
    # Retrieve the CAM by passing the class index and the model output
    activation_map = cam_extractor(out.squeeze(0).argmax().item(), out)

    # plt.imshow(activation_map[0].squeeze(0).numpy()); plt.axis('off'); plt.tight_layout(); plt.show()
    # img_selected = fn + '/' + '1.png'
    print(fn)
    words = fn.split('/')
    case_name = words[-1]
    save_path = './result_hot/satt1/' + case_name
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for i in range(4,6):
        pic_name = fn + '/' + str(i) +'.png'

        img_selected = read_image(pic_name)
        # Resize the CAM and overlay it
        result = overlay_mask(to_pil_image(img_selected), to_pil_image(activation_map[0].squeeze(0), mode='F'), alpha=0.5)
        # Display it
        # plt.imshow(result); plt.axis('off'); plt.tight_layout(); plt.show()
        plt.imshow(result); plt.axis('off'); plt.tight_layout()
        # plt.savefig(save_path + str(i) + '.png', dpi=100)
        plt.savefig(os.path.join(save_path,str(i) + '.png'), dpi=100)
