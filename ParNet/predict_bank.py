
# 批量预测
import os
import json

import numpy as np
from sklearn import metrics
import torch
from PIL import Image
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
# from model_v2 import MobileNetV2
from model import ParNet as create_model

import os
import time

def main():
    time1=time.time()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    img_size = {"B0": 224,
                }
    num_model = "B0"

    data_transform = transforms.Compose(
        [transforms.Resize(img_size[num_model]),
         transforms.CenterCrop(img_size[num_model]),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    root_path=r''
    predict_dataset = datasets.ImageFolder(root=root_path,transform=data_transform)
    tea_list=predict_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in tea_list.items())

    ######################################################################################################
    trainpath = r''
    img_path_list = []
    total_class=[]
    for parent, dirnames, filenames in os.walk(trainpath):
        for filename in filenames:
            if filename.lower().endswith(
                    ('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                img_path_list.append(os.path.join(parent, filename))
                dir_train,file_train=os.path.split(os.path.join(parent, filename))
                file_name,ext=os.path.splitext(file_train)
                total_class.append(file_name[:3])

    a=np.zeros(shape=(1,len(total_class)),dtype=np.int64)
    a = a.flatten()
    for step, item in enumerate(total_class):
        if item.strip() == "Nic":
            a[step] = 0
        elif item.strip() == "nCT":
            a[step] = 1
        elif item.strip() == "pCT":
            a[step] = 2

    y_true=torch.from_numpy(a)


    img_list = []
    for img_path in img_path_list:
        assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
        img = Image.open(img_path).convert("RGB")
        img = data_transform(img)
        img_list.append(img)
    ############################################################################################

    # batch img
    batch_img = torch.stack(img_list, dim=0)

    # read class_indict
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    json_file = open(json_path, "r")
    class_indict = json.load(json_file)

    model = create_model(num_classes=3).to(device)

    # load model weights
    weights_path = r""
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    model.load_state_dict(torch.load(weights_path, map_location=device))


    # prediction
    model.eval()
    with torch.no_grad():
        # predict class
        output = model(batch_img.to(device)).cpu()
        predict = torch.softmax(output, dim=1)
        probs, y_pred = torch.max(predict, dim=1)

        con_mat=metrics.confusion_matrix(y_true,y_pred)
        print(con_mat)

        for idx, (pro, cla) in enumerate(zip(probs, y_pred)):
            print("image: {}  class: {}  prob: {:.3}".format(img_path_list[idx],
                                                             class_indict[str(cla.numpy())],
                                                             pro.numpy()))


    time2 = time.time()
    print("time:" + str(time2 - time1))

if __name__ == '__main__':
    main()
