import torch
import torch.nn as nn
from torchvision import transforms
from model import ParNet

import matplotlib.pyplot as plt

def viz(module, input):
    x = input[0][0]
    min_num = np.minimum(10, x.size()[0])
    for i in range(min_num):
        plt.subplot(1, 10, i+1)
        plt.imshow(x[i].cpu())
    plt.show()


import cv2
import numpy as np
def main():
    t = transforms.Compose([transforms.ToPILImage(),
                            transforms.Resize((224, 224)),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                 std=[0.229, 0.224, 0.225])
                            ])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = ParNet().to(device)
    for name, m in model.named_modules():
        # if not isinstance(m, torch.nn.ModuleList) and \
        #         not isinstance(m, torch.nn.Sequential) and \
        #         type(m) in torch.nn.__dict__.values():
        if isinstance(m, torch.nn.Conv2d):
            m.register_forward_pre_hook(viz)
    img = cv2.imread(r'')
    img = t(img).unsqueeze(0).to(device)
    with torch.no_grad():
        model(img)

if __name__ == '__main__':
    main()