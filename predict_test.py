import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import cv2
from skimage.measure import compare_psnr


#define model
class SuperResolutionCNN(nn.Module):
    def __init__(self):
        super(SuperResolutionCNN, self).__init__()
        self.Conv1 = nn.Conv2d(1, 64, 9, 1, 4)
        self.Conv2 = nn.Conv2d(64, 32, 1, 1, 0)
        self.Conv3 = nn.Conv2d(32, 1, 5, 1, 2)
        self.Relu = nn.ReLU()
        self.sig = nn.Sigmoid()

    def forward(self, x):
        out = self.Relu(self.Conv1(x))
        out = self.Relu(self.Conv2(out))
        out = self.Conv3(out)
        out = x+out
        return out

def predict(filename,model):
    #输入和模型
    low_filename = 'C:/Users/ROG/Desktop/textimage/lowResolution/'+filename
    high_filename = 'C:/Users/ROG/Desktop/textimage/highResolution/'+filename
    img_low = cv2.imread(low_filename)
    img_low = cv2.cvtColor(img_low, cv2.COLOR_BGR2GRAY)
    img_high = cv2.imread(high_filename)
    img_high = cv2.cvtColor(img_high, cv2.COLOR_BGR2GRAY)
    img_tensor=torch.from_numpy(img_low).to(torch.float32).unsqueeze(0).unsqueeze(0)
    y_pred=model(img_tensor)

    y_pred=y_pred.detach().numpy()
    y_pred=y_pred.squeeze()
    y_pred= y_pred.astype(int)
    psnr_h_l = compare_psnr(img_high, img_low)
    psnr_h_p = compare_psnr(img_high, y_pred)
    plt.figure(figsize=(12,3))
    plt.subplot(1,3,1)
    plt.imshow(img_low,cmap='gray')
    plt.title('psnr:%.4f' % (psnr_h_l))
    plt.subplot(1,3,2)
    plt.title('psnr:%.4f' % (psnr_h_p))
    plt.imshow(y_pred,cmap='gray')
    plt.subplot(1, 3, 3)
    plt.imshow(img_high, cmap='gray')
    plt.show()


if __name__ == '__main__':
    model = SuperResolutionCNN()  # define model
    model.load_state_dict(torch.load(r'C:\Users\ROG\Desktop\textimage\srcnn_test\model.pth', map_location='cpu'))  # load model checkpoint
    predict('2.png', model)  # predict image
