from torchvision.models import resnet50
import torch
import torch.nn as nn
import torch.nn.functional as F


    
    
class MyVgg11(nn.Module):
    def __init__(self, class_num):
        super(MyVgg11, self).__init__()
        
        # first: CONV => RELU => CONV => RELU => POOL set
        self.conv1_1 = nn.Conv2d(3, 32, 3, padding = 1)
        self.norm1_1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.dropout1 = nn.Dropout2d(0.2)
        
        # second: CONV => RELU => CONV => RELU => POOL set
        self.conv2_1 = nn.Conv2d(32, 64, 3, padding = 1)
        self.norm2_1 = nn.BatchNorm2d(64)
    
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.dropout2 = nn.Dropout2d(0.2)

        # third: CONV => RELU => CONV => RELU => CONV => RELU => POOL set
        self.conv3_1 = nn.Conv2d(64, 128, 3, padding = 1)
        self.norm3_1 = nn.BatchNorm2d(128)
    
        self.conv3_2 = nn.Conv2d(128, 128, 3, padding = 1)
        self.norm3_2 = nn.BatchNorm2d(128)

        self.pool3 = nn.MaxPool2d(kernel_size=2)
        self.dropout3 = nn.Dropout2d(0.2)  

        # forth: CONV => RELU => CONV => RELU => CONV => RELU => POOL set
        self.conv4_1 = nn.Conv2d(128, 256, 3, padding = 1)
        self.norm4_1 = nn.BatchNorm2d(256)
        self.conv4_2 = nn.Conv2d(256, 256, 3, padding = 1)
        self.norm4_2 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(kernel_size=2)
        self.dropout4 = nn.Dropout2d(0.3)   

        # fifth: CONV => RELU => CONV => RELU => CONV => RELU => POOL set
        self.conv5_1 = nn.Conv2d(256, 512, 3, padding = 1)
        self.norm5_1 = nn.BatchNorm2d(512)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding = 1)
        self.norm5_2 = nn.BatchNorm2d(512)
        self.pool5 = nn.MaxPool2d(kernel_size=2)
        self.dropout5 = nn.Dropout2d(0.3)   

        # fully connected (single) to RELU
        self.fc1 = nn.Linear(512 * 7 * 7, 4096)
        self.normfc_1 = nn.BatchNorm1d(4096)    
        self.dropoutfc_1 = nn.Dropout2d(0.4)
          
        self.fc2 = nn.Linear(4096, 1000)
        self.normfc_2 = nn.BatchNorm1d(1000)    
        self.dropoutfc_2 = nn.Dropout2d(0.2)        

        self.fc3 = nn.Linear(1000, class_num)

    def forward(self, x):        
        out = F.relu(self.norm1_1(self.conv1_1(x)))
        out = self.pool1(out)
        out = self.dropout1(out)
        
        out = F.relu(self.norm2_1(self.conv2_1(out)))
        out = self.pool2(out)
        out = self.dropout2(out)

        out = F.relu(self.norm3_1(self.conv3_1(out)))
        out = F.relu(self.norm3_2(self.conv3_2(out)))
        out = self.pool3(out)
        out = self.dropout3(out)

        out = F.relu(self.conv4_1(out))
        out = F.relu(self.conv4_2(out))
        out = self.pool4(out)
        out = self.dropout4(out)  

        out = F.relu(self.conv5_1(out))
        out = F.relu(self.conv5_2(out))
        out = self.pool5(out)
        out = self.dropout5(out)
        
        # flatten
        out = out.view(-1, 512 * 7 * 7)        
        out = F.relu(self.fc1(out))
        out = self.dropoutfc_1(out)
        
        out = F.relu(self.fc2(out))
        out = self.dropoutfc_2(out)


        out = self.fc3(out)
        # softmax classifier
        # print(out.shape)
        out = F.softmax(out, dim=1)
        return out


class VGG(nn.Module):

    def __init__(self, features, num_classes=153, init_weights=False, pairwise=False):
        super(VGG, self).__init__()
        self.features = features

        if pairwise==True:
            self.classifier = nn.Sequential(
                nn.Linear(512 * 7 * 7, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, num_classes),
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(512 * 7 * 7, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, 1000),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(1000, 1),
                nn.Sigmoid()
            )


    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def make_layers(cfg, initial=3, batch_norm=False):
    layers = []
    in_channels = initial
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}



def vgg16(**kwargs):
    model = VGG(make_layers(cfg['D'], initial=3, batch_norm=True), **kwargs)
    return model


net16 = vgg16()
model_pretrained = resnet50(pretrained=True)



##################### model for pairwise detection #####################

def vgg_pair(**kwargs):
    model = VGG(make_layers(cfg['A'],initial=6,batch_norm=True),pairwise=False)
    return model
net16_pair = vgg_pair()