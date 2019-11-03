## Define the Model
import torch
import torch.nn as nn
import torch.nn.functional as F




class MesoInception4(nn.Module):
    
    def __init__(self):
        super(MesoInception4,self).__init__()

        ## 1,4,4,2 -> inception layer 1

        self.inception_1_conv_1_1  = nn.Conv2d(3,1,1,padding=0)

        self.inception_1_conv_2_1  = nn.Conv2d(3,4,1,padding=0)
        self.inception_1_conv_2_2  = nn.Conv2d(4,4,3,padding=1)

        self.inception_1_conv_3_1  = nn.Conv2d(3,4,1,padding=0)
        self.inception_1_conv_3_2  = nn.Conv2d(4,4,3,padding=2,dilation=2)

        self.inception_1_conv_4_1 = nn.Conv2d(3,2,1,padding=0)
        self.inception_1_conv_4_2 = nn.Conv2d(2,2,3,dilation=3,padding=3)

        self.bn_1 = nn.BatchNorm2d(11)

        ## 2,4,4,2 -> inception layer 2

        self.inception_2_conv_1_1  = nn.Conv2d(11,2,1,padding=0)

        self.inception_2_conv_2_1  = nn.Conv2d(11,4,1,padding=0)
        self.inception_2_conv_2_2  = nn.Conv2d(4,4,3,padding=1)

        self.inception_2_conv_3_1  = nn.Conv2d(11,4,1,padding=0)
        self.inception_2_conv_3_2  = nn.Conv2d(4,4,3,padding=2,dilation=2)

        self.inception_2_conv_4_1 = nn.Conv2d(11,2,1,padding=0)
        self.inception_2_conv_4_2 = nn.Conv2d(2,2,3,dilation=3,padding=3)

        self.bn_2 = nn.BatchNorm2d(12)


        self.conv_1 = nn.Conv2d(12,16,5,padding=2)     ## inpute_channel,output_channel,kernel_size
        self.relu = nn.ReLU(inplace=True)
        self.leakyrelu = nn.LeakyReLU(0.1)
        self.bn_3 = nn.BatchNorm2d(16)
        self.max_pool_1 = nn.MaxPool2d((2,2))

        self.conv_2 = nn.Conv2d(16,16,5,padding=2)
        self.max_pool_2 = nn.MaxPool2d((4,4))

        self.dropout = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(16*8*8,16)
        self.fc2 = nn.Linear(16,2)

    def inception_module_1(self,x):
        ## Gets input for module 1 
        ## returns output of module 1
        x1 =  self.relu(self.inception_1_conv_1_1(x))

        x2 =  self.relu(self.inception_1_conv_2_1(x))
        x2 =  self.relu(self.inception_1_conv_2_2(x2)) 

        x3 =  self.relu(self.inception_1_conv_3_1(x))
        x3 =  self.relu(self.inception_1_conv_3_2(x3))

        x4 =  self.relu(self.inception_1_conv_4_1(x))
        x4 =  self.relu(self.inception_1_conv_4_2(x4))

        y = torch.cat((x1,x2,x3,x4),1)
        y = self.bn_1(y)
        y = self.max_pool_1(y)

        return y

    def inception_module_2(self,x):
        ## Gets input for module 2 
        ## returns output of module 2
        x1 =  self.relu(self.inception_2_conv_1_1(x))

        x2 =  self.relu(self.inception_2_conv_2_1(x))
        x2 =  self.relu(self.inception_2_conv_2_2(x2))

        x3 =  self.relu(self.inception_2_conv_3_1(x))
        x3 =  self.relu(self.inception_2_conv_3_2(x3))

        x4 =  self.relu(self.inception_2_conv_4_1(x))
        x4 =  self.relu(self.inception_2_conv_4_2(x4))

        y = torch.cat((x1,x2,x3,x4),1)
        y = self.bn_2(y)
        y = self.max_pool_1(y)
    
        return y


    def forward(self,x):
        '''
        Forward pass of the model
        '''
        x = self.inception_module_1(x)

        x = self.inception_module_2(x)
        
        x = self.conv_1(x)
        x = self.relu(x)
        x = self.bn_3(x)
        x = self.max_pool_1(x)

        x = self.conv_2(x)
        x = self.relu(x)
        x = self.bn_3(x)
        x = self.max_pool_2(x)
        
        x = x.view(-1,16*8*8) ## Flatten the layer for dense operations

        
        x = self.dropout(x)
        x = self.leakyrelu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = F.sigmoid(x)
        return x
