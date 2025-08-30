
from re import A
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

# 定义3x3卷积块
class progressBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, padding=1):
        super(progressBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class FourConv3x3Blocks(nn.Module):
    def __init__(self):
        super(FourConv3x3Blocks, self).__init__()
        self.cbr1 = progressBlock(in_channels=3, out_channels=32)  # Adjusted to match input channels
        self.cbr2 = progressBlock(in_channels=32, out_channels=64)
        self.cbr3 = progressBlock(in_channels=64, out_channels=128)
        self.cbr4 = progressBlock(in_channels=128, out_channels=128)
        self.cbr5 = progressBlock(in_channels=128,out_channels=128)
        

    def forward(self, x):
        x1 = self.cbr1(x)
        x1 = self.cbr2(x1)
        x1 = self.cbr3(x1)
        x1 = self.cbr4(x1)
        x1 = self.cbr5(x1)
        return x1

class Crossconvolution(nn.Module):
   def __init__(self): 
       super(Crossconvolution,self).__init__()
       self.conv1= nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size=(1, 3), padding=(0, 1))
       self.conv2= nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size=(3, 1), padding=(1,0))
       
       self.Sigmoid= nn.Sigmoid()
       self.conv = nn.Sequential(
         nn.Conv2d(in_channels =256, out_channels= 128, kernel_size=1),
         nn.BatchNorm2d( num_features = 128),#
         nn.ReLU(inplace=True)
       )

   def forward(self,x):
       outut13 = self.conv1(x)#13卷积
       outut31 = self.conv2(x)#31卷积
       Output1 = self.conv2(outut13)#13后31之后
       Output2 = self.conv1(outut31)#31后13之后
       Output1_1 = self.Sigmoid(Output1)
       Output1_2 = self.Sigmoid(Output2)
       Output2_1 = Output1_1 * outut31#下面
       Output2_2 = Output1_2 * outut13
       Output3_1 = outut13 + Output2_2
       Output3_2 = outut31 + Output2_1
       OUTPUT = torch.cat((Output3_1,Output3_2),dim=1)#连接张量的通道，并将通道减少到原来的通道数
       y = self.conv(OUTPUT)
       return y

class  kongdong1(nn.Module):
    def __init__(self): 
        super(kongdong1,self).__init__()
        self.conv3 = nn.Sequential(
        nn.Conv2d(in_channels = 128, out_channels = 64, kernel_size = 3, stride = 1, padding = 1),
        nn.BatchNorm2d(num_features = 64),
        nn.LeakyReLU(inplace = True),)
        self.conv5 = nn.Conv2d(in_channels=64,out_channels=64,kernel_size=5,stride=1,padding=2)
        self.conv7 = nn.Conv2d(in_channels=64,out_channels=64,kernel_size=7,stride=1,padding=3)
        self.conv33 = nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1)
        self.sigmoid = nn.Sigmoid()
    
    
    
    def forward(self,x):
        x = x
        x1 = self.conv3(x)
        x2 = self.conv5(x1)
        x3 = self.conv33(x1)
        x4 = self.conv7(x2)
        x5 = self.conv33(x2)
        x3 = self.conv33(x3)
        x3 = self.sigmoid(x3)
        x5 = self.sigmoid(x5)
        x44 = self.conv33(x4)
        x44 = self.sigmoid(x44)
        y = x44+x5+x3
        y1 = y*x1
        return y1



class Crossconvolution1(nn.Module):
   def __init__(self): 
       super(Crossconvolution1,self).__init__()
       self.conv1= nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size=(1, 3), padding=(0, 1))
       self.conv2= nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size=(3, 1), padding=(1,0))
       
       self.Sigmoid= nn.Sigmoid()
       self.conv = nn.Sequential(
         nn.Conv2d(in_channels =64, out_channels= 32, kernel_size=1),
         nn.BatchNorm2d( num_features = 32),#
         nn.ReLU(inplace=True)
       )

   def forward(self,x):
       outut13 = self.conv1(x)#13卷积
       outut31 = self.conv2(x)#31卷积
       Output1 = self.conv2(outut13)#13后31之后
       Output2 = self.conv1(outut31)#31后13之后
       Output1_1 = self.Sigmoid(Output1)
       Output1_2 = self.Sigmoid(Output2)
       Output2_1 = Output1_1 * outut31#下面
       Output2_2 = Output1_2 * outut13
       Output3_1 = outut13 + Output2_2
       Output3_2 = outut31 + Output2_1
       OUTPUT = torch.cat((Output3_1,Output3_2),dim=1)#连接张量的通道，并将通道减少到原来的通道数
       y = self.conv(OUTPUT)
       return y

class selfattention1(nn.Module):

    def __init__(self,in_channels = 128):
        super(selfattention1,self).__init__()
        self.in_channels = in_channels
        self.query = nn.Conv2d(in_channels, in_channels // 8, kernel_size = 1, stride = 1)
        self.key   = nn.Conv2d(in_channels, in_channels // 8, kernel_size = 1, stride = 1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size = 1, stride = 1)
        self.gamma = nn.Parameter(torch.zeros(1))  #gamma为一个衰减参数，由torch.zero生成，nn.Parameter的作用是将其转化成为可以训练的参数.
        self.softmax = nn.Softmax(dim = -1)
        self.sigmoin = nn.Sigmoid()

        

    def forward(self, input):
        batch_size, in_channels, height, width = input.shape
        # input: B, C, H, W -> q: B, H * W, C // 8
        q = self.query(input).view(batch_size, -1, height * width).permute(0, 2, 1)
        #input: B, C, H, W -> k: B, C // 8, H * W
        k = self.key(input).view(batch_size, -1, height * width)
        #input: B, C, H, W -> v: B, C, H * W
        v = self.value(input).view(batch_size, -1, height * width)
        #q: B, H * W, C // 8 x k: B, C // 8, H * W -> attn_matrix: B, H * W, H * W
        attn_matrix = torch.bmm(q, k)  #torch.bmm进行tensor矩阵乘法,q与k相乘得到的值为attn_matrix.
        attn_matrix = self.softmax(attn_matrix)#经过一个softmax进行缩放权重大小.
        out = torch.bmm(v, attn_matrix.permute(0, 2, 1))  #tensor.permute将矩阵的指定维进行换位.这里将1于2进行换位。
        out = out.view(*input.shape)
        out = self.sigmoin(out)
        return out

class selfattention(nn.Module):

    def __init__(self,in_channels = 32):
        super().__init__()
        self.in_channels = in_channels
        self.query = nn.Conv2d(in_channels, in_channels // 8, kernel_size = 1, stride = 1)
        self.key   = nn.Conv2d(in_channels, in_channels // 8, kernel_size = 1, stride = 1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size = 1, stride = 1)
        self.gamma = nn.Parameter(torch.zeros(1))  #gamma为一个衰减参数，由torch.zero生成，nn.Parameter的作用是将其转化成为可以训练的参数.
        self.softmax = nn.Softmax(dim = -1)
        self.sigmoin = nn.Sigmoid()

        

    def forward(self, input):
        batch_size, in_channels, height, width = input.shape
        # input: B, C, H, W -> q: B, H * W, C // 8
        q = self.query(input).view(batch_size, -1, height * width).permute(0, 2, 1)
        #input: B, C, H, W -> k: B, C // 8, H * W
        k = self.key(input).view(batch_size, -1, height * width)
        #input: B, C, H, W -> v: B, C, H * W
        v = self.value(input).view(batch_size, -1, height * width)
        #q: B, H * W, C // 8 x k: B, C // 8, H * W -> attn_matrix: B, H * W, H * W
        attn_matrix = torch.bmm(q, k)  #torch.bmm进行tensor矩阵乘法,q与k相乘得到的值为attn_matrix.
        attn_matrix = self.softmax(attn_matrix)#经过一个softmax进行缩放权重大小.
        out = torch.bmm(v, attn_matrix.permute(0, 2, 1))  #tensor.permute将矩阵的指定维进行换位.这里将1于2进行换位。
        out = out.view(*input.shape)
        out = self.sigmoin(out)
        return out




class SE_Block(nn.Module):
    def __init__(self):
        super(SE_Block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局自适应池化
        self.fc = nn.Sequential(
            nn.Linear(32, 16, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(16, 32, bias=False),
            nn.Sigmoid()
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(32,16,1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16,32,1,1),
            nn.GELU(),
            nn.Conv2d(32,32,1,1)
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c) # squeeze操作
        y = self.fc(y).view(b, c, 1, 1) # FC获取通道注意力权重，是具有全局信息的
        z = x * y.expand_as(x)
        z = self.conv1(z)
        z = self.conv2(z)
    
        return z


class nnn(nn.Module):
    def __init__(self):
        super(nnn,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(32,16,3,1,1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.conv2 = SE_Block()
        self.conv1 = nn.Sequential(
            nn.Conv2d(16,32,3,1,1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32,32,3,1,1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
    
    
    def forward(self,x):
        x1 = self.conv2(x)
        x2 = self.conv(x)
        x2 = self.conv1(x2)
        x2 = self.conv3(x2)
        y = x2*x1
        y = self.conv3(y)
        y = self.conv3(y)
        return y
        



class ws(nn.Module):
    def __init__(self):
        super(ws,self).__init__()
        self.conv1 = nn.Conv2d(32,32,1,1,0)
        self.sigmoid = nn.Sigmoid()
        self.conv3 = nn.Conv2d(32,32,3,1,1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(64,32,1,1,0)
    
    def forward(self,x):
        x = x
        x1 = self.conv1(x)
        x1 = self.sigmoid(x1)
        x2 = x1*x
        x2 = self.conv3(x2)
        x2 = self.relu(x2)
        x2 = self.conv1(x2)
        x3 = self.conv3(x)
        x3 = self.relu(x3)
        x3 = self.conv1(x3)
        y = torch.cat((x3,x2),dim=1)
        y = self.conv2(y)
        return y
    
    

 
class ChannelAttention(nn.Module):
    def __init__(self, in_planes=32, ratio=4):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
 
        self.fc1   = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
 
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        out = out*x
        return self.sigmoid(out)




    
    
    

class NET(nn.Module):
    def __init__(self):
       super(NET,self).__init__()
       A = nn.Parameter(torch.tensor(0.0))
       self.A = A
       self.bach = nn.LazyBatchNorm2d()
       self.batch = nn.LazyBatchNorm2d(32)
       
       self.sof = nn.Sigmoid()
       self.gelu = nn.GELU()
       self.drop = nn.Dropout()
     
       self.conv1 = nn.Sequential(
           nn.Conv2d(128,32,3,1,1),
           nn.BatchNorm2d(32),
           nn.ReLU(),
       )

       self.conv77 = nn.Sequential(
            nn.Conv2d(in_channels = 128,out_channels = 128,kernel_size=7,stride=1,padding=3),
            nn.BatchNorm2d(num_features=128),
            nn.LeakyReLU(inplace=True)
        )
       self.conv555 = nn.Sequential(
            nn.Conv2d(in_channels = 128,out_channels = 128,kernel_size=5,stride=1,padding=2),
            nn.BatchNorm2d(num_features=128),
            nn.LeakyReLU(inplace=True)
        )
       self.conv333 = nn.Sequential(
            nn.Conv2d(in_channels = 128,out_channels =128,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.LeakyReLU(inplace=True)
        )
       
       self.conv111 = nn.Sequential(
           nn.Conv2d(in_channels=128,out_channels=64,kernel_size=1,stride=1,padding=0),
           nn.BatchNorm2d(64),
           nn.ReLU()
       )
       
    #    self.conv_change=nn.Conv2d(32,3,1,1)
       self.conv_change = nn.Sequential(
           nn.Conv2d(32,32,3,1,1),
           nn.Conv2d(32,3,1,1,0),
           nn.BatchNorm2d(3),
      
       )
       
       self.depth_conv = nn.Conv2d(in_channels=64,
                                    out_channels=128,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1,
                                    groups=32)

       self.point_conv = nn.Conv2d(in_channels=128,
                                    out_channels=128,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    groups=1)

             
       self.depth_conv1 = nn.Conv2d(in_channels=64,
                                    out_channels=64,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1,
                                    groups=32)
       
       self.point_conv1 = nn.Conv2d(in_channels=64,
                                    out_channels=64,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    groups=1)
       self.dropout = nn.Dropout()
       
      
       self.gamma = nn.Parameter(torch.zeros(1))  #gamma为一个衰减参数，由torch.zero生成，nn.Parameter的作用是将其转化成为可以训练的参数.
       self.relu = nn.ReLU()
       self.yu = FourConv3x3Blocks()
       self.fang1 = Crossconvolution()
       self.fang2 = kongdong1()
       self.fang3 = Crossconvolution1()
       self.fang4 = selfattention()
       self.fang5 = ws()
       self.fang6 = nnn()
       self.fang7 = ChannelAttention()
  
       self.batch1 = nn.LazyBatchNorm2d(128)
       self.fangyu = selfattention1()
       self.conv = nn.Sequential(
           nn.Conv2d(128,64,3,1,1),
           nn.Conv2d(64,64,3,1,1),
           nn.Conv2d(64,128,3,1,1),
           nn.LazyBatchNorm2d(128)
       )
       


    def forward(self,x):
        x = self.yu(x)
        y = self.fangyu(x)
        x = x*(y) + self.A*(1-y)
        
        x = self.bach(x)
        x1 = self.conv77(x) #128
        x2 = self.conv555(x) # 128
        x3 = self.conv333(x) #128        
        # x4 = self.conv111(x)
        x1 = self.fang1(x1)
        x1 = self.fang2(x1)
        x1 = self.depth_conv(x1)
        x1 = self.point_conv(x1)
        
        
        x11 = self.sof(x1)
        x22 = x11*x2
        x22 = x22+x2
        x22 = self.fang1(x22)
        x22 = self.fang2(x22)#([4, 64, 32, 32])
        x22 = self.depth_conv(x22)
        x22 = self.point_conv(x22)
        
        
        x222 = self.sof(x22)
        x33 = x3*x222
        x33 = x33+x22#最后一个分支输出的128
        


        x33333 = self.conv1(x33)

        x0 = self.fang3(x33333)
    
     

        t = self.fang4(x0)     
        J = x0*(t)+self.A*(1 - t)
        
        t = self.fang4(J)
        J = J*t + self.A*(1-t)#
        
        

    
    
    
        y = self.conv_change(J)
        return y    
        



if __name__=='__main__':
  

   net = NET()

   x = torch.randn(4, 3 , 32, 32) #批次数，通道数，宽和高度

   out = net(x)

   print(out.shape)

   writer = SummaryWriter('logs/net') #路径还没进行修改
   writer.add_graph(net, (x))  # net是网络

      
   writer.close() 

      



       


       

       
  
   
       
  
       
    

