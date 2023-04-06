import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchvision.models.vgg import vgg16
import pytorch_ssim

import numpy as np
from IQA_pytorch import SSIM

class L_retouch_mean(nn.Module):

    def __init__(self):
        super(L_retouch_mean, self).__init__()
        self.ssim_loss = pytorch_ssim.SSIM()

    def forward(self, x, y):
        x = x.max(1,keepdim=True)[0]
        y = y.max(1,keepdim=True)[0]
        L3_retouch_mean = torch.mean(torch.pow(x-torch.mean(y,[2,3],keepdim=True),2))
        L4_retouch_ssim = 1-torch.mean(self.ssim_loss(x, y))

        return L3_retouch_mean+L4_retouch_ssim

class L_recon(nn.Module):

    def __init__(self):
        super(L_recon, self).__init__()
        self.ssim_loss = pytorch_ssim.SSIM()

    def forward(self, R_low, high):
        L1 = torch.abs(R_low - high).mean()
        L2 = (1- self.ssim_loss(R_low,high)).mean()
        return L1,L2


class L_recon_low(nn.Module):

    def __init__(self):
        super(L_recon_low, self).__init__()
        self.ssim_loss = pytorch_ssim.SSIM()

    def forward(self, R_low, high,ill):
        L1 = (R_low*ill - high*torch.log(R_low*ill + 0.0001)).mean()

        return L1

class L_color_zy(nn.Module):

    def __init__(self):
        super(L_color_zy, self).__init__()

    def forward(self, x, y):
        
        product_separte_color = (x*y).mean(1,keepdim=True)
        x_abs = (x**2).mean(1,keepdim=True)**0.5
        y_abs = (y**2).mean(1,keepdim=True)**0.5
        loss1 = (1-product_separte_color/(x_abs*y_abs+0.00001)).mean() + torch.mean(torch.acos(product_separte_color/(x_abs*y_abs+0.00001)))

        return loss1

class L_grad_cosist(nn.Module):

    def __init__(self):
        super(L_grad_cosist, self).__init__()
        kernel_right = torch.FloatTensor( [[0,0,0],[0,1,-1],[0,0,0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_down = torch.FloatTensor( [[0,0,0],[0,1, 0],[0,-1,0]]).cuda().unsqueeze(0).unsqueeze(0)
        self.weight_right = nn.Parameter(data=kernel_right, requires_grad=False)
        self.weight_down = nn.Parameter(data=kernel_down, requires_grad=False)

    def gradient_of_one_channel(self,x,y):
        D_org_right = F.conv2d(x , self.weight_right, padding="same")
        D_org_down = F.conv2d(x , self.weight_down, padding="same")
        D_enhance_right = F.conv2d(y , self.weight_right, padding="same")
        D_enhance_down = F.conv2d(y , self.weight_down, padding="same")
        return torch.abs(D_org_right),torch.abs(D_enhance_right),torch.abs(D_org_down),torch.abs(D_enhance_down)

    def gradient_Consistency_loss_patch(self,x,y):
        # B*C*H*W
        min_x = torch.abs(x.min(2,keepdim=True)[0].min(3,keepdim=True)[0]).detach()
        min_y = torch.abs(y.min(2,keepdim=True)[0].min(3,keepdim=True)[0]).detach()
        x = x - min_x
        y = y - min_y
        #B*1*1,3
        product_separte_color = (x*y).mean([2,3],keepdim=True)
        x_abs = (x**2).mean([2,3],keepdim=True)**0.5
        y_abs = (y**2).mean([2,3],keepdim=True)**0.5
        loss1 = (1-product_separte_color/(x_abs*y_abs+0.00001)).mean() + torch.mean(torch.acos(product_separte_color/(x_abs*y_abs+0.00001)))

        product_combine_color = torch.mean(product_separte_color,1,keepdim=True)
        x_abs2 = torch.mean(x_abs**2,1,keepdim=True)**0.5
        y_abs2 = torch.mean(y_abs**2,1,keepdim=True)**0.5
        loss2 = torch.mean(1-product_combine_color/(x_abs2*y_abs2+0.00001)) + torch.mean(torch.acos(product_combine_color/(x_abs2*y_abs2+0.00001)))
        return loss1 + loss2


    def forward(self, x, y):

        x_R1,y_R1, x_R2,y_R2  = self.gradient_of_one_channel(x[:,0:1,:,:],y[:,0:1,:,:])
        x_G1,y_G1, x_G2,y_G2  = self.gradient_of_one_channel(x[:,1:2,:,:],y[:,1:2,:,:])
        x_B1,y_B1, x_B2,y_B2  = self.gradient_of_one_channel(x[:,2:3,:,:],y[:,2:3,:,:])
        x = torch.cat([x_R1,x_G1,x_B1,x_R2,x_G2,x_B2],1)
        y = torch.cat([y_R1,y_G1,y_B1,y_R2,y_G2,y_B2],1)

        B,C,H,W = x.shape
        loss = self.gradient_Consistency_loss_patch(x,y)
        loss1 = 0
        loss1 += self.gradient_Consistency_loss_patch(x[:,:,0:H//2,0:W//2],y[:,:,0:H//2,0:W//2])
        loss1 += self.gradient_Consistency_loss_patch(x[:,:,H//2:,0:W//2],y[:,:,H//2:,0:W//2])
        loss1 += self.gradient_Consistency_loss_patch(x[:,:,0:H//2,W//2:],y[:,:,0:H//2,W//2:])
        loss1 += self.gradient_Consistency_loss_patch(x[:,:,H//2:,W//2:],y[:,:,H//2:,W//2:])

        return loss #+loss1#+torch.mean(torch.abs(x-y))#+loss1


class L_bright_cosist(nn.Module):

    def __init__(self):
        super(L_bright_cosist, self).__init__()

    def gradient_Consistency_loss_patch(self,x,y):
        # B*C*H*W
        min_x = torch.abs(x.min(2,keepdim=True)[0].min(3,keepdim=True)[0]).detach()
        min_y = torch.abs(y.min(2,keepdim=True)[0].min(3,keepdim=True)[0]).detach()
        x = x - min_x
        y = y - min_y
        #B*1*1,3
        product_separte_color = (x*y).mean([2,3],keepdim=True)
        x_abs = (x**2).mean([2,3],keepdim=True)**0.5
        y_abs = (y**2).mean([2,3],keepdim=True)**0.5
        loss1 = (1-product_separte_color/(x_abs*y_abs+0.00001)).mean() + torch.mean(torch.acos(product_separte_color/(x_abs*y_abs+0.00001)))

        product_combine_color = torch.mean(product_separte_color,1,keepdim=True)
        x_abs2 = torch.mean(x_abs**2,1,keepdim=True)**0.5
        y_abs2 = torch.mean(y_abs**2,1,keepdim=True)**0.5
        loss2 = torch.mean(1-product_combine_color/(x_abs2*y_abs2+0.00001)) + torch.mean(torch.acos(product_combine_color/(x_abs2*y_abs2+0.00001)))
        return loss1 + loss2

    def forward(self, x, y):

        B,C,H,W = x.shape
        loss = self.gradient_Consistency_loss_patch(x,y)
        loss1 = 0
        loss1 += self.gradient_Consistency_loss_patch(x[:,:,0:H//2,0:W//2],y[:,:,0:H//2,0:W//2])
        loss1 += self.gradient_Consistency_loss_patch(x[:,:,H//2:,0:W//2],y[:,:,H//2:,0:W//2])
        loss1 += self.gradient_Consistency_loss_patch(x[:,:,0:H//2,W//2:],y[:,:,0:H//2,W//2:])
        loss1 += self.gradient_Consistency_loss_patch(x[:,:,H//2:,W//2:],y[:,:,H//2:,W//2:])

        return loss #+loss1#+torch.mean(torch.abs(x-y))#+loss1
# class L_grad_cosist(nn.Module):

#     def __init__(self):
#         super(L_grad_cosist, self).__init__()
#         kernel_right = torch.FloatTensor( [[0,0,0],[0,1,-1],[0,0,0]]).cuda().unsqueeze(0).unsqueeze(0)
#         kernel_down = torch.FloatTensor( [[0,0,0],[0,1, 0],[0,-1,0]]).cuda().unsqueeze(0).unsqueeze(0)
#         self.weight_right = nn.Parameter(data=kernel_right, requires_grad=False)
#         self.weight_down = nn.Parameter(data=kernel_down, requires_grad=False)

#     def gradient_of_one_channel(self,x,y):
#         D_org_right = F.conv2d(x , self.weight_right, padding="same")
#         D_org_down = F.conv2d(x , self.weight_down, padding="same")
#         D_enhance_right = F.conv2d(y , self.weight_right, padding="same")
#         D_enhance_down = F.conv2d(y , self.weight_down, padding="same")
#         return D_org_right,D_enhance_right,D_org_down,D_enhance_down

#     def gradient_Consistency_loss_patch(self,x,y):
#         # B*C*H*W
#         # min_x = torch.abs(x.min(2,keepdim=True)[0].min(3,keepdim=True)[0]).detach()
#         # min_y = torch.abs(y.min(2,keepdim=True)[0].min(3,keepdim=True)[0]).detach()
#         # x = x - min_x
#         # y = y - min_y
#         #B*1*1,3
#         product_separte_color = torch.abs((x*y).mean([2,3],keepdim=True))
#         x_abs = (x**2).mean([2,3],keepdim=True)**0.5
#         y_abs = (y**2).mean([2,3],keepdim=True)**0.5
#         loss1 = (1-product_separte_color/(x_abs*y_abs+0.00001)).mean() + torch.mean(torch.acos(product_separte_color/(x_abs*y_abs+0.00001)))

#         product_combine_color = torch.mean(product_separte_color,1,keepdim=True)
#         x_abs2 = torch.mean(x_abs**2,1,keepdim=True)**0.5
#         y_abs2 = torch.mean(y_abs**2,1,keepdim=True)**0.5
#         loss2 = torch.mean(1-product_combine_color/(x_abs2*y_abs2+0.00001)) + torch.mean(torch.acos(product_combine_color/(x_abs2*y_abs2+0.00001)))
#         return loss1 + loss2


#     def forward(self, x, y):

#         x_R1,y_R1, x_R2,y_R2  = self.gradient_of_one_channel(x[:,0:1,:,:],y[:,0:1,:,:])
#         x_G1,y_G1, x_G2,y_G2  = self.gradient_of_one_channel(x[:,1:2,:,:],y[:,1:2,:,:])
#         x_B1,y_B1, x_B2,y_B2  = self.gradient_of_one_channel(x[:,2:3,:,:],y[:,2:3,:,:])
#         x = torch.cat([x_R1,x_G1,x_B1,x_R2,x_G2,x_B2],1)
#         y = torch.cat([y_R1,y_G1,y_B1,y_R2,y_G2,y_B2],1)

#         B,C,H,W = x.shape
#         loss = self.gradient_Consistency_loss_patch(x,y)
#         loss1 = 0
#         loss1 += self.gradient_Consistency_loss_patch(x[:,:,0:H//2,0:W//2],y[:,:,0:H//2,0:W//2])
#         loss1 += self.gradient_Consistency_loss_patch(x[:,:,H//2:,0:W//2],y[:,:,H//2:,0:W//2])
#         loss1 += self.gradient_Consistency_loss_patch(x[:,:,0:H//2,W//2:],y[:,:,0:H//2,W//2:])
#         loss1 += self.gradient_Consistency_loss_patch(x[:,:,H//2:,W//2:],y[:,:,H//2:,W//2:])

#         return loss#+torch.mean(torch.abs(x-y))+loss1

class L_diff_zy(nn.Module):

    def __init__(self):
        super(L_diff_zy, self).__init__()
        kernel_right = torch.FloatTensor( [[0,0,0],[0,1,-1],[0,0,0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_down = torch.FloatTensor( [[0,0,0],[0,1, 0],[0,-1,0]]).cuda().unsqueeze(0).unsqueeze(0)
        self.weight_right = nn.Parameter(data=kernel_right, requires_grad=False)
        self.weight_down = nn.Parameter(data=kernel_down, requires_grad=False)


    def diff_zy(self, input_I, input_R):

        input_I_W_x,input_I_I_x=self.gradient_n_diff(input_I, "x")  
        input_R_W_x,input_R_I_x=self.gradient_n_diff(input_R, "x")
        input_I_W_y,input_I_I_y=self.gradient_n_diff(input_I, "y")  
        input_R_W_y,input_R_I_y=self.gradient_n_diff(input_R, "y")
        return torch.mean(input_I_I_x - input_R_I_x *torch.log( input_I_I_x+0.0001))+torch.mean(input_I_I_y - input_R_I_y *torch.log( input_I_I_y+0.0001))

    def gradient_n_I(self,input_tensor, direction):
        if direction == "x":
            kernel = self.weight_right
        elif direction == "y":
            kernel = self.weight_down
        gradient_orig1 = F.conv2d(input_tensor , kernel, padding="same")

        gradient_orig_abs = torch.abs(gradient_orig1)

        grad_min1 = torch.min(torch.min(gradient_orig_abs,2,keepdim=True),3,keepdim=True)
        grad_max1 = torch.max(torch.max(gradient_orig_abs,2,keepdim=True),3,keepdim=True)
        grad_norm1 = torch.div((gradient_orig_abs - (grad_min1)), (grad_max1 - grad_min1 + 0.0001))

        gradient_orig = torch.abs(F.avg_pool2d(gradient_orig1,5,stride=1, padding=2,count_include_pad =False))#denoise
        
        gradient_orig_patch_max = F.max_pool2d(gradient_orig, 7,stride=1, padding=3 )
        gradient_orig_patch_min = torch.abs(1-F.max_pool2d(1-gradient_orig, 7,stride=1, padding=3))
        gradient_orig_patch_max = F.avg_pool2d(gradient_orig_patch_max,7,stride=1, padding=3,count_include_pad=False)
        gradient_orig_patch_min = F.avg_pool2d(gradient_orig_patch_min,7,stride=1, padding=3,count_include_pad=False)

        grad_norm = torch.div((gradient_orig - gradient_orig_patch_min), (gradient_orig_patch_max - gradient_orig_patch_min + 0.0001))
        return (grad_norm+0.01).detach()*grad_norm1

    def gradient_n_diff(self,input_tensor, direction):
        if direction == "x":
            kernel = self.weight_right
        elif direction == "y":
            kernel = self.weight_down
        gradient_orig1 = F.conv2d(input_tensor , kernel, padding="same")
        gradient_orig1_abs = torch.abs(gradient_orig1)

        grad_min1 = torch.min(torch.min(gradient_orig_abs,2,keepdim=True),3,keepdim=True)
        grad_max1 = torch.max(torch.max(gradient_orig_abs,2,keepdim=True),3,keepdim=True)
        grad_norm1 = torch.div((gradient_orig_abs - (grad_min1)), (grad_max1 - grad_min1 + 0.0001))

        input_tensor = F.avg_pool2d(input_tensor,[5,5],stride=1, padding=2,count_include_pad =False)#denoise
        gradient_orig = torch.abs(F.conv2d(input_tensor, kernel, padding='same'))
        
        gradient_orig_patch_max = F.max_pool2d(gradient_orig, [7, 7],stride=1, padding=3 )
        gradient_orig_patch_min = torch.abs(1-F.max_pool2d(1-gradient_orig, [7, 7],stride=1, padding=3 ))

        gradient_orig_patch_max = F.avg_pool2d(gradient_orig_patch_max,[7,7],stride=1, padding=3,count_include_pad=False)
        gradient_orig_patch_min = F.avg_pool2d(gradient_orig_patch_min,[7,7],stride=1, padding=3,count_include_pad=False)

        grad_norm = torch.div((gradient_orig - gradient_orig_patch_min), (gradient_orig_patch_max - gradient_orig_patch_min + 0.0001))
        # return tf.stop_gradient(grad_norm+0.05)*grad_norm1
        return (grad_norm+0.05).detach(),grad_norm1


    def forward(self, R_low, high):
        return self.diff_zy(R_low,high)

class L_smooth4(nn.Module):

    def __init__(self):
        super(L_smooth4, self).__init__()
        kernel_right = torch.FloatTensor( [[0,0,0],[0,1,-1],[0,0,0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_down = torch.FloatTensor( [[0,0,0],[0,1, 0],[0,-1,0]]).cuda().unsqueeze(0).unsqueeze(0)
        self.weight_right = nn.Parameter(data=kernel_right, requires_grad=False)
        self.weight_down = nn.Parameter(data=kernel_down, requires_grad=False)


    def gradient_n_p(self,input_tensor, direction):
        if direction == "x":
            kernel = self.weight_right
        elif direction == "y":
            kernel = self.weight_down
        gradient_orig1 = F.conv2d(input_tensor, kernel, padding="same")
        gradient_orig_abs = torch.abs(gradient_orig1)

        gradient_orig_patch_max1 = F.max_pool2d(gradient_orig_abs, kernel_size=9,stride=1, padding=4 )
        gradient_orig_patch_min1 = torch.abs(1-F.max_pool2d(1-gradient_orig_abs, kernel_size=9,stride=1, padding=4))
        grad_max1 = F.avg_pool2d(gradient_orig_patch_max1,kernel_size=17,stride=1, padding=8,count_include_pad=False)
        grad_min1 = F.avg_pool2d(gradient_orig_patch_min1,kernel_size=17,stride=1, padding=8,count_include_pad=False)
        grad_norm1 = torch.div((gradient_orig_abs - (grad_min1).detach()), torch.abs(grad_max1.detach() - grad_min1.detach()) + 0.0001)

        input_tensor2 =  F.avg_pool2d(input_tensor,kernel_size=5,stride=1, padding=2,count_include_pad=False)

        gradient_orig = torch.abs(F.conv2d(input_tensor2, kernel, padding="same"))

        gradient_orig_patch_max = F.max_pool2d(gradient_orig, 7,stride=1, padding=3 )
        gradient_orig_patch_min = torch.abs(1-F.max_pool2d(1-gradient_orig, 7,stride=1, padding=3))
        gradient_orig_patch_max = F.avg_pool2d(gradient_orig_patch_max,7,stride=1, padding=3,count_include_pad=False)
        gradient_orig_patch_min = F.avg_pool2d(gradient_orig_patch_min,7,stride=1, padding=3,count_include_pad=False)

        grad_norm = torch.div((gradient_orig - gradient_orig_patch_min), (torch.abs(gradient_orig_patch_max - gradient_orig_patch_min) + 0.0001))

        return (grad_norm+0.01).detach()*grad_norm1
 
    def forward(self, R_low):
        B, C, H,W= R_low.shape
        if C ==3:
            # R_low = torch.mean(R_low,1,keepdim=True)
            R = R_low[:,0:1,:,:]
            G = R_low[:,1:2,:,:]
            B = R_low[:,2:3,:,:]
            R_low = 0.299*R+0.587*G+0.114*B
        else:
            R_low = R_low
        R_low_x = self.gradient_n_p(R_low, "x")
        R_low_y = self.gradient_n_p(R_low, "y")


        return torch.mean(R_low_x * torch.exp(-10 * R_low_x)) + torch.mean(R_low_y * torch.exp(-10 * R_low_y))#+another_

class L_smooth_ill(nn.Module):

    def __init__(self):
        super(L_smooth_ill, self).__init__()
        kernel_right = torch.FloatTensor( [[0,0,0],[0,1,-1],[0,0,0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_down = torch.FloatTensor( [[0,0,0],[0,1, 0],[0,-1,0]]).cuda().unsqueeze(0).unsqueeze(0)
        self.weight_right = nn.Parameter(data=kernel_right, requires_grad=False)
        self.weight_down = nn.Parameter(data=kernel_down, requires_grad=False)

    def gradient_n_p(self,input_tensor, direction):
        if direction == "x":
            kernel = self.weight_right
        elif direction == "y":
            kernel = self.weight_down
        gradient_orig1 = F.conv2d(input_tensor, kernel, padding="same")
        gradient_orig_abs = torch.abs(gradient_orig1)

        gradient_orig_patch_max1 = F.max_pool2d(gradient_orig_abs, kernel_size=9,stride=1, padding=4 )
        gradient_orig_patch_min1 = torch.abs(1-F.max_pool2d(1-gradient_orig_abs, kernel_size=9,stride=1, padding=4))
        grad_max1 = F.avg_pool2d(gradient_orig_patch_max1,kernel_size=17,stride=1, padding=8,count_include_pad=False)
        grad_min1 = F.avg_pool2d(gradient_orig_patch_min1,kernel_size=17,stride=1, padding=8,count_include_pad=False)
        grad_norm1 = torch.div((gradient_orig_abs - (grad_min1).detach()), torch.abs(grad_max1.detach() - grad_min1.detach()) + 0.0001)

        gradient_orig = F.conv2d(input_tensor, kernel, padding="same")
        gradient_orig = torch.abs( F.avg_pool2d(gradient_orig,kernel_size=5,stride=1, padding=2,count_include_pad=False))

        gradient_orig_patch_max = F.max_pool2d(gradient_orig, 7,stride=1, padding=3 )
        gradient_orig_patch_min = torch.abs(1-F.max_pool2d(1-gradient_orig, 7,stride=1, padding=3))
        gradient_orig_patch_max = F.avg_pool2d(gradient_orig_patch_max,7,stride=1, padding=3,count_include_pad=False)
        gradient_orig_patch_min = F.avg_pool2d(gradient_orig_patch_min,7,stride=1, padding=3,count_include_pad=False)

        grad_norm = torch.div((gradient_orig - gradient_orig_patch_min), (torch.abs(gradient_orig_patch_max - gradient_orig_patch_min) + 0.0001))

        return (grad_norm+0.01).detach()*grad_norm1
 
    def forward(self, R_low,low):
        B, C, H,W= R_low.shape
        if C ==3:
            # R_low = torch.mean(R_low,1,keepdim=True)
            R = R_low[:,0:1,:,:]
            G = R_low[:,1:2,:,:]
            B = R_low[:,2:3,:,:]
            R_low = 0.299*R+0.587*G+0.114*B
            R = low[:,0:1,:,:]
            G = low[:,1:2,:,:]
            B = low[:,2:3,:,:]
            low = 0.299*R+0.587*G+0.114*B

        else:
            R_low = R_low
            low = low
        R_low_x = self.gradient_n_p(R_low, "x")
        R_low_y = self.gradient_n_p(R_low, "y")
        low_x = self.gradient_n_p(low, "x")
        low_y = self.gradient_n_p(low, "y")

        return torch.mean(R_low_x * torch.exp(-10 * R_low_x)* torch.exp(-10 * low_x)) + torch.mean(R_low_y * torch.exp(-10 * R_low_y)* torch.exp(-10 * low_y))#+another_
        return torch.mean(R_low_x * torch.exp(-10 * R_low_x)) + torch.mean(R_low_y * torch.exp(-10 * R_low_y))#+another_


class L_color(nn.Module):

    def __init__(self):
        super(L_color, self).__init__()

    def forward(self, x ):

        b,c,h,w = x.shape

        mean_rgb = torch.mean(x,[2,3],keepdim=True)
        mr,mg, mb = torch.split(mean_rgb, 1, dim=1)
        Drg = torch.pow(mr-mg,2)
        Drb = torch.pow(mr-mb,2)
        Dgb = torch.pow(mb-mg,2)
        k = torch.pow(torch.pow(Drg,2) + torch.pow(Drb,2) + torch.pow(Dgb,2),0.5)

        return k

			
class L_spa(nn.Module):

    def __init__(self):
        super(L_spa, self).__init__()
        # print(1)kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        kernel_left = torch.FloatTensor( [[0,0,0],[-1,1,0],[0,0,0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_right = torch.FloatTensor( [[0,0,0],[0,1,-1],[0,0,0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_up = torch.FloatTensor( [[0,-1,0],[0,1, 0 ],[0,0,0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_down = torch.FloatTensor( [[0,0,0],[0,1, 0],[0,-1,0]]).cuda().unsqueeze(0).unsqueeze(0)
        self.weight_left = nn.Parameter(data=kernel_left, requires_grad=False)
        self.weight_right = nn.Parameter(data=kernel_right, requires_grad=False)
        self.weight_up = nn.Parameter(data=kernel_up, requires_grad=False)
        self.weight_down = nn.Parameter(data=kernel_down, requires_grad=False)
        self.pool = nn.AvgPool2d(4)
    def forward(self, org , enhance ):
        b,c,h,w = org.shape

        org_mean = torch.mean(org,1,keepdim=True)
        enhance_mean = torch.mean(enhance,1,keepdim=True)

        org_pool =  self.pool(org_mean)			
        enhance_pool = self.pool(enhance_mean)	

        weight_diff =torch.max(torch.FloatTensor([1]).cuda() + 10000*torch.min(org_pool - torch.FloatTensor([0.3]).cuda(),torch.FloatTensor([0]).cuda()),torch.FloatTensor([0.5]).cuda())
        E_1 = torch.mul(torch.sign(enhance_pool - torch.FloatTensor([0.5]).cuda()) ,enhance_pool-org_pool)


        D_org_letf = F.conv2d(org_pool , self.weight_left, padding=1)
        D_org_right = F.conv2d(org_pool , self.weight_right, padding=1)
        D_org_up = F.conv2d(org_pool , self.weight_up, padding=1)
        D_org_down = F.conv2d(org_pool , self.weight_down, padding=1)

        D_enhance_letf = F.conv2d(enhance_pool , self.weight_left, padding=1)
        D_enhance_right = F.conv2d(enhance_pool , self.weight_right, padding=1)
        D_enhance_up = F.conv2d(enhance_pool , self.weight_up, padding=1)
        D_enhance_down = F.conv2d(enhance_pool , self.weight_down, padding=1)

        D_left = torch.pow(D_org_letf - D_enhance_letf,2)
        D_right = torch.pow(D_org_right - D_enhance_right,2)
        D_up = torch.pow(D_org_up - D_enhance_up,2)
        D_down = torch.pow(D_org_down - D_enhance_down,2)
        E = (D_left + D_right + D_up +D_down)
        # E = 25*(D_left + D_right + D_up +D_down)

        return E
class L_exp(nn.Module):

    def __init__(self,patch_size):
        super(L_exp, self).__init__()
        # print(1)
        self.pool = nn.AvgPool2d(patch_size)
        # self.mean_val = mean_val
    def forward(self, x, mean_val ):

        b,c,h,w = x.shape
        x = torch.max(x,1,keepdim=True)[0]
        mean = self.pool(x)

        d = torch.mean(torch.pow(mean- torch.FloatTensor([mean_val] ).cuda(),2))
        return d
        
class L_TV(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(L_TV,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h =  (x.size()[2]-1) * x.size()[3]
        count_w = x.size()[2] * (x.size()[3] - 1)
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size
class Sa_Loss(nn.Module):
    def __init__(self):
        super(Sa_Loss, self).__init__()
        # print(1)
    def forward(self, x ):
        # self.grad = np.ones(x.shape,dtype=np.float32)
        b,c,h,w = x.shape
        # x_de = x.cpu().detach().numpy()
        r,g,b = torch.split(x , 1, dim=1)
        mean_rgb = torch.mean(x,[2,3],keepdim=True)
        mr,mg, mb = torch.split(mean_rgb, 1, dim=1)
        Dr = r-mr
        Dg = g-mg
        Db = b-mb
        k =torch.pow( torch.pow(Dr,2) + torch.pow(Db,2) + torch.pow(Dg,2),0.5)
        # print(k)
        

        k = torch.mean(k)
        return k



class Sa_Loss(nn.Module):
    def __init__(self):
        super(Sa_Loss, self).__init__()
        # print(1)
    def forward(self, x ):
        # self.grad = np.ones(x.shape,dtype=np.float32)
        b,c,h,w = x.shape
        # x_de = x.cpu().detach().numpy()
        r,g,b = torch.split(x , 1, dim=1)
        mean_rgb = torch.mean(x,[2,3],keepdim=True)
        mr,mg, mb = torch.split(mean_rgb, 1, dim=1)
        Dr = r-mr
        Dg = g-mg
        Db = b-mb
        k =torch.pow( torch.pow(Dr,2) + torch.pow(Db,2) + torch.pow(Dg,2),0.5)
        # print(k)
        

        k = torch.mean(k)
        return k

class perception_loss(nn.Module):
    def __init__(self):
        super(perception_loss, self).__init__()
        # vgg = vgg16(pretrained=True).cuda()
        features = vgg16(pretrained=True).features
        self.to_relu_1_2 = nn.Sequential() 
        self.to_relu_2_2 = nn.Sequential() 
        self.to_relu_3_3 = nn.Sequential()
        self.to_relu_4_3 = nn.Sequential()

        for x in range(4):
            self.to_relu_1_2.add_module(str(x), features[x])
        for x in range(4, 9):
            self.to_relu_2_2.add_module(str(x), features[x])
        for x in range(9, 16):
            self.to_relu_3_3.add_module(str(x), features[x])
        for x in range(16, 23):
            self.to_relu_4_3.add_module(str(x), features[x])
        
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        h = self.to_relu_1_2(x)
        h_relu_1_2 = h
        h = self.to_relu_2_2(h)
        h_relu_2_2 = h
        h = self.to_relu_3_3(h)
        h_relu_3_3 = h
        h = self.to_relu_4_3(h)
        h_relu_4_3 = h
        # out = (h_relu_1_2, h_relu_2_2, h_relu_3_3, h_relu_4_3)
        return h_relu_4_3
class PSNR(nn.Module):
    def __init__(self, max_val=0):
        super().__init__()

        base10 = torch.log(torch.tensor(10.0))
        max_val = torch.tensor(max_val).float()

        self.register_buffer('base10', base10)
        self.register_buffer('max_val', 20 * torch.log(max_val) / base10)

    def __call__(self, a, b):
        mse = torch.mean((a.float() - b.float()) ** 2)

        if mse == 0:
            return 0

        return 10 * torch.log10((1.0 / mse))
ssim = SSIM()
psnr = PSNR()
def validation(model, val_loader):

    ssim = SSIM()
    psnr = PSNR()
    ssim_list = []
    psnr_list = []
    for i, imgs in enumerate(val_loader):
        with torch.no_grad():
            low_img, high_img,hist,img_name = imgs[0].cuda(), imgs[1].cuda(),imgs[2].cuda(),imgs[3]
            enhanced_image,retouch_image,ill_image = model(low_img,hist)
            # print(enhanced_img.shape)
        ssim_value = ssim(enhanced_image, high_img, as_loss=False).item()
        #ssim_value = ssim(enhanced_img, high_img).item()
        psnr_value = psnr(enhanced_image, high_img).item()
        # print('The %d image SSIM value is %d:' %(i, ssim_value))
        ssim_list.append(ssim_value)
        psnr_list.append(psnr_value)

    SSIM_mean = np.mean(ssim_list)
    PSNR_mean = np.mean(psnr_list)
    print('The SSIM Value is:', SSIM_mean)
    print('The PSNR Value is:', PSNR_mean)
    return SSIM_mean, PSNR_mean