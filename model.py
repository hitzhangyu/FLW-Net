import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import pytorch_ssim


# class CSDN_Tem(nn.Module):
#     def __init__(self, in_ch, out_ch):
#         super(CSDN_Tem, self).__init__()
#         self.depth_conv = nn.Conv2d(
#             in_channels=in_ch,
#             out_channels=in_ch,
#             kernel_size=3,
#             stride=1,
#             padding=1,
#             groups=in_ch
#         )
#         self.point_conv = nn.Conv2d(
#             in_channels=in_ch,
#             out_channels=out_ch,
#             kernel_size=1,
#             stride=1,
#             padding=0,
#             groups=1
#         )

#     def forward(self, input):
#         out = self.depth_conv(input)
#         out = self.point_conv(out)
#         return out

class CSDN_Tem(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(CSDN_Tem, self).__init__()
        self.depth_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=1
        )
        # self.point_conv = nn.Conv2d(
        #     in_channels=in_ch,
        #     out_channels=out_ch,
        #     kernel_size=1,
        #     stride=1,
        #     padding=0,
        #     groups=1
        # )

    def forward(self, input):
        out = self.depth_conv(input)
        # out = self.point_conv(out)
        return out

class CSDN_Temd(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(CSDN_Temd, self).__init__()
        self.depth_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=3,
            stride=1,
            padding=2,
            dilation=2,
            groups=1
        )

    def forward(self, input):
        out = self.depth_conv(input)
        # out = self.point_conv(out)
        return out

class Hist_adjust(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Hist_adjust, self).__init__()
        self.point_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1
        )
    def forward(self, input):
        out = self.point_conv(input)
        return out


class enhance_net_nopool(nn.Module):

	def __init__(self,scale_factor,nbins):
		super(enhance_net_nopool, self).__init__()

		self.relu = nn.LeakyReLU(inplace=True)
		self.scale_factor = scale_factor
		self.nbins = nbins
		self.upsample = nn.UpsamplingBilinear2d(scale_factor=self.scale_factor)
		number_f = 16

#   FLW-Net
		self.e_conv1 = CSDN_Tem(4,number_f) 
		self.e_conv2 = CSDN_Tem(number_f,number_f) 
		self.e_conv3 = CSDN_Tem(number_f+3+1,number_f) 
		self.e_conv4 = CSDN_Temd(number_f,number_f) 
		self.e_conv5 = CSDN_Temd(number_f,number_f) 
		self.e_conv6 = CSDN_Tem(number_f*2,number_f) 
		# self.e_conv6_1 = CSDN_Tem(number_f,number_f) 
		# self.e_conv6_2 = CSDN_Tem(number_f,number_f) 

		self.e_conv7 = CSDN_Tem(number_f,6) 
#   GFE-Net
		self.g_conv1 = Hist_adjust(self.nbins+1,number_f) 
		self.g_conv2 = Hist_adjust(number_f,number_f) 
		self.g_conv3 = Hist_adjust(number_f+self.nbins+1,number_f) 
		self.g_conv4 = Hist_adjust(number_f,number_f) 
		self.g_conv5 = Hist_adjust(number_f,7) 


	def retouch(self, x,x_r):

		x = x + x_r[:,0:1,:,:]*(-torch.pow(x,2)+x)
		x = x + x_r[:,1:2,:,:]*(-torch.pow(x,2)+x)
		x = x + x_r[:,2:3,:,:]*(-torch.pow(x,2)+x)
		x = x + x_r[:,3:4,:,:]*(-torch.pow(x,2)+x)
		x = x + x_r[:,4:5,:,:]*(-torch.pow(x,2)+x)
		x = x + x_r[:,5:6,:,:]*(-torch.pow(x,2)+x)

		enhance_image = x + x_r[:,6:7,:,:]*(-torch.pow(x,2)+x)

		return enhance_image
		
	def forward(self, x, hist):
		# hist[:,-1:,:,:] = 0.5
		x_V = x.max(1,keepdim=True)[0]
		if self.scale_factor==1:
			# x_V_down = torch.mean(x_V,[2,3],keepdim=True)
			x_V_up = torch.mean(x_V,[2,3],keepdim=True)+x_V*0
		else:
			x_V_down = F.interpolate(x_V,scale_factor=1/self.scale_factor, mode='bilinear')
			x_V_up = F.interpolate(x_V_down,scale_factor=self.scale_factor, mode='bilinear')
			# x_V_up = torch.mean(x_V,[2,3],keepdim=True)+x_V*0

		g1 = self.relu(self.g_conv1(hist))
		g2 = self.relu(self.g_conv2(g1))
		g3 = self.relu(self.g_conv3(torch.cat([g2,hist],1)))
		g4 = self.relu(self.g_conv4(g3))
		g5 = self.relu(self.g_conv5(g4))

		retouch_image = self.retouch(x_V,g5)

		x1 = self.relu(self.e_conv1(torch.cat([x-x_V_up/2,x_V_up/2],1)))
		x2 = self.relu(self.e_conv2(x1))
		x3 = self.relu(self.e_conv3(torch.cat([x2,x,retouch_image],1)))
		x4 = self.relu(self.e_conv4(x3))
		x5 = self.relu(self.e_conv5(x4))
		x6 = self.relu(self.e_conv6(torch.cat([x3,x5],1)))
		# x6_1 = self.relu(self.e_conv6_1(x6))
		# x6_2 = self.relu(self.e_conv6_2(x6_1))


		enhance_image = F.softplus(self.e_conv7(x6))

		return enhance_image[:,0:3,:,:],retouch_image,enhance_image[:,3:,:,:]
