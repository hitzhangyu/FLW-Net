import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import os
import sys
import argparse
import time
import dataloader
import model
import Myloss
import numpy as np
from torchvision import transforms
from Myloss import validation


PSNR_mean = 0
SSIM_mean = 0


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def train(config):

	os.environ['CUDA_VISIBLE_DEVICES']='0'
	scale_factor = config.scale_factor
	FLW_net = model.enhance_net_nopool(scale_factor,config.nbins).cuda()

	# FLW_net.apply(weights_init)
	if config.load_pretrain == True:
	    FLW_net.load_state_dict(torch.load(config.pretrain_dir))
	train_dataset = dataloader.MemoryFriendlyLoader_zy4(low_img_dir = config.lowlight_images_path,\
				high_img_dir = config.highlight_images_path,task=config.task,batch_w=config.patch_size,batch_h=config.patch_size,\
				nbins = config.nbins,exp_mean=config.exp_mean)		
	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=True)


	val_dataset = dataloader.MemoryFriendlyLoader_zy5(low_img_dir = config.val_lowlight_images_path,\
				high_img_dir = config.val_highlight_images_path,task=config.task,batch_w=50,batch_h=config.patch_size,\
				nbins = config.nbins,exp_mean=config.exp_mean)	

	val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=config.num_workers, pin_memory=True)


	L_color_zy = Myloss.L_color_zy()
	L_spa = Myloss.L_spa()
	L_exp = Myloss.L_exp(16)
	L_TV = Myloss.L_TV()

	L_grad_cosist = Myloss.L_grad_cosist()
	L_bright_cosist = Myloss.L_bright_cosist()
	L_recon = Myloss.L_recon()
	L_retouch_mean = Myloss.L_retouch_mean()
	L_smooth4 = Myloss.L_smooth4()
	L_smooth_ill = Myloss.L_smooth_ill()
	L_recon_low = Myloss.L_recon_low()

	optimizer = torch.optim.Adam(FLW_net.parameters(), lr=config.lr, weight_decay=config.weight_decay)
	
	FLW_net.train()
	loss = 0
	ssim_high = 0
	psnr_high = 0

	for epoch in range(config.num_epochs):
		for iteration, (img_lowlight,img_highlight,hist,img_name) in enumerate(train_loader):

			img_lowlight = img_lowlight.cuda()
			img_highlight = img_highlight.cuda()
			hist = hist.cuda()

			E = 0.5

			enhanced_image,retouch_image,ill_image  = FLW_net(img_lowlight,hist)
			Loss_1 = L_grad_cosist(enhanced_image,img_highlight)
			Loss_6 = L_bright_cosist(enhanced_image,img_highlight)
			loss_2, Loss_ssim = L_recon(enhanced_image,img_highlight)
			loss_3 = L_retouch_mean(retouch_image,img_highlight)
			loss_col = torch.mean(L_color_zy(enhanced_image,img_highlight))

			loss_5 = L_recon_low(enhanced_image,img_lowlight,ill_image)
			# Loss_4 = L_smooth4(enhanced_image)+L_smooth4(enhanced_image[:,0:1,:,:])+L_smooth4(enhanced_image[:,2:3,:,:])+L_smooth4(enhanced_image[:,1:2,:,:]) + L_smooth4(retouch_image)
			Loss_ill = L_smooth_ill(ill_image,enhanced_image)+L_smooth_ill(ill_image[:,0:1,:,:],enhanced_image[:,0:1,:,:])+L_smooth_ill(ill_image[:,2:3,:,:],enhanced_image[:,2:3,:,:])+L_smooth_ill(ill_image[:,1:2,:,:],enhanced_image[:,1:2,:,:]) 
			Loss_TV = L_TV(retouch_image)
			# loss_spa = torch.mean(L_spa(enhanced_image, img_highlight))
			loss_exp = torch.mean(L_exp(enhanced_image,E))

			loss_unsupervised = loss_exp  #+ Loss_TV  + loss_spa 
			loss_supervised = Loss_ssim + loss_2+Loss_1 +Loss_6 + loss_col #+ 0.25* Loss_ill +loss_5 +loss_3  + 

			loss = loss_supervised #+ loss_unsupervised
			# best_loss
			
			optimizer.zero_grad()
			loss.backward()
			# torch.nn.utils.clip_grad_norm(FLW_net.parameters(),config.grad_clip_norm)
			optimizer.step()

			# if ((iteration+1) % config.display_iter) == 0:
			# 	print("Loss at iteration", iteration+1, ":", Loss_ssim.item())

			if ((iteration+1) % config.display_iter) == 0:
				if ((epoch+1)%config.snapshot_iter) ==0:
					torchvision.utils.save_image(torch.concat([enhanced_image[0],ill_image[0]],axis=2), config.sample_dir+str(epoch) + '.png')
					# torch.save(FLW_net.state_dict(), config.snapshots_folder + "Epoch" + str(epoch) + '.pth') 		
					FLW_net.eval()
					PSNR_mean, SSIM_mean = validation(FLW_net, val_loader)
					if SSIM_mean > ssim_high:
						ssim_high = SSIM_mean
						print('the highest SSIM value is:', str(ssim_high))
						torch.save(FLW_net.state_dict(), os.path.join(config.snapshots_folder, "best_Epoch" + '.pth'))
					with open(config.snapshots_folder + 'log.txt', 'a+') as f:
						f.write('epoch' + str(epoch) + ':' + 'the SSIM is' + str(SSIM_mean) + 'the PSNR is' + str(PSNR_mean) + '\n')

if __name__ == "__main__":

	parser = argparse.ArgumentParser()

	# Input Parameters
	parser.add_argument('--lowlight_images_path', type=str, default="data/Train/low/")
	parser.add_argument('--highlight_images_path', type=str, default="data/Train/high/")
	parser.add_argument('--val_lowlight_images_path', type=str, default="data/Test/low/")
	parser.add_argument('--val_highlight_images_path', type=str, default="data/Test/high/")
	

	parser.add_argument('--task', type=str, default="train")
	parser.add_argument('--nbins', type=int, default=14)
	parser.add_argument('--patch_size', type=int, default=100)
	parser.add_argument('--exp_mean', type=float, default=0.55)
	parser.add_argument('--sample_dir', type=str, default="./sample/")

	parser.add_argument('--lr', type=float, default=0.0001)
	parser.add_argument('--weight_decay', type=float, default=0.0001)
	parser.add_argument('--grad_clip_norm', type=float, default=0.1)
	parser.add_argument('--num_epochs', type=int, default=30000)
	parser.add_argument('--train_batch_size', type=int, default=171)
	parser.add_argument('--val_batch_size', type=int, default=16)
	parser.add_argument('--num_workers', type=int, default=0)
	parser.add_argument('--display_iter', type=int, default=2)
	parser.add_argument('--snapshot_iter', type=int, default=20)
	parser.add_argument('--scale_factor', type=int, default=20)
	parser.add_argument('--snapshots_folder', type=str, default="snapshots_Flwnet/")
	parser.add_argument('--load_pretrain', type=bool, default= False)
	parser.add_argument('--pretrain_dir', type=str, default= "snapshots_Flwnet/best_Epoch.pth")

	config = parser.parse_args()

	if not os.path.exists(config.snapshots_folder):
		os.mkdir(config.snapshots_folder)


	train(config)








	
