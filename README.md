Tested in Windows; In the Linux environment, it may be necessary to modify the image loading method to avoid the mismatch of input and reference（Thx [2665207323](https://github.com/hitzhangyu/FLW-Net/issues/10) ）.

torch = 1.13.0

Test Command: python lowlight_test.py
(Chage the "filePath" and "filePath_high" to your own data path)


Train Command: python lowlight_train.py 
(Chage the "lowlight_images_path","highlight_images_path", "val_lowlight_images_path", and "val_highlight_images_path" to your own data path)

====================================================================================
new checkpoint in training：Looks like an interesting color change！！

<<==<<==<<==<<==<<==<<==<<==<<==<<==<<==<<==<<==<<==<<==<<==<<==<<==<<==<<==<<==<<==
![image](https://github.com/hitzhangyu/FLW-Net/assets/30136020/a4614d6f-52dc-44ac-a9a0-5426111d12f8)
![image](https://github.com/hitzhangyu/FLW-Net/assets/30136020/c09511cc-11bd-40ee-b2a5-5de4aaf6849b)
![999_3](https://github.com/hitzhangyu/FLW-Net/assets/30136020/cd9ce204-768d-4a08-9df4-2baafa72fb75)
![1099_5](https://github.com/hitzhangyu/FLW-Net/assets/30136020/b9d40da9-ff48-47e1-a6a4-98d4f7641d75)
![208](https://github.com/hitzhangyu/FLW-Net/assets/30136020/116dc8f5-011d-4eff-b783-130d2ba7306a)
![155](https://github.com/hitzhangyu/FLW-Net/assets/30136020/f6d681f5-e67c-47a0-aa55-398db06ece54)

