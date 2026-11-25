## Requirements and Installation
We recommended the following dependencies.
*  Python 3.6
*  PyTorch 1.4.0
*  tqdm 4.43.0
*  munch 2.5.0
*  torchvision 0.5.0



## Preparing Data 
The datasets analyzed during the current study are publicly available. The MIT-Adobe FiveK dataset is available at https://data.csail.mit.edu/graphics/fivek/. The LOL dataset can be accessed via https://daooshee.github.io/BMVC2018website/. The LOLv2 dataset is available at https://github.com/flyywh/CVPR-2020-Semi-Low-Light. The SICE dataset is accessible at https://github.com/csjcai/SICE.

### Getting the MIT-Adobe FiveK Dataset
 - Download the dataset from https://data.csail.mit.edu/graphics/fivek/. (~50GB, SHA1)
 - Extract the data.
 - Open `fivek.lrcat` with Lightroom. Just click "upgrade" if Lightroom asks you to upgrade.

### Generating the Low-quality Images(MIT)
 - Import the FiveK dataset into Adobe Lightroom.
 - In the `Collections` list (bottom left), select collection **`Inputs/InputAsShotZeroed`**.
 - Export all images in the following settings:
   - Select all images at the bottom or in the middle (select one and press `Ctrl-A`), right-click any of them and select `Export/Export...`. 
   - Export Location: `Export to`=`Specific folder`, `Folder`=`Your folder for low-quality images`.
   - File Settings: `Image Format`=`PNG`, `Color Space`=`sRGB`, `Bit Depth`=`8 bit/component`
   - Image Sizing: `Resize to Fit`=`Short Edge`, select `Don't Enlarge`, Fill in `512 pixels`, `Resolution` doesn't matter to ignort it.
   - Finally, click `Export`.

### Generating the High-quality Images(MIT)
 - Import the FiveK dataset into Adobe Lightroom.
 - In the `Collections` list (bottom left), select collection **`Experts/C`**.
 - Export all images in the following settings:
   - Select all images at the bottom or in the middle (select one and press `Ctrl-A`), right-click any of them and select `Export/Export...`. 
   - Export Location: `Export to`=`Specific folder`, `Folder`=`Your folder for high-quality images`.
   - File Settings: `Image Format`=`PNG`, `Color Space`=`sRGB`, `Bit Depth`=`8 bit/component`
   - Image Sizing: `Resize to Fit`=`Short Edge`, select `Don't Enlarge`, Fill in `512 pixels`, `Resolution` doesn't matter to ignort it.
   - Finally, click `Export`.


## Testing
```
python main.py --mode test --version DWRGAN-FiveK --pretrained_model 92 --is_test_nima True --is_test_psnr_ssim True
```

## Training
Prepare the training, testing, and validation data. The folder structure should be:
```
data
└─── fiveK
	├─── train
	|	├─── exp
	|	|	├──── a1.png                  
	|	|	└──── ......
	|	└─── raw
	|		├──── b1.png                  
	|		└──── ......
	├─── val
	|	├─── label
	|	|	├──── c1.png                  
	|	|	└──── ......
	|	└─── raw
	|		├──── c1.png                  
	|		└──── ......
	└─── test
		├─── label
		| 	├──── d1.png                  
		| 	└──── ......
		└─── raw
			├──── d1.png                  
			└──── ......
```
```raw/```contains low-quality images, ```exp/``` contains unpaired high-quality images, and ```label/``` contains corresponding ground truth.

To train DWRGAN on FiveK, run the training script below.
```
python main.py --mode train --version DWRGAN-FiveK --use_tensorboard True --is_test_nima True --is_test_psnr_ssim True
```

This script will create a folder named ```./results``` in which the resulting are saved. 
- The PSNR results will be saved to here: ```./results/psnr_val_results``` (including PSNR for each valiaded epoch and the summary)
- The SSIM results will be saved to here: ```./results/ssim_val_results``` (including SSIM for each valiaded epoch and the summary)
- The NIMA results will be saved to here: ```./results/nima_val_results``` (including NIMA for each valiaded epoch and the summary)
- The training logs will be saved to here: ```./results/DWRGAN-FiveK/logs```
- The models will be saved to here: ```./results/DWRGAN-FiveK/models```
- The intermediate results will be saved to here: ```./results/DWRGAN-FiveK/samples```
- The validation results will be saved to here: ```./results/DWRGAN-FiveK/validation```
- The test results will be saved to here: ```./results/DWRGAN-FiveK/test```


To view training results and loss plots, run ```tensorboard --logdir=results/DWRGAN-FiveK/logs```, and click the URL accordingly (For example, http://nzk-ub:6007/).

The summary of PSNR test results will be save to ```./results/psnr_val_results/PSNR_total_results_epoch_avgpsnr.csv```. Find the best epoch in the last line of ```PSNR_total_results_epoch_avgpsnr.csv```.




