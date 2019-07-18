# One Shot Image-to-Image Translation
We discuss how a deep learning model may tackle the one-shot learning scenario on image translation. We propose a two-step training strategy to solve the blurry image result due to the lack of training data. Furthermore, we successfully get the group- ing information when extracting features.

## Environment
Tensorflow 1.8.0,
Conda 4.6.14

![Alt Network Structure](./img/OHG_structure.pdf)

## Installation
### Clone One Shot Image-to-Image Translation
Clone this repository to your own machine
```
git clone https://github.com/YiChuanChang/One-shot-Image-to-Image-Translation.git
cd cascaded-pix2pix---Chinese-handwriting-generator-
```
## Prepare data

There are two training steps. For, first step we use COCO datasets. For second step, you can choose the dataset you like.

You can download the COCO dataset here.
You can download other datasets I used here.

Prepare the data as the following structure:
```text
  dataset
  ├── coco
      ├── trainA
          └── png 
      └── trainB
          └── png
  └── the dataset you want
      ├── testA
          └── png 
      ├── testB
          └── png   		
      ├── trainA
          └── png (one image)
      └── trainB
          └── png (one image)
```

## Train model
```
python main.py --phase train
```
| parameters        | Dafault Value          | description|
| :------------     |:-------------          | :-----|
| --phase 		   	| 'train'  		        | train or test|
| --exp_name 			| 'ZSIT_train'           | folder to save all stuff|
| --checkpoint_dir 	| 'checkpoint'           | Directory name to save the |checkpoints
| --result_dir 		| 'results'              | Directory name to save the generated images|
| --log_dir 			| 'logs'                 | Directory name to save training logs|
| --sample_dir 		| 'samples'              | Directory name to save the samples on training|
| --datasets        | ['coco', 'cityscapes'] | dataset_name second one is what you want to train|
| --augment_flag    | True    			     | Image augmentation use or not |
| --epoch           | 25					     | The total number of training epoch|
| --iteration       | 1000      			     | The number of training iterations|
| --batch_size      | 1                      | The batch size|
| --print_freq      | 50                     | The number of image_print_freq|
| --save_freq       | 50                     | The number of ckpt_save_freq|
| --lr              | 0.001                  | The learning rate|
| --img_h           | 256                    | The size of image hegiht|
| --img_w  			| 256                    | The size of image width|
| --img_c  			| 3                      | The size of image channel|
| --gan_w  			| 1.0                    | weight of adversarial loss|
| --con_w  			| 3.0                    | weight of content reconstruction loss|
| --sty_w  			| 1.0                    | weight of style reconstruction loss|
| --rec_w  			| 1.0                    | weight of image reconstruction loss|
| --per_w  			| 0.0                    | weight of perceptual loss|
| --ch  				| 64                     | base channel number per layer|
| --n_cont_res  		| 4                      | number of residual blocks in content encoder|
| --n_cont_downsample  | 2    	            | number of residual blocks in content encoder|
| --n_style_res 	   | 4      		            | number of residual blocks in style encoder|
| --n_style_downsample  | 2                  | number of residual blocks in style encoder|
| --n_upsample  		| 2                      | number of residual blocks in decoder|
| --useUNet  			| True                   | use UNet or not|
| --n_scale  			| 3                      | number of scales of discriminator|
| --n_dis  			| 4                      | number of discriminator layer|
| --sn  				| True                   | use spetral norm or not|

## Test
```
python main.py --phase test 
```












|||