# Memory-Efficient Video Compressive Sensing Using Cross-Frame Attention Network

This is a project of course Computational Imaging of Westlake University. This project seek to train a end-to-end snapshot compressive imaging (SCI) system to recover RGB videos from raw camera measurements.
More specifically, we train a UNet contains our customized cross-frame attention machenism. The network takes a coarse estimation as input and predict the reconstruction of video frames.

## Quickstart

### Setup environment

1. Install Pytorch

This project is experimented on Pytorch-2.0, please refer to [Pytorch's official webpage](https://pytorch.org/) for installation.

2. Install dependency packages

```bash
git clone https://github.com/lizhiqi49/cfattn-sci
cd cfattn-sci
pip install -r requirements.txt
```

### Setup dataset

The raw data we used is the same as [this repository](https://github.com/BoChenGroup/RevSCI-net). 
When the raw data is downloaded, please use script `data/data_gen.py` to generate your training data. 

### Start training

1. Configure hyper-parameters

File `unet_config.json` is used to configure the architecture of our model, which is a U-Net. 
And the training config files of two training stages are under directory `configs`.
You can also configure your own training hyper-parameters under `configs/{exp_name}.yaml`.

2. Configure Accelerate

This project uses library [Accelerate](https://github.com/huggingface/accelerate) for mixed-precision and distributed training, before training start, you need configure your accelerate using `accelerate config` on your shell. 

3. Train!

For training stage 1:
```
accelerate launch train.py --config configs/sci_stage1.yaml
```

For training stage 2:
```
accelerate launch finetune.py --config configs/sci_stage2.yaml
```


4. Evaluation

Put your test frames in a directory and name those image files using single number, for example, '5.png'.
And test the model using:

```
python test.py \
--pretrained_unet_path {your_pretrained_model_dir} \
--test_video_dir {your_test_data_dir} \
--use_cross_frame_attn      # remove this flag if do not want to use cf-attn
```

This command will save the reconstructed frames in an image `./test_result.png`.
If you want to save video, please add `--save_video` flag.