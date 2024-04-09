# Chop & Learn

### [Project Page](https://chopnlearn.github.io/) | [Paper](https://arxiv.org/pdf/2309.14339)

Official Pytorch implementation of the paper:

**Chop & Learn: Recognizing and Generating Object-State Compositions, ICCV 2023**<br>
[Nirat Saini<sup>*</sup>](https://www.cs.umd.edu/~nirat/),
[Hanyu Wang<sup>*</sup>](https://hywang66.github.io/),
[Archana Swaminathan](https://archana1998.github.io/),
[Vinoj Jayasundara](https://vinojjayasundara.github.io/),
[Bo He](https://boheumd.github.io/),
[Kamal Gupta](https://kampta.github.io/),
[Abhinav Shrivastava](https://www.cs.umd.edu/~abhinav/)

## Environment Setup

Please create a new conda environment and install the required packages, and activate the environment using the following commands:

```bash
conda env create -f chopnlearn.yml
conda activate chopnlearn
```

If this is the first time you are using the `accelerate` library, you may need to configure the library by running the following command:

```bash
accelerate config
```

## Data Preparation
Please download the Chop & Learn image dataset from [here](https://drive.google.com/drive/folders/1QylDeUJ8h-CjLRJ8Z9bsdCoQ2uMs59W_) and extract the `image_dataset` folder in the root directory of the repository. The dataset contains the following files:
- `image_dataset/images`: contains the images
- `image_dataset/train_split.json`: contains the training split
- `image_dataset/test_split.json`: contains the test split


## Compositional Image Generation

Unless otherwise mentioned, all the commands in this section should be run from the `generation` directory, and all the relative paths mentioned in this section are relative to the `generation` directory.
If you are in the root directory of this repository, simply:

```bash
cd generation # from the root directory
```

If you want to download all pre-trained checkpoints for the compositional image generation models, as well as the object/state classifier, you can run the following command:
```bash
bash download_checkpoints.sh --all
```
Then you can skip all the checkpoint downloading steps in the following sections.


### Training
To train the compositional image generation model [SD+TI+FT](https://chopnlearn.github.io/#:~:text=Inversion%20%2B%20Fine%2Dtuning%20(-,SD%20%2B%20TI%20%2B%20FT,-)), run the following command:

```bash
accelerate launch train.py \
    --data ../image_dataset/images \
    --train_split_json ../image_dataset/train_split.json \
    --val_split_json ../image_dataset/test_split.json \
    --learning_rate=5e-6 --lr_warmup_steps=500  \
    --learning_rate_textual_inversion=3e-3 \
    --max_train_steps=16000 \
    --train_batch_size=4 --save_steps=400 \
    --exp_name=default 
```

The trained model checkpoints and logs will be saved in the `logs/default` directory.

To disable the textual inversion loss, i.e., to reproduce the [SD+FT](https://chopnlearn.github.io/#:~:text=Stable%20Diffusion%20%2B%20Fine%2Dtuning) model, simply set `--learning_rate_textual_inversion=0` but `--learning_rate=5e-6`.

Similarly, to only do the textual inversion, i.e., to reproduce the [SD+TI](https://chopnlearn.github.io/#:~:text=Diffusion%20%2B%20Textual%20Inversion%20(-,SD%20%2B%20TI,-)) model, set `--learning_rate=0` but `--learning_rate_textual_inversion=3e-3`.

### Sampling

If you haven't, please download the pre-trained compositional image generation model checkpoints and extract them using the following command:

```bash
bash download_checkpoints.sh SD+TI+FT # or SD+TI or SD+FT
```
We provide the pre-trained checkpoints for `SD+TI+FT`, `SD+TI`, and `SD+FT` models, and we use the `SD+TI+FT` model as the example in this section.
`SD+TI+FT`, `SD+TI`, and `SD+FT` are models with textual inversion and fine-tuning, textual inversion only, and fine-tuning only, respectively.

You can also find the urls for the checkpoints in the `download_checkpoints.sh` file and download them manually. Remember to extract the downloaded checkpoints and place them in the `./checkpoints` directory.


To sample from the 'SD+TI+FT' model, run the following command:

```bash
python sample.py \
    --ckpt_path ./checkpoints/SD+FT+TI \
    --data ../image_dataset/images \
    --split ../image_dataset/test_split.json \
    --out_dir ./samples/SD+FT+TI \
    --num_images_per_prompt 20 \
    --batch_size 4 --ti
```

The generated samples will be saved in the `./samples/SD+FT+TI` directory. Note that the `--ti` flag should only be used when sampling from the model with textual inversion. If you are sampling from the `SD+FT` model, do not use the `--ti` flag.

### Evaluation

In this section, we provide the code to evaluate the compositional image generation models using two automatic metrics mentioned in the Section 4.2 of our [paper](https://arxiv.org/pdf/2309.14339): **Patch FID** and **Object/State Accuracy using a Classifier**.

#### Patch FID

To compute the Patch FID score, run the following command:

```bash
python evaluation/calculate_fid.py \
    --path_gen ./samples/SD+FT+TI \
    --path_gt ../image_dataset/images \
    --split ../image_dataset/test_split.json \
    --tmp_dir /dev/shm/chopnlearn \
    --fid_report_path ./evaluation/fid_report.csv
```

This will compute the Patch FID score between the generated images saved in `./samples/SD+FT+TI` and the ground truth images in `../image_dataset/images` using the test split defined in `../image_dataset/test_split.json`. The computed FID scores will be saved in the `./evaluation/fid_report.csv` file.

#### Object/State Accuracy

The automatic object/state accuracy evaluation requires a pre-trained object/state classifier. We provide the pre-trained classifier checkpoint for the object/state classification task. 
<!-- Please download the checkpoint from [here](), extract it, and place it in the `./checkpoints` directory.  -->
If you haven't, please download the pre-trained object/state classifier checkpoint and extract it using the following command:

```bash
bash download_checkpoints.sh classifier
```

The extracted checkpoint is named as `object_state_classifier_checkpoint.ckpt`.

To compute the object/state accuracy, run the following command:

```bash
python evaluation/evaluate_accuracy.py \
    --classifier_ckpt_path ./checkpoints/object_state_classifier_checkpoint.ckpt \
    --path_gen ./samples/SD+FT+TI \
    --data ../image_dataset/images \
    --split ../image_dataset/test_split.json \
    --output_dir ./evaluation/accuracy_results
```

This will compute the object/state accuracy between the generated images saved in `./samples/SD+FT+TI` and the ground truth images in `../image_dataset/images` using the test split defined in `../image_dataset/test_split.json`. The computed accuracy results will be saved in the `./evaluation/accuracy_results` directory. The following two files will be saved in the output directory:
- `SD+FT+TI_acc.json`: contains the object/state accuracy results
- `SD+FT+TI_details.csv`: contains the detailed object/state labels (used in generation) and predictions for each image.

## Compositional Image Recognition

Under construction.


## Citation

If you find this code useful in your research, please consider citing:

```
@inproceedings{saini2023chop,
  title={Chop \& learn: Recognizing and generating object-state compositions},
  author={Saini, Nirat and Wang, Hanyu and Swaminathan, Archana and Jayasundara, Vinoj and He, Bo and Gupta, Kamal and Shrivastava, Abhinav},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={20247--20258},
  year={2023}
}
```