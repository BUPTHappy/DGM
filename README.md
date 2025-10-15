# Fast Policy

## 🛠️ Installation
Build and run the Docker container:
```console
$ docker build --platform=linux/amd64 -t user/project:latest .
$ docker run --platform=linux/amd64 -it user/project:latest .
```

Activate the python environment
```console
$ eval "$(micromamba shell hook --shell bash)"
$ micromamba activate fast_policy
```

## Testing
### Model Performance Table

| Dataset Name | Diffusion Head | Accuracy Score | Model Checkpoint | Config File |
|--------------|:--------------:|:--------------:|:----------------:|:----------------:|
| Push-T       | MLP            | 0.99              | [Download](https://drive.google.com/file/d/1yCkthGuVR675N4wlhWXfyzKxCYA6eU11/view?usp=drive_link)              | pusht_MLP.yaml
| Push-T       | DiT-Hybrid     | 0.99              | [Download](https://drive.google.com/file/d/17lXuxlXlyLEzrZE911Vyigff7yZm1c5D/view?usp=drive_link)| pusht_DiT_hybrid.yaml
| LIBERO-10    | MLP            | 0.92              | [Download](https://drive.google.com/file/d/17E7809Xc83JJlknRJbRHXmanQJrM3FFR/view?usp=drive_link)              |libero10_MLP.yaml
| LIBERO-10    | DiT-Hybrid     | 0.96              | [Download](https://drive.google.com/file/d/180_LDdvnrBtthwUOi90s3TYMT7gT9GLb/view?usp=drive_link) |libero10_DiT_hybrid.yaml                |


Run rollouts with:
```
# Push-T MLP
CUDA_VISIBLE_DEVICES=0 python eval_sim.py \
  --checkpoint checkpoints/pusht_MLP.ckpt \
  --output_dir checkpoints/pusht_MLP_rollout/

# Push-T DiT-Hybrid
CUDA_VISIBLE_DEVICES=0 python eval_sim.py \
  --checkpoint checkpoints/pusht_DiT_hybrid.ckpt \
  --output_dir checkpoints/pusht_DiT_hybrid_rollout/

# LIBERO-10 MLP
CUDA_VISIBLE_DEVICES=0 python eval_sim.py \
  --checkpoint checkpoints/libero10_MLP.ckpt \
  --output_dir checkpoints/libero10_MLP_rollout/ \
  --num_sampling_steps 2 \
  --stochasticity_rate 0 \
  --use_ucgm

# LIBERO-10 DiT-Hybrid
CUDA_VISIBLE_DEVICES=0 python eval_sim.py \
  --checkpoint checkpoints/libero10_DiT_hybrid.ckpt \
  --output_dir checkpoints/libero10_DiT_hybrid_rollout \
  --num_sampling_steps 2 \
  --stochasticity_rate 0 \
  --use_ucgm
```

### Testing Inference Speed
Profile the inference speed of the models with the following commands
```
PYTHONPATH=. python3 profiling/time_profiling_image.py  --config-dir=. \
    --config-name=libero10_MLP.yaml \
    model.policy.selected_training_mode=policy_model \
    save_folder="profiling/time_comparison/libero10_MLP.txt"

PYTHONPATH=. python3 profiling/time_profiling_image.py  --config-dir=. \
    --config-name=uva_libero10.yaml \
    model.policy.selected_training_mode=policy_model \
    save_folder="profiling/time_comparison/libero10_baseline.txt"

PYTHONPATH=. python3 profiling/time_profiling_image.py  --config-dir=. \
    --config-name=libero10_DiT_hybrid.yaml \
    model.policy.selected_training_mode=policy_model \
    save_folder="profiling/time_comparison/libero10_DiT_hybrid.txt"
    
```

#### Inference Speed Comparison

The following table summarizes the inference speed (in milliseconds) of different model components, measured on a single Tesla V100-SXM2-32GB GPU:

| Component            | Baseline<br>(100 steps) | Ours MLP<br>(2 steps) | Ours DiT Hybrid<br>(2 steps) |
|----------------------|:----------------------:|:---------------------:|:----------------------------:|
| **VAE Image Encoder**|          43            |         43            |            43                |
| **Transformer**      |          44            |         44            |            44                |
| **Action Diffusion** |         261            |          5            |             9                |
| **Total**            |         348            |         92            |            96                |

*Values in ms. Our models achieve significant speedup in the Action Diffusion component compared to the baseline.*

### Measuring Computational Complexity

```
PYTHONPATH=. python3 profiling/profile_flops.py  --config-dir=. \
    --config-name=libero10_MLP.yaml \
    model.policy.selected_training_mode=policy_model \
    save_folder="profiling/flops_comparison/libero10_MLP.txt"

PYTHONPATH=. python3 profiling/profile_flops.py  --config-dir=. \
    --config-name=uva_libero10.yaml \
    model.policy.selected_training_mode=policy_model \
    save_folder="profiling/flops_comparison/libero10_baseline.txt"
    
PYTHONPATH=. python3 profiling/profile_flops.py  --config-dir=. \
    --config-name=libero10_DiT_hybrid.yaml \
    model.policy.selected_training_mode=policy_model \
    save_folder="profiling/flops_comparison/libero10_DiT_hybrid.txt"
```

#### FLOPs comparison
| Component             | Baseline<br>(100 steps) | Ours MLP<br>(2 steps) | Ours DiT Hybrid<br>(2 steps) |
|-----------------------|:-----------------------:|:---------------------:|:----------------------------:|
| **VAE Image Encoder** |          553            |          553          |             553              |
| **Transformer**       |          374            |          374          |             374              |
| **Action Diffusion**  |          261            |           13          |              14              |
| **Total**             |         1188            |          940          |             941              |

*Values in GFLOPs (\(10^9\) FLOPs). Our models achieve significant savings in the Action Diffusion component compared to the baseline.*


## Training
#### Download and Prepare Datasets
* [PushT](https://diffusion-policy.cs.columbia.edu/data/training/pusht.zip) from [Diffusion Policy](https://github.com/real-stanford/diffusion_policy).
* [Libero10](https://utexas.box.com/shared/static/cv73j8zschq8auh9npzt876fdc1akvmk.zip) from [LIBERO](https://libero-project.github.io/main.html). We replayed the data to extract the absolute actions and appended language tokens from [CLIP](https://openai.com/index/clip/) using `AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")`. Download both the original [hdf5 file](https://drive.google.com/file/d/1_6Kc7e-s30MblbX8YjpxSofe9ZRPk3xv/view?usp=sharing) and the converted [dataset](https://drive.google.com/file/d/1cPU2RVAvtukyapcWly8zP1y-dlOEF2ko/view?usp=sharing). Then, extract their contents and place them in the `data` folder.

#### Download Pretrained Models
Run the following command to download the pretrained VAE.

```console
$ python unified_video_action/utils/download.py
```

We use the pretrained transformer backbone of Unified Video Action Model.
Download the checkpoints from [ here for PushT](https://drive.google.com/file/d/1OduHcxfc2hqUYSccMQNf9g-vAt-q2UhF/view?usp=sharing) and  [ here for Libero10](https://drive.google.com/file/d/11c2VrmaRp48yw__5A5xpcu8EPzkexHSi/view?usp=sharing)



#### Train the model
```
 yes n | accelerate launch --num_processes=4 train.py \
    --config-dir=. \
    --config-name=pusht_MLP.yaml \
    model.policy.selected_training_mode=policy_model \  
    logging.project=pusht_MLP \
    hydra.run.dir="checkpoints/pusht"
```
