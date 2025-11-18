# DGM: Dynamic Generative Modeling for Visuomotor Policy Learning

![DGM Overview](overview.png)

## Abstract

Learning visuomotor policies that can both perceive and act efficiently in the physical world remains a central challenge in embodied AI. Diffusion-based policies produce high-fidelity actions but suffer from slow iterative inference, whereas consistency-based policies enable real-time control but degrade in precision. We present **Dynamic Generative Modeling (DGM)**, a framework that unifies these approaches by treating their behaviors as endpoints of a continuous generative spectrum, which simultaneously achieves high performance and fast inference. Rather than fixing this generative regime a priori, we develop DGM-Opt, a bi-level optimization algorithm that jointly learns policy weights and generative hyperparameters. DGM-Opt alternates between visuomotor policy optimization and Bayesian search over the generative regime, allowing the policy and its underlying generative dynamics to co-evolve and automatically converge to the task-optimal setting. Meanwhile, DGM enhances a Transformer-based visuomotor backbone with frame-wise local causal attention to maintain spatio-temporal coherence essential for embodied motion. On **PushT** and **Libero10**, our approach achieves **99.6%** and **95%** success rates using only two inference steps, outperforming diffusion baselines by $7\times$ in speed while preserving superior performance. These results demonstrate that adaptive DGM offers a scalable route to real-time embodied intelligence.

---

## Installation

##### Install the conda environment:

```console
conda install mamba -c conda-forge
amba env create -f dgm_environment.yml
eval "$(mamba shell hook --shell bash)"
source ~/.bashrc
```

##### Activate the environment:

```console
conda activate dgm
```

---

## Experiments

### Testing

Download the pretrained checkpoints from the following links and put them in the `checkpoints/` folder.

| Task     | Architecture | Success Rate | Speed(ms) | Checkpoint                                                                                         |
| -------- | ------------ | ------------ | --------- | -------------------------------------------------------------------------------------------------- |
| PushT    | UVA baseline | 0.96         | 166.79    | [PushT_uva][https://drive.google.com/file/d/1OduHcxfc2hqUYSccMQNf9g-vAt-q2UhF/view?usp=sharing]    |
| PushT    | DGM          | **0.99**     | **24.61** | [PushT_dgm][https://drive.google.com/file/d/1-u2eSRnEvFNrqvSmr1JFuU4KVaIsbSFN/view?usp=sharing]    |
| Libero10 | UVA baseline | 0.90         | 180.09    | [Libero10_uva][https://drive.google.com/file/d/11c2VrmaRp48yw__5A5xpcu8EPzkexHSi/view?usp=sharing] |
| Libero10 | DGM          | **0.95**     | **31.42** | [Libero10_dgm][https://drive.google.com/file/d/1FGH87MkvBrBEAOWK1uMjb1MW4TBVVGsR/view?usp=sharing] |

```bash
mkdir checkpoints
pip install gdown
gdown 1-u2eSRnEvFNrqvSmr1JFuU4KVaIsbSFN -O checkpoints/pusht_dgm.ckpt
gdown 1FGH87MkvBrBEAOWK1uMjb1MW4TBVVGsR -O checkpoints/libero10_dgm.ckpt
```

#### Run Evaluation

##### PushT

```bash
CUDA_VISIBLE_DEVICES=0 python eval_sim.py --checkpoint checkpoints/pusht_dgm.ckpt --output_dir checkpoints/pusht_dgmo/ --num_sampling_steps 2
```

##### Libero10

```bash
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl python eval_sim.py \
      --checkpoint checkpoints/libero10_dgm.ckpt \
      --output_dir checkpoints/libero10_dgm/ \
      --dataset_path data/libero_10/
```

### Training

DGM is trained on top of the pretrained UVA model. During training, we freeze all components of the UVA backbone (including the VAE encoder/decoder, autoregressive Transformer, and video generation head) and only train the diffusion head for action prediction. This strategy allows DGM to leverage the rich visual representations learned by UVA while adapting the action generation mechanism to the dynamic generative modeling framework.

#### Download Pretrained Models

We start from a pretrained VAE model and a pretrained image generation model [MAR](https://github.com/LTH14/mar). Run the following command to download the pretrained models.

```bash
python unified_video_action/utils/download.py
```

Additionally, you need to download the pretrained UVA checkpoints for the specific tasks:

```bash
mkdir checkpoints
pip install gdown
# Download UVA PushT checkpoint
gdown 1OduHcxfc2hqUYSccMQNf9g-vAt-q2UhF -O checkpoints/pusht.ckpt
# Download UVA Libero10 checkpoint
gdown 11c2VrmaRp48yw__5A5xpcu8EPzkexHSi -O checkpoints/libero10.ckpt
```

#### Executing Training

##### PushT

```bash
accelerate launch --num_processes=4 train.py \
    --config-dir=. \
    --config-name=dgm_pusht.yaml \
    model.policy.selected_training_mode=policy_model \
    logging.project=pusht_dgm \
    hydra.run.dir="checkpoints/pusht_dgm"
```

##### Libero10

```bash
accelerate launch --num_processes=4 train.py \
    --config-name=dgm_libero10.yaml \
    +ignore_payload_cfg=true \
    model.policy.selected_training_mode=policy_model \
    model.policy.autoregressive_model_params.use_ucgm=true \
    model.policy.autoregressive_model_params.num_sampling_steps=2 \
    task.dataset.dataset_path=data/libero_10  \
    task.dataset.use_cache=true \
    logging.project=libero10_dgm \
    hydra.run.dir="checkpoints/libero10_dgm"
```

---

## Dataset

For these two simulation datasets `PushT` and `Libero10`. Download the datasets and put them in the `data` folder following steps below:

#### PushT

- [PushT](https://diffusion-policy.cs.columbia.edu/data/training/pusht.zip) from [Diffusion Policy](https://github.com/real-stanford/diffusion_policy).

- Prepare `PushT`:

  ```bash
  cd data/
  wget https://diffusion-policy.cs.columbia.edu/data/training/pusht.zip
  unzip pusht.zip
  ```

#### Libero10

- [Libero10](https://utexas.box.com/shared/static/cv73j8zschq8auh9npzt876fdc1akvmk.zip) from [LIBERO](https://libero-project.github.io/main.html). We followed UVA replaying the data to extract the absolute actions and appended language tokens from [CLIP](https://openai.com/index/clip/) using `AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")`. Download both the original [hdf5 file](https://drive.google.com/file/d/1_6Kc7e-s30MblbX8YjpxSofe9ZRPk3xv/view?usp=sharing) and the converted [dataset](https://drive.google.com/file/d/1cPU2RVAvtukyapcWly8zP1y-dlOEF2ko/view?usp=sharing). Then, extract their contents and place them in the `data` folder.

- Prepare `Libero10`:

  ```bash
  cd data/
  gdown https://drive.google.com/uc?id=1_6Kc7e-s30MblbX8YjpxSofe9ZRPk3xv
  gdown https://drive.google.com/uc?id=1cPU2RVAvtukyapcWly8zP1y-dlOEF2ko
  unzip libero10.zip
  cd ../..
  git clone https://github.com/Lifelong-robot-learning/LIBERO.git
  ```

---

## License

This repository is provided under the MIT license. For more details, please refer to [LICENSE](LICENSE).

---

## Acknowledgement

- This work builds upon the theoretical foundation of [UCGM (Unified Continuous Generative Models)](https://github.com/LINs-lab/UCGM), which provides the algorithmic framework for our dynamic generative modeling approach.
- This work builds upon [Unified Video Action (UVA)](https://github.com/real-stanford/unified_video_action), which provides the foundational visuomotor policy architecture.
- Lots of code are inherited from [Diffusion Policy](https://github.com/real-stanford/diffusion_policy) and [MAR](https://github.com/LTH14/mar).
- We use the public datasets from [Diffusion Policy](https://github.com/real-stanford/diffusion_policy) (PushT) and [LIBERO](https://libero-project.github.io/main.html) (Libero10).
