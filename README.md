# Tree Canopy Detection Competition

## Overview

Tree canopy segmentation plays a vital role in preventing utility-related hazards and ensuring efficient vegetation management. Inaccurate segmentation can hinder clearance calculations for linear assets, disrupt routine RoW maintenance, and reduce the effectiveness of tree health or species classification systems. Overlapping canopies and varying sensor resolutions create mixed signals that confuse models and lower performance. Current approaches often fail to generalize across regions and imagery types, limiting their real-world utility. This leads to a pressing need for models that not only perform well in controlled conditions but are resilient in the diverse and noisy data typical of field deployments. A general-purpose solution is needed to bring consistency, safety, and automation to this domain.

Solafune Competition platform invites users all over the world to participate in a unique Tree Canopy Detection challenge aimed at solving this global issue. By engaging with a range of high-resolution and sensor-agnostic imagery datasets, participants will be tasked with building and validating segmentation models that excel in complex and variable environments. This competition encourages innovation in machine learning and geospatial AI, with real-world impact in utility safety and environmental monitoring. Evaluation criteria will emphasize robustness across locations, clarity of canopy separation, and computational efficiency. Top solutions will be recognized and may be deployed in critical vegetation management workflows worldwide. Join the Solafune platform to push the boundaries of what's possible in tree canopy detection.

## Prizes

**Total Prize: $12,000**

- **1st place:** $4,000
- **2nd place:** $2,500
- **3rd place:** $2,000
- **4th place:** $1,000
- **5th place:** $500

### Additional Awards

- **The Discussion Award:** $1,000 USD
  - Given to the participant whose topic, created in the discussion feature during the competition period, receives the most votes from prize winners after final judging, excluding their own topics (the recipient can be one of the top winners).

- **The Solafune Award:** $1,000 USD
  - Given to the person who has created content that the management judges has made the most contribution to this competition and the platform, among the topics created by participants during the competition period in the discussion feature (there may or may not be a recipient, including top winners).

- **The Solafune Tools Award:** Solafune original items will be gifted
  - Given to the person who has created improvements or tools that impacted the competition, either on the dataset, pre-processing, post-processing, or model, which in the end improves users' scores in the leaderboard. The improvements or tools must be submitted as a pull request to the **Solafune-Tools** and then must publish a discussion topic about their commits.

**Note:** The potential winners will cooperate in the submission of the source code and the URL of the solution write-up published in the competition discussion after the competition ends (about the top 10 teams). The prize money will be paid after the completion of the reproducibility inspection of the source code and identity verification. Please refer to the terms of participation for the conditions for receiving the prize money.

## Data Overview

### Dataset Source Information

The aerial and Satellite imagery are modified and used from:

- **SWISSIMAGE (Switzerland)** | Dataset | License: OGD
- **NAIP (USA)** | Dataset | License: US-PD
- **Planet SkySat Public Ortho Imagery, RGB (Various Locations)** | Dataset | License: CC-BY 4.0
- **OAM-TCD (Various Locations)** | Dataset | License: CC-BY 4.0
- **New Zealand Imagery (New Zealand)** | Dataset | License: CC-BY 4.0
- **OilPalm_Dataset.id (Indonesia)** | Dataset | License: CC-BY 4.0
- **OliveTreeCrownDB (Morocco)** | Dataset | License: CC-BY 4.0

### Image Format

- **RGB TIFF file format**
- A set of RGB format images in three bands consists of 1 (Red), 2 (Green), and 3 (Blue).
- **Location:** Various locations around the world.

### Annotation Information

**Segmentation Task**

JSON format. Training data contains polygon information in the following format:

```json
{
    "images": [
        {
            "file_name": "40cm_train_7.tif",
            "width": 1024,
            "height": 1024,
            "scene_type": "agriculture_plantation",
            "cm_resolution": 40,
            "annotations": [
                {
                    "class": "individual_tree",
                    "confidence_score": 1.0,
                    "segmentation": [
                        polygon1_x1,
                        polygon1_y1,
                        polygon1_x2,
                        polygon1_y2,
                        ...
                    ]
                },
                {
                    "class": "group_of_trees",
                    "confidence_score": 1.0,
                    "segmentation": [
                        polygon2_x1,
                        polygon2_y1,
                        polygon2_x2,
                        polygon2_y2,
                        ...
                    ]
                }
            ]
        }
    ]
}
```

### Submission Format

Submissions should consist of one file in JSON format. This file must adhere to the prescribed format and reference the training data and sample data JSON.

**File naming:** `your_submission_name.json`

## Solafune-Tools

Solafune-Tools is an open-source repository that provides internal geodata creation and management tools. This package includes functionalities to download STAC catalogs and Sentinel-2 imagery from the Planetary Computer and assemble them into a cloudless mosaic. Additional tools will be added in the future to enhance its capabilities.

This repository enables participants in the Solafune competitions to access shared resources, contribute their own tools, and collaborate on the ongoing development of solafune-tools. We have also integrated evaluation functions and tools for verifying users' submissions, streamlining the competition process, and ensuring consistency.

- **GitHub Repository:** [Solafune-Tools](https://github.com/solafune-tools)

### How to Contribute

We welcome and encourage contributions from the community. To share your tools or functions with other users:

1. **Fork the Repository:** Begin by forking the Solafune-Tools repository to your GitHub account.
2. **Add Your Tools:** Place your tools or functions in the `solafune_tools/community_tools` directory within your forked repository.
3. **Submit a Pull Request:** After adding your contributions, submit a pull request to the original Solafune-Tools repository.
4. **Integration:** Upon review and approval, your tools will be integrated into Solafune-Tools, making them available for all users.

For detailed instructions on how to integrate your tools with Solafune-Tools, please refer to our README.

### Validating Submission

To enhance accessibility and streamline the process, we have now integrated the `competition_tools` module into the solafune-tools package. This allows users to easily access submission validators directly.

Check for `solafune_tools.competition_tools` for the usages.

## Evaluation Method

The evaluation metric is the **Weighted Mean Average Precision (mAP)** used on **Intersection over Union (IoU)** in Instance Segmentation Task.

Solafune offers a new type of competition where we are using weight to assign relative importance to different evaluation units. The weighted score means we want to create a model that is capable of certain scenery types and a certain centimeters resolution.

### Scene Type Weights

We seek the importance of the agriculture/plantation and urban area type because these two scenes will be the ones mainly for the end product.

```python
{
    "agriculture_plantation": 2.00,
    "urban_area": 1.50,
    "rural_area": 1.00,
    "industrial_area": 1.25,
    "open_field": 1.00
}
```

### Resolution Weights

The lowest resolution of the image, the harder to identify canopy of the tree.

```python
{
    "10": 1.00,
    "20": 1.25,
    "40": 2.00,
    "60": 2.50,
    "80": 3.00
}
```

### Evaluation Metric Definition

The threshold of the IoU is at **0.75**. The definition of the evaluation metric is as follows:

\[
\text{mAP}_i = \frac{1}{C_i} \sum_{c=1}^{C_i} \text{AP}_{i,c}
\]

\[
w_i = w_{\text{scene}(i)} \cdot w_{\text{resolution}(i)}
\]

\[
\text{weighted mAP} = \frac{\sum_{i=1}^{N} w_i \cdot \text{mAP}_i}{\sum_{i=1}^{N} w_i}
\]

Where:
- \( \text{mAP}_i \) = Mean Average Precision for image \( i \)
- \( w_i \) = Weight for image \( i \)
- \( w_{\text{scene}(i)} \) = Scene type weight for image \( i \)
- \( w_{\text{resolution}(i)} \) = Resolution weight for image \( i \)
- \( C_i \) = Number of classes in image \( i \)
- \( \text{AP}_{i,c} \) = Average Precision for class \( c \) in image \( i \)

We also have a Python implementation available on GitHub. Please refer to it.

**Reference:** [Intersection over union-based mean average precision](https://github.com/solafune-tools)

## Model Recommendations & Technical Approach

Given the multi-resolution, multi-region, sensor-agnostic tree canopy segmentation nature of this competition, you should prioritize **robust generalization + instance separation + efficiency** rather than only peak performance on a single resolution.

Below is a practical, competition-proven model recommendation stack, from strong baseline → top-tier solution, tailored to the dataset and evaluation goals.

### 🔥 Recommended Model Stack (Quick Reference)

- **Best overall choice (very strong):** Mask2Former + Swin Transformer backbone
- **Best efficiency–performance tradeoff:** U-Net++ or HRNet + multi-scale training
- **Best canopy separation (overlapping trees):** Instance segmentation (Mask R-CNN / CondInst) + post-merge
- **Winning-style ensemble:** Mask2Former + CNN-based segmentation model

### 1️⃣ Mask2Former (Top Recommendation)

#### Why it fits this competition perfectly

Mask2Former is designed for mixed-scale, mixed-domain segmentation and handles:
- Variable resolution imagery (40cm ↔ meter-level)
- Overlapping canopies
- Global context (important for plantations vs forests)
- Polygon-based output conversion

#### Suggested configuration

- **Backbone:** Swin-B or Swin-L
- **Input size:** 1024 × 1024 (native resolution)
- **Task:** Semantic segmentation (merge `individual_tree` + `group_of_trees` → `tree_canopy`)
- **Loss:**
  - Dice Loss
  - Focal Loss
  - BCE (optional)

#### Pros & Cons

✅ **Pros:**
- Excellent generalization
- Handles scale variation
- Strong leaderboard performance in remote sensing tasks

❌ **Cons:**
- Heavy (needs good GPU, ~24GB for Swin-L)

### 2️⃣ U-Net++ / HRNet (Strong & Efficient Baseline)

#### Why use it

If you want fast training, stability, and low risk, this is ideal.

#### Recommended setup

- **Backbone:** EfficientNet-B4 or ResNet-50
- **Multi-scale training:** random resize (512–1024)
- **Loss:**
  - Dice + BCE
  - Boundary loss (optional)

#### Pros & Cons

✅ **Pros:**
- Very stable
- Lower VRAM usage
- Easy polygon extraction

❌ **Cons:**
- Slightly weaker in separating overlapping canopies

👉 Many top teams still include this in ensembles

### 3️⃣ Instance Segmentation (Optional but Powerful)

To explicitly separate overlapping canopies:

#### Best options

- Mask R-CNN
- CondInst
- SOLOv2

#### Strategy

1. Predict individual tree masks
2. Merge close instances → canopy polygons
3. Convert to Solafune JSON format

#### When to use

✔ Dense urban trees  
✔ Orchard / plantation imagery

#### Tradeoff

- ✅ Better separation
- ⚠️ More complex post-processing

### 4️⃣ Multi-Resolution Training Strategy (Critical)

Regardless of model choice, implement these training tricks that matter more than architecture:

#### Random resolution sampling

Train with multiple resolutions: **512, 768, 1024**

#### Scene-type conditioning

Use `scene_type` as auxiliary embedding to help the model adapt to different environments.

#### Strong augmentations

- Color jitter
- Random blur (simulates sensor differences)
- JPEG compression artifacts

### 5️⃣ Loss Function Recommendation (Important)

Best performing combo for tree canopy:

```
Total Loss = Dice + Focal + Boundary Loss
```

#### Why?

- **Dice** → area coverage
- **Focal** → class imbalance
- **Boundary** → canopy edges & separation

### 6️⃣ Final Recommendation (What to Actually Submit)

#### Single-model submission

- **Mask2Former (Swin-B)** + multi-scale training

#### High-score ensemble

- Mask2Former
- U-Net++ (EfficientNet)
- Boundary-aware post-processing

### 7️⃣ Bonus: Solafune-Tools Contribution Idea (Award Potential 🏆)

You can win **Solafune Tools Award** by contributing:

- **Polygon simplification tool** (RDP / Visvalingam)
- **Multi-resolution tiling + merging utility**
- **Boundary-aware mask → polygon converter**

These directly improve leaderboard scores and help the entire community.

## Leaderboard

During the competition, the **Public score**, calculated with about 35% of the evaluation data, will be displayed on the leaderboard. After the competition ends, the **Private score**, calculated with all the evaluation data, will be displayed.

**The ranking on the private leaderboard will be the final result.**

## Important Notes

- The evaluation method and submission format may change during the event.
- After the competition ends, you are obligated to immediately delete all data provided by Solafune. If you do not agree, you are not allowed to participate. If it is found that you have not deleted the data after the competition, we will claim the full amount of actual damages as a penalty.
- The competition may be terminated midway or extended at the discretion of the management. Please note that there will be no prize money if the competition is terminated midway.
- There are no restrictions on the environment or language of implementation.
- For the top winners, you will be asked to submit your source code after the competition ends (about the top 10 teams). If you do not submit your source code within the deadline, or if it is confirmed that you clearly cannot generate answer data, your submission will be invalidated. Depending on the situation, we may permanently freeze your account.

## Getting Started

### Installation

1. **Install PyTorch** (based on your CUDA version):
   ```bash
   pip install torch==2.1.0+cu121 torchvision==0.16.0+cu121 torchaudio==2.1.0+cu121
   ```

2. **Install all other dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Install the package in editable mode**:
   ```bash
   pip install -e .
   ```

4. **Install Weights & Biases** (for logging and visualization):
   ```bash
   pip install wandb
   wandb login  # Follow prompts to enter your API key
   ```

### Setup Workflow

1. **Download the dataset** from the Solafune competition platform
2. **Organize your data** in the following structure:
   ```
   data/Tree_Canopy_Data/
   ├── images/
   │   ├── train/
   │   └── val/
   └── annotations/
       ├── train.json
       └── val.json
   ```

3. **Train the model** (single GPU):
   ```bash
   python tools/train.py tcd/configs/tcd_config.py --work-dir work_dirs/tcd_exp1
   ```
   
   **Train the model** (multi-GPU, e.g., 4 GPUs):
   ```bash
   torchrun --nproc_per_node=4 tools/train.py tcd/configs/tcd_config.py --work-dir work_dirs/tcd_exp1 --gpus 4
   ```
   
   **Optional arguments:**
   - `--resume-from <checkpoint_path>`: Resume training from a checkpoint
   - `--load-from <pretrained_path>`: Load pretrained weights
   - `--seed <number>`: Set random seed (default: 42)
   - `--deterministic`: Use deterministic algorithms

### ⚠️ Small Dataset Strategy: Handling 108 Samples with ~298 Instances per Image

**Current Dataset Statistics:**
- **Training images:** 108 samples
- **Total annotations:** 32,152 instances
- **Average instances per image:** ~298 instances
- **High diversity:** Multiple scene types, resolutions (10cm-80cm), and geographic locations
- **Multi-scale:** Instances vary from individual trees to large groups

**Key Challenges:**
1. **Small Dataset (108 samples)** → High risk of overfitting
2. **Many Instances per Image (~298)** → Model needs to handle dense predictions
3. **High Diversity** → Model must generalize across scene types, resolutions, and locations
4. **Multi-Scale Instances** → Individual trees vs. large groups in same image

#### 1. Model Architecture Adjustments

**Use Smaller Backbone:**
- **Swin-T** (96 dims) instead of Swin-B (128 dims) - recommended
- **ResNet-50** as alternative (even smaller)

**Model Capacity Adjustments:**
- **Increase queries:** 100 (to handle ~298 instances per image)
- **Fewer decoder layers:** 6 instead of 9 (prevent overfitting)
- **Smaller pixel decoder:** 4 layers instead of 6
- **Freeze more layers:** First 3 stages of backbone (instead of 2)
- **Lower learning rate:** 0.00001 initially, with backbone_lr_mult=0.01

**Training Command:**
```bash
# Single GPU
python tools/train.py tcd/configs/tcd_config.py

# 4 GPUs (using torchrun)
torchrun --nproc_per_node=4 tools/train.py tcd/configs/tcd_config.py

# 4 GPUs (alternative - using torch.distributed.launch)
python -m torch.distributed.launch --nproc_per_node=4 tools/train.py tcd/configs/tcd_config.py
```

#### 2. Aggressive Data Augmentation (Critical)

**Goal:** Increase effective dataset size from 108 to 1000+ variations

**Spatial Augmentations:**
- **Multi-scale training:** Random resize to [512, 768, 1024, 1280] with ratio_range=(0.5, 2.0)
- **Random cropping:** Large crop size (1024×1024)
- **Random flipping:** Both horizontal and vertical (prob=0.5)
- **Random rotation:** Small angles (-15° to +15°) to preserve tree shapes
- **Elastic deformation:** Simulates different viewing angles (prob=0.3)

**Photometric Augmentations (Simulate Sensor Differences):**
- **Color jitter:** brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1
- **Random blur:** Gaussian blur (prob=0.3, sigma_range=(0.5, 2.0))
- **Random noise:** Gaussian noise (prob=0.2, std_range=(0.01, 0.05))
- **Gamma correction:** Simulates different exposure (prob=0.3, gamma_range=(0.7, 1.3))
- **JPEG compression:** Simulates compression artifacts (prob=0.2, quality_range=(60, 100))

**Advanced Augmentations:**
- **MixUp:** Combines two images (prob=0.3, alpha=0.2)
- **Mosaic:** Combines 4 images (prob=0.5)
- **Copy-Paste:** Pastes instances from other images (prob=0.3, max_num_pasted=10)

#### 3. Training Strategy

**Learning Rate Schedule:**
- **Initial LR:** 0.00001 (very conservative)
- **Cosine annealing with warmup:** warmup_iters=500, warmup_ratio=0.001, min_lr=0.000001
- **Progressive unfreezing:**
  - Stage 1 (epochs 0-10): Freeze all backbone layers
  - Stage 2 (epochs 10-20): Unfreeze last stage
  - Stage 3 (epochs 20-30): Unfreeze last 2 stages
  - Stage 4 (epochs 30+): Fine-tune all layers

**Regularization:**
- **Weight decay:** 0.05 (strong regularization)
- **Dropout:** drop_rate=0.2, attn_drop_rate=0.2, ffn_drop=0.2
- **Label smoothing:** 0.1 (for classification head)
- **Early stopping:** Monitor validation mAP, patience=10 epochs

#### 4. Loss Function Adjustments (For Many Instances)

**Combined Loss Functions:**
- **Focal Loss:** Handles class imbalance (gamma=2.0, alpha=0.25, weight=2.0)
- **Dice Loss:** Handles instance overlap (weight=1.0)
- **Boundary Loss:** Improves edge separation (weight=0.5)

**Instance Matching Strategy:**
- **Hungarian matcher:** cost_class=2.0, cost_mask=5.0, cost_dice=5.0
- **More sampling points:** num_points=12544 (for dense instances)

#### 5. Multi-Scale Training & Testing

**Training:**
- Train with multiple resolutions: [512, 768, 1024, 1280]
- Wide aspect ratio range: (0.5, 2.0)

**Test-Time Augmentation (TTA):**
- Multi-scale testing: scales=[0.8, 1.0, 1.2]
- Slide window inference: crop_size=(1024, 1024), stride=(512, 512)

#### 6. Handling High Diversity

**Scene-Type Conditioning (Optional):**
- Add scene_type as auxiliary embedding to help model adapt to different environments
- Inject into decoder: `decoder_input = backbone_features + scene_embedding`

**Resolution-Aware Training:**
- Higher weight for high-resolution samples (matches evaluation weights):
  - 10cm: 1.0, 20cm: 1.25, 40cm: 2.0, 60cm: 2.5, 80cm: 3.0

**Stratified Sampling:**
- Already implemented: Maintains distribution of scene_type and cm_resolution in train/val splits
- Ensures model sees all types during training

#### 7. Transfer Learning Strategy

**Use SatlasPretrain Backbone:**
- **Aerial_SwinB_SI** checkpoint (already downloaded in `pretrained/`)
- Pretrained on aerial/satellite imagery → better initialization
- Freeze early stages, fine-tune later stages progressively

#### 8. Validation & Post-Processing

**Validation Strategy:**
- Keep 20% as validation (already done)
- Monitor validation metrics closely
- Consider K-fold cross-validation (5-fold) for very small datasets

**Post-Processing for Dense Instances:**
- **Non-Maximum Suppression (NMS):** Remove overlapping predictions (iou_threshold=0.5)
- **Instance merging:** Merge close instances if needed (distance_threshold=50 pixels)

#### 9. Expected Results

With these strategies:
- **Effective dataset size:** 108 → 1000+ (via augmentation)
- **Overfitting risk:** Reduced via regularization
- **Generalization:** Improved via multi-scale training
- **Instance handling:** Better via adjusted loss functions
- **Diversity handling:** Improved via scene-type awareness

**Target Metrics:**
- Validation mAP: >0.5 (with IoU threshold 0.75)
- Training/validation gap: <0.1 (indicates good generalization)

**Note:** For WandB logging, make sure to install wandb and set your API key:
```bash
pip install wandb==0.16.0
export WANDB_API_KEY="your-api-key-here"  # Optional, will prompt if not set
```
4. **Explore the data** structure and annotations
5. **Develop your segmentation model**
6. **Validate your submission** using `solafune_tools.competition_tools`
7. **Submit your results** in the required JSON format

## Resources

- [Solafune Competition Platform](https://solafune.com)
- [Solafune-Tools GitHub Repository](https://github.com/solafune-tools)
- [Evaluation Metric Implementation](https://github.com/solafune-tools)

## License

Please refer to the individual dataset licenses mentioned in the Data Overview section.

