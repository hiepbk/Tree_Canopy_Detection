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

2. **Install mmcv with CUDA support**:
   ```bash
   pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1.0/index.html
   ```

3. **Install all other dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Install MMDetection in editable mode**:
   ```bash
   pip install -e . --no-build-isolation
   ```

### Setup Workflow

1. **Download the dataset** from the Solafune competition platform
2. **Convert data to COCO format**:
   ```bash
   python tools/dataset_converters/tree_canopy2coco.py \
       --input data/Tree_Canopy_Data/raw_data/trainval_annotations.json \
       --output-dir data/Tree_Canopy_Data \
       --train-ratio 0.8 \
       --seed 42
   ```
3. **Train the model** (single GPU):
   ```bash
   python tools/train.py configs/mask2former/mask2former_swin-b-p4-w12-384-in21k_8xb2-lsj-50e_tree_canopy.py
   ```
   
   **Train the model** (4 GPUs with WandB logging):
   ```bash
   bash tools/dist_train.sh configs/mask2former/mask2former_swin-b-p4-w12-384-in21k_8xb2-lsj-50e_tree_canopy.py 4
   ```

### ⚠️ Small Dataset Considerations (~150 samples)

If you have a small dataset (e.g., ~150 samples), using Mask2Former with Swin-B may lead to overfitting. **Recommended approach:**

1. **Use the smaller Swin-T config** (optimized for small datasets):
   ```bash
   # Single GPU
   python tools/train.py configs/mask2former/mask2former_swin-t-p4-w7-224_8xb2-lsj-50e_tree_canopy.py
   
   # 4 GPUs
   bash tools/dist_train.sh configs/mask2former/mask2former_swin-t-p4-w7-224_8xb2-lsj-50e_tree_canopy.py 4
   ```

2. **Key differences in Swin-T config:**
   - **Smaller backbone:** Swin-T (96 dims) vs Swin-B (128 dims)
   - **Reduced queries:** 50 vs 100
   - **Fewer decoder layers:** 6 vs 9
   - **Frozen backbone stages:** First 2 stages frozen
   - **Lower learning rate:** 0.00005 vs 0.0001
   - **Stronger augmentation:** Color jitter, wider scale range
   - **More dropout:** Added throughout for regularization

3. **Alternative: ResNet-50 backbone** (even smaller):
   - Use `mask2former_r50_8xb2-lsj-50e_coco.py` as base
   - Modify for tree canopy dataset (similar to Swin-T config)

4. **Additional strategies for small datasets:**
   - **Early stopping:** Monitor validation mAP and stop if it plateaus
   - **Strong data augmentation:** Use all available augmentations
   - **Transfer learning:** Freeze more backbone layers
   - **Cross-validation:** Consider K-fold if dataset is very small
   - **Pseudo-labeling:** If you have unlabeled test data, use it for training
   
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

