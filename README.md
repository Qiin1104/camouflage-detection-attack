
#  Interpretable Adversarial Attacks on Camouflaged Object Detection via Layer Segmented Activation Mapping

> **Important Notice**: This repository contains the official implementation of the manuscript **"Interpretable Adversarial Attacks on Camouflaged Object Detection via Layer Segmented Activation Mapping"**

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17538737.svg)](https://doi.org/10.5281/zenodo.17538737)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


## 🔗 Permanent Resources
- **Source Code**: [GitHub Repository](https://github.com/Qiin1104/camouflage-detection-attack)
- **DOI**: [10.5281/zenodo.XXXXXXX](https://doi.org/10.5281/zenodo.XXXXXXX)
- **Dataset**: *Not publicly available due to military sensitivity*


## Project Overview

This project implements an adversarial attack method for camouflaged object detection based on Guided LayerSAM. Our approach:

- Implements interpretable adversarial attacks using Layer Segmented Activation Mapping
- Supports white-box attacks on ZoomNext and transfer attacks on SINetV2 and IdeNet
- Achieves high stealthiness (PSNR > 30 dB, SSIM > 0.8) while significantly reducing mIoU
- Provides comprehensive evaluation metrics and visualization tools

## Performance Results

| Model | Original mIoU | After Attack | Reduction |
|-------|---------------|--------------|-----------|
| ZoomNext* | 0.638 | 0.550 | 13.8% |
| SINetV2 | 0.491 | 0.360 | 26.7% |
| IdeNet | 0.723 | 0.618 | 14.6% |

*White-box model
![Comparison chart of attack effects](images/P6.pdf)

## Project Structure
camouflage-detection-attack/  
├── 📁 ZoomNext-main/ # White-box model (has its own README)  
├── 📁 IdeNet-main/ # Black-box model 1 (has its own README)  
├── 📁 SINet-V2-main/ # Black-box model 2 (has its own README)  
├── 📁 LayerCAM-jitter/ # Interpretability core 
├── 📁 scripts/ # Utility scripts  
├── 📄 README.md # Main project documentation (this file)  
├── 📄 requirements.txt # Python dependencies  
└── 📄 LICENSE # MIT License

### Important Note:
**Each subproject maintains its own README file** with detailed usage instructions. Please refer to:
- `ZoomNext-main/README.md` - for ZoomNext specific usage
- `IdeNet-main/README.md` - for IdeNet specific usage  
- `SINet-V2-main/README.md` - for SINet-V2 specific usage


## Quick Start


### Prerequisites 
 Python 3.8+
For ZoomNext / IdeNet / SINet-V2: PyTorch >= 1.9, CUDA 11.8 (示例安装)
  ```bash
  pip install torch==1.9.0+cu118 torchvision -f https://download.pytorch.org/whl/torch_stable.html
  pip install -r requirements.txt
  ```
  For LayerCAM-jitter: Jittor 
```bash
python3 -m pip install jittor==1.6.0 -f https://pypi.org/simple`
  ```

### Installation

```bash
# Clone repository
git clone https://github.com/Qiin1104/camouflage-detection-attack.git
cd camouflage-detection-attack

# Install dependencies
pip install -r requirements.txt
```


## Download Pre-trained Models

Please download the pre-trained model weights from the official sources:

| Model | Official Source | Save Location |
|-------|----------------|---------------|
| ZoomNext | [Official Repository](https://github.com/lartpang/ZoomNeXt) |  `ZoomNext-main/checkpoints/` folder |
| IdeNet | [Official Repository](https://github.com/Xiaoqi-Zhao-DLUT/IdeNet) |  `IdeNet-main/checkpoints/` folder |
| SINet-V2 | [Official Repository](https://github.com/GewelsJI/SINet-V2) |  `SINet-V2-main/checkpoints/` folder |

## Basic Usage

#### Generate adversarial attacks using ZoomNext (white-box):

```bash
cd ZoomNext-main
python main_for_image.py --input path/to/images
```

#### Evaluate on black-box models:

```bash
cd IdeNet-main
python MyTesting.py --input path/to/adversarial/images
```

#### LayerCAM analysis:

```bash
cd LayerCAM-jitter
python MyTesting.py --model zoomnext --image path/to/image
```

## Peproducing Paper Results

Our method achieves:

-   **13.8% mIoU reduction** on ZoomNext (white-box)
    
-   **26.7% mIoU reduction** on SINetV2 (black-box)
    
-   **14.6% mIoU reduction** on IdeNet (black-box)
    

To reproduce these results:

1.  Train/download all three base models
    
2.  Generate adversarial samples using ZoomNext + LayerCAM
    
3.  Evaluate transfer attacks on SINetV2 and IdeNet

## Component Details

### ZoomNext-main (White-box Model)

-   Base camouflaged object detection model
    
-   Used as white-box for attack generation
    
-   Key files: `main_for_image.py`, `methods/`, `configs/`
    

### IdeNet-main (Black-box Model 1)

-   Camouflaged object detection model
    
-   Used for transfer attack evaluation
    
-   Key files: `MyTesting.py`, `lib/`, `evaltools/`
    

### SINet-V2-main (Black-box Model 2)

-   Camouflaged object detection model
    
-   Used for transfer attack evaluation
    
-   Key files: `test.py`, `cam/`, `utils/`
    

### LayerCAM-jitter (Interpretability Framework)

-   LayerCAM implementation for feature visualization
    
-   Core of our Guided LayerSAM approach
    
-   Requires Jittor framework
    
-   Key files: `MyTesting.py`, `lib/`, `utils/`

![Comprehensive Evaluation Metrics](images/Pm1.pdf)
## Key Parameters


| Parameter | Recommended Value | Description |
|-----------|------------------|-------------|
| epsilon | 0.1 | Perturbation budget |
| iterations | 10 | Attack iterations |
| top_k | 0.5 | Heatmap focus region ratio |
| target_layer | tra-1-conv | LayerSAM target layer |


## Citation

**This code repository is directly related to the manuscript currently submitted to The Visual Computer.**

If you use this code in your research, please cite our submitted paper:

```bibtex
@article{wang2025interpretable,
  title={Interpretable Adversarial Attacks on Camouflaged Object Detection via Layer Segmented Activation Mapping},
  author={Wang, Qinzi and Xu, Chuanzhen and Tan, Xiaolin and Fu, Sihua},
  journal={The Visual Computer},
  year={2025},
  volume={},
  number={},
  pages={},
  note={Submitted},
  url={https://github.com/Qiin1104/camouflage-detection-attack},
  doi={10.5281/zenodo.17538737}
}
```
## Contact

**Corresponding Author**: Sihua Fu  
**Email**: fsihua@sdxiehe.edu.cn

**Other Authors**:
- Qinzi Wang
- Chuanzhen Xu  
- Xiaolin Tan

## License

This project is licensed under the MIT License - see the LICENSE file for details.

_Note: Each subproject maintains its own README and licensing. Please refer to individual directories for specific details._

