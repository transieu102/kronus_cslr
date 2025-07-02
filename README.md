# UIT Kronus - CSLR

This repository contains the solution code for the Continuous Sign Language Recognition (CSLR) challenge, hosted at ICCV 2025: [ICCV MSLR Challenge](https://iccv-mslr-2025.github.io/MSLR/).

**Note:** This codebase is from the final challenge submission and requires significant clean-up. Please expect hard-coded paths and settings; we plan to update and refactor soon.

## Achievements
- **Task 1 (Signer Independent):** Top 2
- **Task 2 (Unseen Sentence):** Top 3

## Installation
Install all required dependencies using pip:

```bash
pip install -r requirements.txt
```

## Inference (Last Submission)

1. **Download Checkpoints:**
   - Download the model checkpoint(s) from the provided Google Drive link: [Google Drive - Challenge Resources](https://drive.google.com/drive/folders/1EMjx0MYVghyJSPlEqKaahomhDNjMm3dM?usp=sharing)

2. **Get Last Submission Info:**
   - For each task, find the corresponding last submission info in the appropriate file (e.g., `SI_last_submission_info`, `US_last_submission_info`) from the same Google Drive link above.

3. **Adjust Inference Script:**
   - Edit `inference_batch_fusion.py` to update paths and settings according to your environment and the downloaded checkpoint.

4. **Run Inference:**
   - Execute the following command:
     ```bash
     python inference_batch_fusion.py
     ```

## Reproduce Training

1. **Get Config:**
   - Download the configuration file for the model you want to reproduce from the [Google Drive - Challenge Resources](https://drive.google.com/drive/folders/1EMjx0MYVghyJSPlEqKaahomhDNjMm3dM?usp=sharing).

2. **Run Training:**
   - Use the following command, replacing `<config_path>` with your config file:
     ```bash
     python main.py --config <config_path>
     ```

## Custom Data

1. **Create Config:**
   - Prepare a configuration file for your custom dataset. Refer to examples in the `configs/` folder.

2. **Run Training:**
   - Use your config file with `main.py`:
     ```bash
     python main.py --config <your_custom_config.yaml>
     ```

**Note:**
- The current codebase contains hard-coded paths and settings. Please modify the relevant files as needed for your setup. We will provide a more flexible and user-friendly version soon.