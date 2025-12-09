# hurricane-cs7643

## Hurricane Model

### Python Environment & Dependencies

In order to execute the training and evaluation code in this section, the correct Python version and dependencies need installed. We recommend installing Anaconda3 version 2023.03 because it was used to execute the experiments for this model - Python version 3.10.

It is recommended to create a Python virtual environment and install dependencies using requirements.txt file in the root directory of this repository.

### Training

This section outlines the steps required to perform training on any of the documented hurricane model variants.

1. Edit the following configuration yaml file with desired training parameters: configs/config_hurricane.yaml

   - Model Variant Keys:
     - Base: 'hurricane'
     - Dropout: 'hurricane_do'
     - Concat-Skip: 'hurricane_c'
     - Add-Skip: 'hurricane_a'
     - LoRA: 'hurricane_lora'

2. Execute the following code in the terminal:

   ```bash
   python train_hurricane.py
   ```

   A CSV containing training metrics and a json file containing hyperparameters from the configuration yaml are saved output/hurricane

3. Obtain loss curves from the 'Plotting' section in the following Jupyter notebook: notebook/hurricane_analysis.ipynb

### Evaluation (Holdout Set)

1. Execute the following code in the terminal:

   ```bash
   python inference_hurricane.py
   ```

   Final challenge score and F1 scores are printed in the terminal.
