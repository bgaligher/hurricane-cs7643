# hurricane-cs7643

## References

Code from the following public repositories was used/modified for our project:
   1. https://github.com/vdurnov/xview2_1st_place_solution
   2. https://github.com/PaulBorneP/Xview2_Strong_Baseline

## Hurricane Model

### Python Environment & Dependencies

In order to execute the training and evaluation code in this section, the correct Python version and dependencies need to be installed. We recommend installing Anaconda3 version 2023.03 because it was used to execute the experiments for this model - Python version 3.10.

It is recommended to create a Python virtual environment and install dependencies using requirements.txt file in the root directory of this repository.

### Dataset preparation

1. Download the xBD dataset following the instructions at link: https://xview2.org/

2. Execute the following code in the terminal to isolate wind data from xBD:

   ```bash
   python wind.py
   ```

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



## Strong Baseline Model

To run the Strong Baseline model, we suggest creating a new environment and then executing the following instructions.

### Environment & Codebase

First download the following repository: https://github.com/PaulBorneP/Xview2_Strong_Baseline.

Then replace the files in that repository with those found in the xview2_strong_baseline folder from this hurricane-cs7643 repository. This includes:
1. supervised_dataset.py
2. supervised.py
3. main.py
4. requirements.txt
5. supervised_data.yaml
6. config.yaml

Create a Python virtual environment and install dependencies using the newly replaced requirements.txt file.

### Dataset preparation

1. Download the xBD dataset following the instructions at link: https://xview2.org/

2. Execute the following code in the terminal to isolate wind data from xBD:

   ```bash
   python wind.py
   ```

3. Execute the following code in the terminal to create masks for the datasets

```bash
   create_masks.py
   ``` 
### Training and Evaluation

1. Execute the following code in the terminal:

 ```bash
   python train_main.py
   ```
Final challenge score and F1 scores are logged.
