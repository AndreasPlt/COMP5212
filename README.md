# COMP5212 Machine Learning Project

🎉 Welcome to our COMP5212 machine learning project repository! 🎉

## Project Overview

This project focuses on geolocation prediction (country) using Google Street View pictures. We employ the MobileNet architecture due to its high efficiency and low compute cost, making it suitable for real-world applications. The project consists of several key aspects, including the configuration folder, model implementation, training loop, and testing loop.

## Folder Structure

- `config/`: Contains the YAML configuration file.
- `models/`: Contains model implementation files.
- `train.py`: Training loop script.
- `test.py`: Testing loop script.
- `main.py`: Main script for training or testing, with configurable hyperparameters.

## Usage

1. Clone the repository:

   ```
   git clone https://github.com/AndreasPlt/COMP5212.git
   ```
2. Navigate to the project directory:

   ```
   cd COMP5212
   ```
3. Install the required dependencies:

   ```
    pip install -r requirements.txt
   ```
4. Navigate to source directory:

   ```
    cd src
   ```
5. Customize the configuration in the configs/config.yaml file as needed.
   Run the main script with desired options:

   ```
    python main.py --config configs/default.yaml
   ```

Alternatively, choose a different config.yaml.
Adjust other hyperparameters in the config file, if desired.

## Citation

If you use our project or code in your research or work, please cite our repository:

    ``    @misc{comp5212-project,     author = {Louis Markert, Vierling Lukas, Andreas Pletschko, Lim Choon Hao},     title = {COMP5212 Machine Learning Project},     year = {2023},     publisher = {GitHub},     journal = {GitHub repository},     howpublished = {\url{https://github.com/AndreasPlt/COMP5212.git}},     }    ``

## Milestones

Here are some milestones we achieved during the project:

- Milestone 1: Dataset collection and preprocessing
- Milestone 2: MobileNet model implementation with pre-trained weights
- Milestone 3: Training loop and validation setup
- Milestone 4: Testing loop and evaluation metrics
- Milestone 5: Hyperparameter tuning and performance optimization
  Feel free to explore the repository and provide feedback or suggestions.

We hope you find our project useful! 😊
