# MPE Environment Setup

We create a separate conda environment for MPE experiments with the following key dependencies:

## Installation

1. **Transfer Environment File:**
    - Move the environment configuration file `mpe.yml` to your target server.

2. **Create Conda Environment:**
    - Run the following command to create a new Conda environment:
      ```
      conda env create -n mpe -f mpe.yml
      ```

    - **Note:** If you prefer to install the environment manually (instead of using the `.yml` file), make sure to include the following key packages.  We recommend using the exact versions listed below, as they have been tested to be mutually compatible:
      - `python = 3.8.8`
      - `gym = 0.26.2`
      - `pettingzoo = 1.17.0`
      - `pygame = 2.1.2`
      - `matplotlib = 3.5.3`
      - `pandas = 1.5.1`
      - `pytorch (CPU version, e.g., installed via pip3 install torch torchvision; note that the current code is not compatible with GPU. Please modify accordingly if GPU support is needed.)`

## Replacement of Files
After installing the required environment, you need to replace two specific files for proper functionality. Follow these steps:

1. **Replace `clip_out_of_bounds.py`:**
    - Navigate to the following path in your Anaconda environment: 
      ```
      Anaconda\envs\mpe\Lib\site-packages\pettingzoo\utils\wrappers
      ```

2. **Replace `simple_tag.py`:**
    - Go to the following directory in your Anaconda environment: 
      ```
      Anaconda\envs\mpe\Lib\site-packages\pettingzoo\mpe\scenarios
      ```

Make sure to perform these replacements to ensure the correct functioning of the environment.

**Note:** The `clip_out_of_bounds.py` and `simple_tag.py` used for replacement can be found in "materials".

