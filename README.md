# Deep Neural Network Analysis SuSe2024
## Blackbox Interpretability - LIME
This repository is from the 2024 course "Deep Neural Network Analysis" from the university of Osnabrück, held by Lukas Niehaus.
Topic for this group project is methods to interpret blackbox models using LIME.
A presentation PDF and scripts with visualizations are provided here, the blackbox models used in this repository are taken from the [repository](https://github.com/lucasld/neural_network_analysis/tree/main/) for group 1 of the course.

# Installation
1. Clone the repository:

   ```bash
   git clone https://github.com/madammann/DNNA24_blackbox_lime.git
   ```

2. Navigate to the project directory:
   ```bash
   cd DNNA24_blackbox_lime
   ```

3. Install dependencies:
    * Using environment.yml
        ```bash
        conda env create -f envirnoment.yml
        ```
    * Using requirements.txt
        ```bash
        pip install -r requirements.txt
        ```
        
4. Download the model checkpoints folder from [github.com/lucasld/neural_network_analysis](https://github.com/lucasld/neural_network_analysis/tree/main/).

5. Put the folder in the root directory of this repository.

# About
## Repository structure
This repository contains two jupyter notebooks, one called segmentation and one called lime.
The Lime notebook contains the pipelines for image and tabular explanations used, the segmentation one visualizes how slic segmentation works.

# Contact
Marlon Dammann <mdammann@uni-osnabrueck.de>  
Iheb Marouani<imarouani@uni-osnabrueck.de>

# References
https://github.com/marcotcr/lime

