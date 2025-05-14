# TGATGNN: Transformer Global Attention Graph Neural Network

## Abstract

Machine learning (ML) has received significant attention in the field of materials science for its potential to accelerate discovery and property prediction. In this study, we employ a Graph Neural Network (GNN) to predict the properties of inorganic crystal structures using data obtained from the Materials Project. The reliability of the training dataset, calculated from Density Functional Theory (DFT), was first validated, and the performance of the GNN was benchmarked against conventional ML models. The primary objective of this work is the mitigation of the over-smoothing problem inherent to GNNs, which occurs due to repeated data aggregation across multiple layers. Transformer-based mechanisms were used to enhance the model's ability to capture long-range dependencies within the graph. Four different Transformer integration strategies were investigated. The best-performing model demonstrated a minimum reduction of 4.72-24.66% in Mean Absolute Error (MAE) across all properties compared to the original model. The degree of over-smoothing was quantified using cosine similarity, with results indicating a substantial reduction in our proposed model. To exhibit predictive capability the proposed architecture was tested on high-entropy alloys (HEAs), confirming its robustness on complex materials system. 

## Summary Result

| Property | MAE | Uncertainty(±) | Units |
|---|---|---|---|
| Formation Energy | 0.03893 | 0.00035 | eV/atom |
| Absolute Energy | 0.05949 | 0.00304 | eV/atom |
| Fermi Energy | 0.33186 | 0.00162 | eV/atom |
| Band Gap | 0.31100 | 0.00382 | eV |
| Bulk Modulus | 0.04479 | 0.00064 | log(GPa) |
| Shear Modulus | 0.08596 | 0.00080 | log(GPa) |
| Poisson's Ratio | 0.02930 | 0.00021 | - |


## Necessary Installations
1. **Pytorch:** Use the following command to install:
  ```bash
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
  ```
2. **Other packages:** Use the following command to install:
  ```bash
  pip install numpy pandas scikit-learn pymatgen torch_geometric torch_scatter
  ```  
## Data Preparation

1.  **Download Dataset:** Obtain the material property dataset from [this link](https://widgets.figshare.com/articles/12522524/embed?show_title=1).
2.  **Unzip Files:** Extract the contents of the downloaded file.
3.  **Move Data Folder:** Place the extracted `data` folder into the same directory as the GATGNN scripts.

## USAGE

### Model Training

Use the `train.py` script with the desired parameters to train the GATGNN model.

**Parameters:**

`--model_name` Choose which TGATGNN model variant to use

`--property` Property to predict (absolute-energy, band-gap, bulk-modulus, fermi-energy, formation-energy, poisson-ratio, shear-modulus)

`--data_src` Dataset source (CGCNN or MEGNET)

`--num_layers` Number of layers

`--num_neurons` Number of neurons per layer

`--num_heads` Number of attention heads

`--use_hidden_layers` Whether to use hidden layers after pooling

`--global_attention` Pooling method (composition or cluster)

`--cluster_option` Clustering strategy (fixed, random, or learnable)

`--concat_comp` Concatenate composition vector after pooling

`--train_size` Size of the training set (e.g., 0.8)

**Example Usage:**

To train a model to predict the shear modulus using the CGCNN dataset with 6 layers and 128 neurons:

```bash
python train.py --property shear-modulus --data_src CGCNN --num_layers 6 --num_neurons 128
```
Upon successful training, the trained model will be saved in the `TRAINED` folder.

### Model Evaluation
Use the evaluate.py script with the same parameters used during training to evaluate the trained model.

**Example Usage:**

To evaluate the model trained for shear modulus with 6 layers and 128 neurons using the CGCNN dataset:

```bash
python evaluate.py --property shear-modulus --data_src CGCNN --num_layers 6 --num_neurons 128
```
The evaluation results will be saved in the `RESULTS` folder.

### Material Property Prediction
Use the `predict.py` script with the same parameters used during training to predict the properties of new materials.

Before running the prediction script, ensure that the Crystallographic Information File (.cif) of the material(s) you want to predict is placed in the prediction-directory folder within the `DATA` directory.

**Parameters:**

`--property` The material property the trained model predicts.
[additional_parameters]: The same parameters used during training.
`--to_predict` The name of the .cif file (without the .cif extension) located in the `data/prediction-directory` folder.
**Example Usage:**

To use the trained shear modulus model (6 layers, 128 neurons) to predict the property of the material structure defined in mp-1.cif:


```bash
python predict.py --property shear-modulus --num_layers 6 --num_neurons 128 --to_predict mp-1
```
**Important Notes:**

Ensure that the parameters used for training, evaluation, and prediction are identical.
The model must be successfully trained before you can perform evaluation or prediction.

## Acknowledgement
We use GATGNN from "Louis, Steph-Yves, Yong Zhao, Alireza Nasiri, Xiran Wang, Yuqi Song, Fei Liu, and Jianjun Hu*. "Graph convolutional neural networks with global attention for improved materials property prediction." Physical Chemistry Chemical Physics 22, no. 32 (2020): 18141-18148." as original model.
