# SemanGraphPPI （The content is being continuously updated ...）
---
A repo for "Protein Interaction Pattern Recognition Using Heterogeneous Semantics Mining and Hierarchical Graph Representation".

## Contents

* [Abstracts](#abstracts)
* [Requirements](#requirements)
   * [Download projects](#download-projects)
   * [Configure the environment manually](#configure-the-environment-manually)
* [Usages](#usages)
   * [Data preparation](#data-preparation)
   * [Training](#training)
   * [Pretrained models](#pretrained-models)
* [Contact](#contact)

## Abstracts
Discovering the inherent routine of interactions between proteins is of great importance in deciphering microscopic life systems. It is essentially a pattern recognition task since the most crucial step is to uncover the complex interaction patterns hidden in the biological knowledge graphs. Recent advances have shown great promise in this regard; however, existing solutions still overlook three critical issues: 1) category heterogeneity, 2) relation heterogeneity, and 3) annotation scarcity, which severely hinder the comprehensive identification and understanding of protein interaction patterns. To address these issues, we introduce a protein interaction pattern recognition framework based on heterogeneous semantics mining and hierarchical graph representation, namely SemanGraphPPI. Our model integrates the annotation knowledge graph with the interaction knowledge graph through hierarchical graph representation, enabling end-to-end function representation of proteins. Additionally, it effectively mines heterogeneous function semantics of proteins by explicitly modeling the heterogeneous information in the annotation knowl
edge graph, and enhances the function representation of *under-labeled* proteins through a well-designed hierarchical knowledge enhancement module. Exhaustive experiments on three bench mark datasets demonstrate that our proposed model achieves state-of-the-art performance compared to other baseline methods and exhibits good generalization and efficiency in large-scale PPI prediction.
![SemanGraphPPI architecture](https://github.com/bixiangpeng/SemanGraphPPI/blob/main/framework.png)

## Requirements

* ### Download projects

   Download the GitHub repo of this project onto your local server: `git clone https://github.com/bixiangpeng/SemanGraphPPI/`


* ### Configure the environment manually

   Create and activate virtual env: `conda create -n SemanGraphPPI python=3.8 ` and `conda activate SemanGraphPPI`
   
   Install specified version of pytorch: ` conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia`
   
   Install specified version of PyG: ` conda install pyg==2.5.2 -c pyg`
   
   :bulb: Note that the operating system we used is `ubuntu 22.04` and the version of Anaconda is `24.1.2`.

  
##  Usages

* ### Data preparation
  There are three benchmark datasets were adopted in this project, including `DIP S. cerevisiae`, `STRING S. cerevisiae`, and `STRING H. sapiens`.

  🌳 To facilitate understanding, the following is an explanation of the `data` file directory:
    ```text
       >  data
          ├── Annotation_KG               - A folder for annotation knowledge graph data.           
          │   ├── BP_subgraph.pkl             - A file for BP(Biological Process) knowledge subgraph.
          │   ├── MF_subgraph.pkl             - A file for MF(Molecular Function) knowledge subgraph.
          │   └── CC_subgraph.pkl             - A file for CC(Cellular Component) knowledge subgraph.
          ├── DIP_S.cerevisiae            - A folder for DIP S. cerevisiae dataset.
          │   ├── train.tsv                   - A TSV file for training dataset. 
          │   ├── test.tsv                    - A TSV file for test dataset. 
          │   ├── Interaction_KG              - A folder for interaction knowledge graph data.
          │       ├── IKG_edge.pkl                - A pkl file recording the edges in interaction knowledge graph.
          │       ├── edge_index_map_dict.pkl     - A pkl file recording the index of interaction edges in interaction knowledge graph.
          │       ├── index_map_dict.pkl          - A pkl file recording the index of protein nodes in interaction knowledge graph.
          │       ├── annotation_index_map.pkl    
          │       └── annotation_batch.pkl        
          ├── STRING_H.sapiens            - A folder for STRING H. sapiens dataset.
          └── STRING_S.cerevisiae         - A folder for STRING S. cerevisiae dataset.

   ```


* ### Training
  After processing the data, you can retrain the model from scratch with the following command:
  ```text
  For Yeast dataset:
    python my_main.py --datasetname yeast --output_dim 1

  For Multi-species dataset:
    python my_main.py --datasetname multi_species --output_dim 1 --identity any

  For Multi-class dataset:
    python my_main.py --datasetname multi_class --output_dim 7  

   ```
  
  Here is the detailed introduction of the optional parameters when running `my_main.py`:
   ```text
    --datasetname: The dataset name, specifying the dataset used for model training.
    --output_dim: The parameter for specifying the number of PPI categories in the dataset.
    --identity: The threshold of identity, specifying the multi-species dataset under this sequence identity.
    --device_id: The device, specifying the GPU device number used for training.
    --batch_size: The batch size, specifying the number of samples in each training batch.
    --epochs: The number of epochs, specifying the number of iterations for training the model on the entire dataset.
    --lr: The learning rate, controlling the rate at which model parameters are updated.
    --num_workers: This parameter is an optional value in the Dataloader, and when its value is greater than 0, it enables multiprocessing for data processing.
    --rst_path: The parameter for specifying the file saving path.  ```

* ### Pretrained models

   If you don't want to re-train the model, we provide pre-trained model parameters as shown below. 
<a name="pretrained-models"></a>
   | Datasets | Pre-trained models          | Description |
   |:-----------:|:-----------------------------:|:--------------|
   | Yeast    | [model](https://github.com/bixiangpeng/SSPPI/blob/main/model_pkl/yeast/model.pkl) | The pretrained model parameters on the Yeast. |
   | Multi-species     | [model_01](https://github.com/bixiangpeng/SSPPI/blob/main/model_pkl/multi_species/model_01.pkl) &nbsp; , &nbsp; [model_10](https://github.com/bixiangpeng/SSPPI/blob/main/model_pkl/multi_species/model_10.pkl) &nbsp; , &nbsp; [model_25](https://github.com/bixiangpeng/SSPPI/blob/main/model_pkl/multi_species/model_25.pkl) &nbsp; , &nbsp; [model_40](https://github.com/bixiangpeng/SSPPI/blob/main/model_pkl/multi_species/model_40.pkl)  &nbsp; , &nbsp; [model_any](https://github.com/bixiangpeng/SSPPI/blob/main/model_pkl/multi_species/model_any.pkl)      | The Pretrained model parameters on the Multi-species under different sequence identities. |
   | Multi-class    | [model](https://github.com/bixiangpeng/SSPPI/blob/main/model_pkl/multi_class/model.pkl)   | The pretrained model parameters on the Multi-class dataset. |
  
   Based on these pre-trained models, you can perform PPI predictions by simply running the following command:
   ```text
    For Yeast dataset:
      python inference.py --datasetname yeast --output_dim 1
  
    For Multi-species dataset:
      python inference.py --datasetname multi_species --output_dim 1 --identity any
  
    For Multi-class dataset:
      python inference.py --datasetname multi_class --output_dim 7  

   ```

## Contact

We welcome you to contact us (email: bixiangpeng@stu.ouc.edu.cn) for any questions and cooperations.

