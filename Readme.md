This repository contains code for a submitted paper to TSE. Detailed information is as follows:
### Title:
Pretrain, Prompt, and Transfer: Evolving Digital Twins for Time-to-Event Analysis in Cyber-physical Systems
### Abstract: 
Cyber-Physical Systems (CPSs), e.g., elevator systems and autonomous driving systems, are progressively permeating our everyday lives. To ensure their safety, various analyses need to be conducted, such as anomaly detection and time-to-event analysis (the focus of this paper). Recently, it has been widely accepted that digital Twins (DTs) can serve as an efficient method to aid in the development, maintenance, and safe and secure operation of CPSs. However, CPSs frequently evolve, e.g., with new or updated functionalities, which demand their corresponding DTs be co-evolved, i.e., in synchronization with the CPSs. To that end, we propose a novel method, named PPT, utilizing an uncertainty-aware transfer learning for DT evolution. Specifically, we first pretrain PPT with a pretraining dataset to acquire generic knowledge about the CPSs, followed by adapting it to a specific CPS with the help of prompt tuning. Results highlight that PPT is effective in time-to-event analysis in both elevator and ADSs case studies, on average, outperforming a baseline method by 7.31 and 12.58 in terms of Huber loss, respectively. The experiment results also affirm the effectiveness of transfer learning, prompt tuning and uncertainty quantification in terms of reducing Huber loss by at least 21.32, 3.14 and 4.08, respectively, in both case studies.
### Dataset:
This code uses two dataset:
- Orona Elevator Dataset: We can not provide this data due to non-disclosure agreement with Orona.
- DeepScenario Automous Driving Dataset: Please refer to the original dataset [link](https://github.com/Simula-COMPLEX/DeepScenario/tree/main/deepscenario-dataset)
### Implementation
- ```config.py```. This file contains settings of the experiments, e.g., file paths, batch sizes, transfer learning pairs.
- ```data_analysis.py```. This file performs serveral analysis on the dataset, which are used for the design of DTM and DTC.
- ```dataset.py```. This file encapsule the deepscenario and elevator data into pytorch dataset class.
- ```hyperparameter_tuning.sh```. This script performs hyperparmater tuning on the cluster.
- ```model.py```. This file contains model structure of DTM and DTC.
- ```process_data.py```. This file conduct pre-processing on the DeepScenario and elevator data.
- ```produce_promt.py```. This file generates prompt template for the src and target model to fill in.
- ```train.py```. This file contains code to train a DT.
- ```transfer_learning.py```. This file performs transfer learning between source DT and target DT.
- ```uncertainty_quantification.py```. This file contains three uncertainty quantification methods: UT,bayesian and ensemble
- ```utils.py```. This file contains some utility functions.