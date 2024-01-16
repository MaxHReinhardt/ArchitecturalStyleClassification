# Architectural Style Classification for Mobile Applications

This project aims at developing an architectural style classification model eligible for the use on mobile devices and 
is conducted as part of the course Applied Deep Learning at TU Wien. The approach combines the MobileNets architecture 
with a channel-spatial attention mechanism. A detailed project description can be found in the document 
Assignment_1.pdf that contains a proposal for the project.

The developed model can be used via a Streamlit app: https://architecturalstyleclassification-ecqzmfyac9ucp5f9pvahxu.streamlit.app/


## Summary of Model Development (Assignment 2)
A model was developed based on the approach proposed in Assignment 1 (see Assignment_1.pdf). The following sections 
summarize changes compared to the project proposal, the performance metric that guided model development, the experiments 
conducted to find the right architecture and hyperparameters, the performance of the best model found and the time 
spent on the single tasks.

### Changes Compared to Project Proposal 
A first change compared the project proposal is the use of a bigger dataset. For the model development, a dataset from 
Kaggle with approximately 10,000 images from 25 classes [1] was used that contains the images from the architectural 
style dataset published by Xu et. al. [3] as well as additional images scraped from Google images. 

Further, the Convolutional Block Attention Module (CBAM) was not only applied in the last layer, also model 
architectures with CBAM integrated after each convolutional layer were tested. More information are provided in the 
experiments section.

### Performance Metric and Target Value
As the project aims at developing a model that achieves a reasonable performance and runs on mobile devices, it has two 
conflicting goals. However, since all tested model configurations are based on MobileNetV1, all of them should be 
eligible for the use on mobile devices. Additionally, the classical use cases for architectural style classification
(tourism, education) do not require exceptionally fast inference. Therefore, accuracy was used as the primary metric 
guiding model development. Accuracy was also reported for the model proposed by Wang et. al. [2] that 
currently can be considered as state of the art. The model achieves an accuracy of 64.39% on the architectural 
style dataset from Xu et. al. [3] combining the Inception-v3 model with a channel-spatial attention mechanism. Due to 
the significantly lower model capacity, it is very unlikely to obtain a similar accuracy with a model based on 
MobileNetV1, despite the use of a larger dataset. However, the goal of this project is to get there as close as possible
while retaining the advantages of MobileNets.

### Experiments
The model architecture of the MobileNetV1 allows to control model complexity via the resolution of input images as well
as a width parameter. Further, the modelling approach proposed in this project requires finding a suitable integration
of CBAM. In the training process, learning rate, batch size and weight decay could be varied to optimize model 
performance.

A stepwise procedure was applied to find an optimal configuration. Thereby, the first experiments aimed at finding a 
suitable configuration for the model architecture while keeping training parameters fixed. Following, different sets of
training hyperparameters were tested on the best found architecture. In the following, the experiment setups and key
results are briefly summarized. Detailed results can be retrieved from /experiments/results.ipynb. 

Finding the right architecture:
* Experiment 1: The first experiment aimed at finding a suitable model size for the MobileNetV1 architecture. Therefore, 
three different resolutions (384x384, 256x256, 192x192) and four different model widths (100%, 75%, 50%, 25%) were 
tested using a grid search. The experiment showed that for all model widths, the highest resolution led to similar or 
better results compared to smaller resolutions. Further, a medium model width (50%-75%) led to best results. A possible 
explanation would be that the regularizing effect of a smaller architecture might be of advantage due to the relatively 
small dataset. Very small architectures with a width around 25% seem to have too little capacity for complex 
classification tasks. 
* Experiment 2: The second experiment aimed at investigating the effect of adding CBAM after the last convolutional 
layer as it proved to be effective in the model proposed by Wang et. al. [2]. Therefore, three additional models were 
trained with a resolution of 384x384, varying model widths (100%, 75%, 50%) and an integrated CBAM module after the 
convolutional layers. All three models performed slightly worse compared to the corresponding MobileNets without the 
CBAM integration. 
* Experiment 3: As experiment 1 implies possible advantages of smaller architectures due to their regularizing effect, 
a possible explanation for the worse performance of the models created in experiment 2 is that the CBAM integration adds
complexity to the models which might result in worse generalization. To test this hypothesis, experiment 2 was repeated 
with little weight decay (1e-4). The results were comparable to experiment 2.
* Experiment 4: As an alternative approach to integrating CBAM into the MobilNetV1 architecture, an integration of the 
module after each convolutional layer, potentially maximizing its effect, was tested. Resolution and widths were chosen 
corresponding to experiment 2 and 3. This CBAM integration could improve model performance for a width of 100% and 75%, 
only for width 50% performance were worse compared to vanilla MobileNetV1. The MobileNetV1 with a CBAM integration
after each convolutional layer, a resolution of 384x384 and a width of 100% appeared to be the most promising 
architecture. 

Finding the right hyperparameters for model training: 
* Experiment 5: Using the best found model architecture, in experiment 5, it was searched for an optimal hyperparameter
configuration using random search. Therefore, learning rates were varied between 1e-4 and 1e-2, batch sizes were varied 
between 32 and 128. In total, 10 random combinations were evaluated. However, no better model could be found with 
experiment 5.
* Experiment 6: The results of experiment 5 indicate that an optimal learning rate might lie between 0.002 and 0.005. 
Regarding batch size, results were inconclusive. Thus, experiment 6 repeated experiment 5 with 8 randomly drawn
combinations, learning rates between 0.002 and 0.005 and batch sizes between 32 and 128. Experiment 6 did also not 
reveal a better hyperparameter configuration. However, experiment 5 and 6 provide additional indication that the 
initially chosen learning rate of 0.003 and batch size of 64 are indeed close to an optimal configuration.

### Model Performance
The best found model has MobileNetV1 architecture with a CBAM integration after each convolutional layer, a 
resolution of 384x384 and a width of 100%. The training was performed with a learning rate of 0.003, a batch size of 64 
and without weight decay. On the validation set, the model achieved an accuracy of 58.6%. 

To determine the model's performance on unseen data, it was further evaluated on the test set (see 
/experiments/results.ipynb). Thereby, it achieved an accuracy of 58.0%. The average inference speed for one batch of 64
images was 5.8 seconds on a laptop CPU.

### Work-Breakdown
In the following, the time needed for the single development steps are reported. The task structure follows the work-
breakdown from Assignment 1. However, the training of MobileNet and MobileNet + CBAM is represented as one task in this
documentation as the same preprocessing and training modules were used for both models. The actual execution of the 
training was performed as part of the experiments. Testing and refactoring are subsumed under the related tasks. 

Efforts are reported in days whereby one day corresponds to approximately 8 hours which were not necessarily worked in 
one piece. All in all, model development took approximately 8 days (9 were estimated in Assignment 1).

Amount of time spend on tasks in days (estimated durations from Assignment 1 in brackets):
* Dataset collection and preparation: 1 (0)
* Implementing MobileNet: 1 (2.5)
* Integrating CBAM: 1 (2)
* Implement Training incl. data preprocessing (for all models): 3 (3.5)
* Fine-tuning and experiments: 2 (1)

Next steps:
* Application development: ? (3)
* Report writing: ? (3)
* Presentation preparation: ? (1)


## Running the scripts
Requirements: Python 3.9 and packages as specified in requirements.txt.

Follow these steps to prepare the data:
1. Clone the repository
2. Download the dataset with the following link (https://www.kaggle.com/datasets/dumitrux/architectural-styles-dataset)
3. Create a folder "data" within the repository and move the unpacked downloaded dataset directory into that folder
4. Run src/data_management.py to split the dataset and create annotation files

After preparing the data, follow these steps to run the tests (locally):
1. For efficient testing, it is recommended to create two files "data/dataset/train_annotation_tiny.csv" and 
"data/dataset/validation_annotation_tiny.csv" by duplicating the corresponding annotation files and deleting most rows, 
such that tests can be performed on small example datasets
2. Run the tests in tests/tests.py (to shorten runtimes, resolutions/widths/batch sizes can be decreased)

After preparing the data, follow these steps to explore different model and training configurations (in colab):
1. Zip the data folder and upload it to Google Drive
2. Open colab_execution/experiment_compare_model_hyperparameters.ipynb or 
colab_execution/experiment_compare_training_hyperparameters.ipynb from GitHub in Colab
3. Optionally, define configurations as wished 
4. Run the cells


## References
[1] Kaggle. (n.d.). Architectural styles. https://www.kaggle.com/datasets/dumitrux/architectural-styles-dataset

[2] Wang, B., Zhang, S., Zhang, J., and Cai, Z. (2023). Architectural style classification based on CNN and channel–
spatial attention. Signal, Image and Video Processing, 17(1), 99-107.

[3] Xu, Z., Tao, D., Zhang, Y., Wu, J., Tsoi, A.C. (2014) Architectural style classification using multinomial latent 
logistic regression. Proceedings of the European conference on computer vision (ECCV).


