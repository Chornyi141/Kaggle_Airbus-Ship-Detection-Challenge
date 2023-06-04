# Airbus Ship Detection Challenge
UNet model for solving ship detection task.

## File structure
To load from Kaggle:
- ./train_v2/
- ./test_v2/
- ./train_ship_segmentations_v2.csv

Paths to folders with images can be changed in the 'config' block within each Notebook.

Trained model will be saved into:
- ./model.h5

## Data analysis
EDA is stored in 'Data analysis.ipynb'.
### Integrity check
All images was loaded to check if there are corruptions or incorrect sizes.
Image '6384c3e78.jpg' in the train set was corrupted, so it was removed from all further actions.
<i>It's recomended to sckip integrity check due to its high time consumtion.</i>
### Analysis results
- dataset is imbalanced: only 22% of images contains ships and more than 90% of ships takes less than 1% of an image space (sparce segmentation space).
- images contain such objects: sea, ships, coastes/lands, industrial objects, clouds, underwater sands, cities, harbors.
- segmentations represented as rectangles and they doesn't intersect for different ships.

## Model training
Model training is stored in 'Model training.ipynb'. 

### Data limitation
Due to limited time and computational resources model was trained only on first 10000 samples.

### RLE
Segmentations in training dataset stored in RLE (Run-length encoding). So decoder and encoder were written.
 
In this dataset RLE encoding is performed on flattened image. Every two continuous values represents index of pixel in 1D array and number of segmented (with ship) pixels.

### Data loaders
Due to high amount of pictures (190k+ in train dataset) and difficult data preparation custom loaders were written. At first python iterators were created with all needed logic and then they passed into Tensorflow Dataset class. Also shuffling is included in these loaders so Dataset class doesn't need to preload extra data.

Data loaders workflow:
- shuffle image paths
- load image
- create segmentation mask
- augment and normalize image
- augment both image and mask
- reshuffle dataset if it ends

Some image augmentations (brightness, contrast and noise) applied only to original image as they are only needed to confuse model's input. Geometrical augmentations applied to both image and mask as changing original image entails changes on mask.

### UNet 
UNet is a good model architecture for detecting small objects on images so it can be suitable for this task. Basically it consists of three parts: downsampling part (backbone), bottom part (neck) and upsampling part (head). 

Every downsampling block contains 2 convolutional layers and max pooling. Bottom of UNet doesn't need max pooling and only contains 2 conv layers. Upsampling blocks is a reversal to downsampling and also they contains skip connections: 
- instead of max pooling we apply transpose-convolutional layer to upscale our image;
- skip connection transfer features from symmetrical part of backbone so the next convolutional layer can spread new features based on old high resolutional features;
- 2 more conv layers.

Also we apply BatchNormalization after every convolutional layer to improve convergence speed. And a Dropout layer on the bottom of the network to apply more regularization and improve generalization of the model. 

Alternatively we can load previously trained model to continue its training. To do that run 'load model' block instead of 'create model'

Before training the model we need to define its optimizer. We choose Adam as it performes well in all cases. As loss metric we choose dice score, which is combination of recall and precision metrics. Keras doesn't have this metric by default so we can write our own implementation and pass it into compiler. Dice score is higher when model performs better so to use it as loss function we need (1 - dice_score).
To optimize learning rate we can use learning rate decay to decrease it every N iterations.

### Model training
To get some automation during training we will use callback system:
- ModelCheckpoint is used for saving best model based on validation loss;
- EarlyStopping is used to stop learning when validation loss doesn't improve for some number of epochs.

### Training results
Result of training is 'model.h5' file with trained model with best validation score.
This model loss is 0.64 dice loss for training data and 0.68 for validation data. This is quite large loss, it is because model haven't trained enough time, trained on a small part of the dataset and doesn't operate with enough number of features in convolutinal layers due to memory limitations.

In a Notebook 'Model inference.ipynb' created some pipeline to check model results and create a submission to Kaggle in RLE format from test data (ran only on first 100 images to check if it is correct). From there you can see how model works on unseen data from the test set. 
- It often misrecognizes lands/coasts with ships.
- Segmentation of ships is very messy, especially for images with noises.
- Sometimes it can misrecognize strong waves with ships.
- It relatively good recognizes large clearly visible ships.

## Further improvements
- Add more features on convolutional layers.
- Transfer learning: either use a suitable model to extract features from images (model trained on a similar task or at least any other image classifier) or train a classifier on this dataset and use its backbone in a segmentation model.
- Rebalance dataset in two possible ways: remove a part of pictures without ships or apply class weights.
- Train the model until validation loss stops improving (I had not enough time to wait for it).
- Train the model on the full dataset.
- To distinguish pixels on edges between closely placed ships we can apply weights to that pixels.
