
**Vehicle Detection Project**

---
### README

The goal of this project is to detect vehicles on the road. I used a deep learning approach with a full-convolutional network to distinguish vehicles from non-vehicles. The reason why I went with this approach instead of using the Histogram of Oriented Gradients (HOG) features that was originally part of this project is these features are learnt by CNN in its inter-mediate layer without hand-coding them. The classifier that has been used here is more scalable than an SVM.

All the code is available through the [Project.ipynb](/Project.ipynb) Python notebook
The final result could be seen from this [output.mp4](/output.mp4) or from YouTube from [here](https://youtu.be/uuvNpC7VU4c)

### Data preparation
Data being key to any classifier/machine learning I would like to mention the different sources of data that I tried:

###### Primary source
[GTI vehicle image database](http://www.gti.ssr.upm.es/data/Vehicle_database.html)

[KITTI vision benchmark suite](http://www.cvlibs.net/datasets/kitti/)

![Additional source](http://www.gti.ssr.upm.es/data/Data/Images/VehicleDatabase/VehicleDatabase.png)

###### Additional source
[Udacity annotated driving dataset](https://github.com/udacity/self-driving-car/tree/master/annotations)

![Additional source](https://raw.githubusercontent.com/udacity/self-driving-car/master/annotations/images/auttico.png)

Additional data source had to be pre-processed to make it available for training. I used the below functions to help me in doing that. It must be noted that not the entire dataset was used. The total dataset was close to 5 GB. The time it took to do conversion was more than 2 hours on my machine. Here I crop the image to the region of interest and place it to a data folder based on the category of the image from csv file. Traffic lights and Pedestrians are considered as non-vehicles in this setup

###### Specific data augmentation
My model was classifying certain section of the roads as vehicles when it is not actually a road. It was just a cone on the side. It also had issues with detecting white cars when seen from a longer distance. Getting specific images on these and duplicating them and augmenting them helped my model learn better.

```python
def save_as_sample(inp, out, xmin, ymin, xmax, ymax, label):
    # Read the file
    img = cv2.imread(inp)

    # Crop to the region of interest
    img = img[ymin:ymax, xmin:xmax]

    # Resize to a desired dimension
    img = cv2.resize(img, (64, 64))

    # Find the category
    if label == 'pedestrian' or label == 'trafficLight':
        category = 'non-vehicles'
    else:
        category = 'vehicles'

    # Write the image to disk
    cv2.imwrite('data/additional/{0}/{1}'.format(category, out), img)

def build_additional_samples(path, input_idx):    
    with open('{}/labels.csv'.format(path)) as log_file:
        # Read the csv file
        reader = csv.reader(log_file)
        
        idx = 0
        if input_idx == 0:
            idx = 1
    
        # Read all the lines from csv file
        for line in tqdm(reader):     
            if len(line) == 1:
                log = line[0].split(' ')
            else:
                log = line
            
            xmin = int(log[0 + idx])
            ymin = int(log[1 + idx])        
            xmax = int(log[2 + idx])
            ymax = int(log[3 + idx])
            inp = '{0}/{1}'.format(path, log[input_idx])
            label = log[5 + idx].lower().replace('"', '')
            out = log[0 + idx] + log[1 + idx] + log[2 + idx] + log[3 + idx] + '_' + log[input_idx]

            save_as_sample(inp, out, xmin, ymin, xmax, ymax, label)
```

> 
Number of vehicles examples before data augmentation = 8792
Number of non-vehicles examples before data augmentation = 8968
Total number of vehicles examples = 11429
Total number of non-vehicles examples = 10768

**Note: I did not end up using the additional data for my final training

`90% of the data was used for training and remaining for test`

### Classifier

For my classifier I used a convolutional neural network inspired from the previous projects. 
The fundamental idea behind the network is a CNN works by running through the image (`similar to a sliding window search`) at every stage. Histograms of gradients are akin to the convolutions and pooling that happens at each stage in our network. By separating the flattening step in the architecture I managed to use the last layer of the CNN to directly detect the regions where there is a high probability of vehicle to exist. This along with traditional machine learning method of using heat maps helps in quickly detecting the vehicles. Initially I tried the sliding window technique but found this CNN layer technique more effective

###### Architecture

![training plot](/examples/architecture.png)

``` python
def build_model(hp, flatten=False):
    # Input
    inp = Input(hp.input_shape)
    
    # Normalization
    out = Lambda(lambda x: x/127.5 - 1.0)(inp)
    
    # Convolution + Dropout
    out = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(out)
    out = Dropout(hp.dropout)(out)
    
    # Convolution + Dropout
    out = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(out)
    out = Dropout(hp.dropout)(out)
    
    # Convolution + MaxPooling + Dropout
    out = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(out)
    out = MaxPooling2D(pool_size=(8, 8))(out)
    out = Dropout(hp.dropout)(out)
    
    # Convolution + Dropout
    out = Convolution2D(64, 8, 8, activation='relu')(out)
    out = Dropout(hp.dropout)(out)
    
    # Convolution
    out = Convolution2D(1, 1, 1, activation='tanh')(out)
    
    # Only during training time
    if flatten:
        # Flatten
        out = Flatten(name="flatten")(out)
    
    # Create the model
    model = Model(inp, out)
    
    return model
    
    
def train():
    # Step 0: Init hyper-parameters
    hp = HyperParameters()

    # Step 1 : Extracting the required data
    samples_train, samples_valid = extract_samples()
    gen_train = batch_generator(samples_train, hp.batch_size)
    gen_valid = batch_generator(samples_valid, hp.batch_size)

    # Step 2: Set other Hyper-parameters    
    hp.input_shape = read_image(samples_train[0]).shape
    num_samples = len(samples_train) #* 2
    
    # Step 3 : Train the model
    model = build_model(hp, True)    
    model.summary()
           
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])    
    history_object = model.fit_generator(gen_train,
                                        nb_epoch=hp.epochs,
                                        samples_per_epoch=num_samples,
                                        nb_val_samples=num_samples,
                                        validation_data=gen_valid,
                                        verbose=1)
    
    # Display some metrics
    show_plot(history_object)
    
    # Preserve the weights
    model.save_weights('model.h5')

```

##### Batch generator

I used a batch generator to scale up to large datasets. Neverthless I ended up not using the large dataset. It was quicker to iterate on smaller dataset. I have discussed about this in future work.

```python

def batch_generator(samples, batch_size):
    num_samples = len(samples)

    while 1:
        shuffle(samples)

        for offset in range(0, num_samples, batch_size):
            end = offset + batch_size
            batch_samples = samples[offset:end]

            images = []
            labels = []

            for batch_sample in batch_samples:                
                # Get the image
                image = read_image(batch_sample)
                
                # Data augmentation through flipping
#                 flipped = np.fliplr(image)
                
                # Check if this sample is a vehicle(1)
                label = int(not('non-vehicles' in batch_sample))
                
                # Add all the data to the batch            
#                 images.extend([image, flipped])
#                 labels.extend([label, label])
                images.extend([image])
                labels.extend([label])
                
            X_train = np.array(images)
            y_train = np.array(labels)
                
            # Shuffling again so to overcome the repeat due to data augmentation
            yield shuffle(X_train, y_train)

```


I used an adam optimizer to do the training. Mean squared error is used to compute the loss of the model

The hyper parameters I used were
`epochs=18, batch_size=64, dropout=0.5`

![training plot](/examples/training.png)

I extensively used dropout to avoid over-fitting with the data.


![tests on samples](/examples/tests.png)



### Detecting regions of activation (Window Search)

I initially had multiple window scales `(60,60), (90,90), (120,120)` to build the bounding boxes for searching for vehicles. Having different scales of windows helps the classifier identify the vehicle better. This also helped in grouping the detected vehicles through heatmap later. Later this was not scaling well when I tried to run through the videos.

So I decided to modify the architecture of my network and use the detections directly from the convolutional layer output. Basically the output of the CNN before flattening is going to perform the sliding search. Areas of the image where there is more activations while using a hyperbolic tan function (between -1 and 1) gives the areas to choose.

I decided to clip off the region that is not necessary for vehicle detection. Having an `activation threshold of 0.9` helped in reducing false activations. Also by keeping a `threshold of 9` while building the heat map helped in keeping the false positive minimum.

##### Detecting active regions from CNN

![detections](/examples/detections.png)

##### Using threshold

![test images](/examples/test_images.png)
---

### Video Implementation

#### The pipeline

```python
# PIPELINE
# PIPELINE
ACTIVATION_THRESHOLD = 0.9

# Initialize the lane detector
# calibration_data = calibrate()
# lane_detector = LaneDetector(calibration_data)
class VehicleDetector():
    def __init__(self, n_frames=3):       
        # Initialize
        self.history = deque(maxlen=n_frames)
    
    def process(self, img):        
        # Find the areas on the image where vehicle detections are activated
        bboxes = detect_vehicles(img)   

        # Visualize the bounding boxes
        #out = draw_boxes(img, bboxes, color=(0, 255, 0), thick=2)
        
        # Add lane lines to the image
        #img = lane_detector.process(img)

        # Build up heatmap
        heat = add_heat(img, bboxes)
        
        # Add it to the queue
        self.history.append(heat)
               
        # Apply threshold to eliminate false positives
        heat = apply_threshold(sum(self.history)//len(self.history), 9)
                
        # Visualize the heatmap 
        heatmap = np.clip(heat, 0, 255)
        #plt.imshow(heatmap, cmap='hot')             

        # Find final boxes from heatmap using label function
        self.labels = label(heatmap)
                                            
        out = draw_labeled_bboxes(np.copy(img), self.labels)

        return out

```
The final result could be seen from this [output.mp4](/output.mp4) or from YouTube from [here](https://youtu.be/uuvNpC7VU4c)


#### Avoiding false positives

I sum the heatmaps of 3 consecutive frames from the video and take the average of their values to threshold them

Activations

```python
    # This finds us rectangles that are interesting
    xx, yy = np.meshgrid(np.arange(activations.shape[2]), np.arange(activations.shape[1]))
    x = (xx[activations[0,:,:,0] > ACTIVATION_THRESHOLD])
    y = (yy[activations[0,:,:,0] > ACTIVATION_THRESHOLD])
```

Threshold (3) on heatmap

```python 
def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap
```



![test images](/examples/heatmap.png)

---

### Future work

The current pipeline uses a minimal amount of dataset and would fail in real use cases. Since the network uses a small number of parameters (limited by my GPU) the convolutions were not activating for vehicles that are far off in the road. Training the model with more general dataset would create a better model. Also to improve the pipeline I could experiment with other available object detection pipelines. The current pipeline does not work that fast and would be impractical for real-time object detection. [YOLO:Real-Time Object Detection](https://pjreddie.com/darknet/yolo/) is something that would be really good to try out 


![YOLO](https://pjreddie.com/media/image/Screen_Shot_2016-11-17_at_11.14.54_AM.png)




