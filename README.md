## Writeup 

---

## **Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier.
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car_not_car.png
[image2]: ./output_images/car_not_car_histogram.PNG
[image3]: ./output_images/car_not_car_spatial.PNG
[image4]: ./output_images/car_not_car_HOG.PNG
[image5]: ./output_images/search_car.PNG
[image6]: ./output_images/bboxes_and_heat_final.PNG
[image7]: ./output_images/parameter.PNG
[video1]: ./project_video_output.mp4
[videodescussion]: ./project_video_discussion.mp4

### Rubric Points

##### ObjectDetection_Final.ipynb :
    This jupyter Notebook contains code for Data Loading ,Data Visualisation, Classifier , Training Model , Predicting Cars from Video.
    
##### Porject_Output_Video.mp4 :
    Final output of the video with cars detected.

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

Initially i loaded all the `Vehicles` and `Non-Vehicle` from the `Data sets` provided by Udacity in Project Resources using `glob` class. The code related to it is in `cell 2 `.

Below are examples of Vehicle and Non-Vehicle images

![alt text][image1]

Below insights were found after reading the data
`Total No. of Vehicle Images- 8792 , Shape - (64,64,3)`
`Total No. of Non-Vehicle Images- 8968 , Shape - (64,64,3)`

After Visualizing the Data i tried to extract features from the DataSets using various functions such as `bin_spatial` , `color_hist` , `get_hog_features`  for different features.

Code can be seen under `cell 3` for functions and `cell  4 ,5 ,6` for visualisation

Below are the visualisation of different features over data set.

`Spatial Fetaure` : 
Parameters Used :
Spatial Size : 32x32
Color Space : YCrCb
![alt text][image2]

`Histogram Fetaure` : 
Parameters Used :
Color Space : YCrCb
Histogram Bins : 32
![alt text][image3]

`Hog Fetaure` : 
Parameters Used :
Color Space : YCrCb
orientations : 9
pixels_per_cell : 32
cells_per_block : 
![alt text][image4]


#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various options of paramters manually and found few issues , below are the combinations and remarks 

![parameter][image7]

After which i came to use below Parameters for HOG and Other features
Color Space : YCrCb
Orientation : 9 
Spatial Size : 32x32
Histogram Bins : 32
Pixels Per Cell and Cells Per Block : 8x8 , 2x2

The final parameter values can be seen under `cell 7`.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

For this project , i used the same classifer as discussed in Udacity Code i.e `LinerSVC` . 
Used various features with paramters values as described in above block to train my classifer. I tried various combination of features along with HOG to achive maximum Test Accuracy , i mainly played with color space , I came to the conclusion of using YCrCb as final color space as with HLS and HSV color space , only white vehicle and light areas were detected more prominantly causing more false negative , where as with YUV color space combination dark color objects were more dominating , on using YCrCb color space , dark objects were still more dominating than light objects but detection was better than other color channels

To train my model i used all the three features , which calculated to be ` features` extracted from single image.

After this i read the images in `BGR` format using opencv so that the values lies between 0 to 255 and in case of color space changes no scaling issue is encountered .
I looped through all vehicle and non-vehicle imaages and used ` function` to extract features from images passed.
Later on i created one hot vector for both vehicles and non-vehicle images which acted as labels or y for me.
Then i shuffled both features and ground truth variables using `sklearn.utils.shuffle` function
I used `sklearn.model_selection.train_test_split` function to split my data into Train and Test data in 80:20 ratio.

Below were the insights 

`Initiate Feature Extraction Process`
`Car Feature Shape : (8460,)`
`Total No. of Images Processed : 8792`
`Non-Car Feature Shape : (8460,)`
`Total No. of Images Processed : 8968 `

Training and Test Data Details
`Total Data (car & non-car) :  17760`
`Total Ground Truth Data (car & non-car) :  17760`
`Total Train Data (car & non-car) :  14208`
`Total Ground Truth Data (car & non-car) :  14208`
`Total Test Data (car & non-car) :  3552`
`Total Ground Test Data (car & non-car) :  3552 `

Then i normalised my data using scaler function `StandardScaler()` and normalized both training and test data as part of preprocessing step.
After normalizing the data i trained my classifer using `LinearSVC` and default paramters , it returned an Test Accuracy of .9862

Then i saved the parametrs used in feature extraction , classifer and scale factor to be used later in pipeline , so that need to do training again and again , every time i open my jupyter notebook.

All above code can be seen from `cell 8 to 11`


### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I took help from Udacity Code to implement sliding window search in method slide_window. This method returns the windows list based on window size and search start and stop loaction under both x and y axis. 
I decided to use three windows of different size starting from smaller and then ranging to bigger so that cars which are ahead are also detcted alsong with the cars which are much near.

Also i decided to use only the lower half of the image and ignore anything above horizon such as sky and hills so that no of windows can be reduced and processing time can be improved.

I took three windows with size 64x64 , 96x96 , 128x128 each having differnt Y start and stop values , because car which are in small size are near to horizon and cars in large size are near to bottom of image.

Also have tried various overlapping percentage such as (80% , 70%) all are varied for different windows.

This leads to `253 windows` to be searched upon.

Code can bee seen under `cell 12 , 13 , 14`
![alt text][image5]



#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

I defined by pipeline under function  `pipeline` . Below is the snapshot of pipeline working on test image.
Code can been seen under `cell 15,16,17`

![alt text][image6]


I have also applied storing previous bounding boxes of passed frames which are then passed to heat map every time , leading to good heat map and stable boxes on detected objects . To improve the processing performance of video have only processed 1 random frame from every 5 frames passed and using old boxes when frame is not processed , this helped reduce false positives too.

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result][video1]

Code under `cell 18`

#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

As already described in earlier points have used heatMap logic to filter fasle positive.
I recorded the positions of positive detections in processed frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  Also have added logic to store previous frames result of bounding boxes and passsing them through through heat map function and generate more heat and apply higher threshold to remove false positives.

This can be seen under `cell 15,16 `


![alt text][image6]


---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Problems Faced : 
1. Most prominent problem faced was to decide parameters for classifer which give good test accuracy and also predicts cars properly throughout the video. I struggled between various color spaces as YUV , HLS and YCrCb color spaces as all gave good accuracy but one or the another was not able to identify either white or black car . Then i finally used YCrCb which is able to detect dark objects prominantly but not light object , but it went well throughout the video.

2. Also faced issue in improving the processing time of video as earlier it took around 3 hrs to process the video due to low processing power systems . This created issues to test on various parameters , also test video was not helpful as it contained only 1 sec frame. so to over come this the restricted my search to right side. 

3. Also was confused which would work better , either taking mean of last few heatmaps of the frame to store bounding boxes of last few frames and then passing them to heatmap , after so much hit and trial i came to conclusion of not using mean of heat map . The differnce was minor . Have also uploaded [video][videodescussion] of that for reference , please don't consider as submitted , this is just for reference.

Pipeline :

The pipeline defined is video related rather than generic , and would fail if another car comes in-front of the car abruptly.
Could have used Windows Sub-Sampling code as described in Udacity course to see is it can be generalized and processing time can be reduced.
