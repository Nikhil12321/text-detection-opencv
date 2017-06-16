# Text detection in natural scenes using opencv

A simple implementation using C++ and python.

## Structure

1. Your file of interests are detect.cpp, classify.py and test.py. The working is a little weird and the control jumps between detect.cpp to the test.py.

2. The entire process works on detect.cpp except the classification part which is in the classify.py and test.py files.

3. I have already trained an SVC classifier on training images picked up from ICDAR dataset which is stored in svc.pkl which can classify HOG (Histogram of Oriented Gradients) descriptors of regions to check whether they contain texts.

4. The opencv HOG is used with the parameters below

```cpp
HOGDescriptor hog( Size(dim, dim), Size(4, 4), Size(2, 2), Size(2, 2), 2 );
```

Or you can write your own classifer and change the text.py so that it can properly classify the text regions we pass onto it.


## What do I need to run this?

1. OpenCV 2.4
2. Scikit-learn
3. cmake

## Algorithm

1. Detects potential text regions using Maximally Stable Extremal Regions (MSER)
2. filters the regions using classifier.
3. Redundant boxes (one inside the other are removed)
4. the nearby boxes are combined

## Where is the detected text?
You can send the detected text boxes to any text recognition library like Tesseract and can easily get the text in the boxes.

## How to train your own classifier?

1. You need your own text dataset. Go to ICDAR website and download their latest training dataset. It comes with another text file which contains regions in the form of end points of the rectangle of where the text is present in the picture.
I have included a sweet function getPoints which accepts a line of the text file containing points and returns a Point variable. The function is in the detect.cpp file

2. Now that you have the text regions, train the classifier on these regions using any feature detector. I used HOG.

3. These are the positive regions. For negative regions, you can take the same dataset and pick up random rectangles from the images and take them as negative samples. Make sure your data is now skewed.

4. Now that you have features (positive and negative), train any classifier on it and save it a pickle.

5. To extract positive and negative features, I have included two functions readFilesPostive() and readFilesNegative(). They accept location of images and trains a HOG classifier on them and saves all the features in two files positive.txt and negative.txt. You need to change them in accordance to your data.
