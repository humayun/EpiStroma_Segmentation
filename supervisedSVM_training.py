# Create a plot of accuracy for different superpixel size and compactness factor
# Supervised mode with 6 fold validation

# Divide all the images into superpixels. Extract the mean intensity values and
# true label for them.
# Use the LeaveOneLabelOut to select index of training and testing sets. Measure accuracy in each case
# and use the mean value

from __future__ import print_function
from skimage import graph, data, io, segmentation, color, util
import numpy 
from sklearn.cluster import KMeans
from skimage.filter import roberts, sobel
from mpl_toolkits.mplot3d import Axes3D
import glob
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib
from sklearn.cross_validation import KFold
from sklearn.svm import SVC
import pickle

path1 = "C:/Users/sghanta2/Dropbox/BeckLab/SupplementaryMaterial/EpiStromaTrainingImages/3002564s3/NKI_Training/"
path2 = "C:/Users/sghanta2/Dropbox/BeckLab/SupplementaryMaterial/EpiStromaTrainingImages/3002564s3/NKI_Label/"
os.chdir(path1)

numSegments = [x for x in range(50, 200, 20)]
compactnessValue = [x for x in range(10, 50,5)]
#numSegments = numpy.linspace(20,200,2)
#compactnessValue = numpy.linspace(5,50,2)
X,Y = numpy.meshgrid(numSegments,compactnessValue)
ZAccuracyM =numpy.zeros((X.shape[0],X.shape[1]))

a = 6
b = 3
arrayAccuracy = []
Label =[]
arrayKmeans = [[]]
for file in glob.glob("*.jpg"):
    image = util.img_as_float(io.imread(path1+file))
    # Load the label data
    imageLabel = util.img_as_float(io.imread(path2+file[2:len(path2)-1]))
    # Segment the image into superpixels
    print(file)
    segments = segmentation.slic(image, n_segments = X[a][b], compactness=Y[a][b], max_iter=50, sigma = 5, enforce_connectivity=True)
    # Save the dimensions of the image
    imageShape = image.shape
    # Extract mean intensity value for each channel
    SegStromaEpithelium = numpy.zeros((imageShape[0], imageShape[1], imageShape[2]))
    for segmentNum in range(0,segments.max()):
        newArray = image[segments==segmentNum] 
        newArrayLabel = imageLabel[segments==segmentNum]    
        
        RedArray = numpy.trim_zeros(newArray[:,0])
        GreenArray = numpy.trim_zeros(newArray[:,1])
        BlueArray = numpy.trim_zeros(newArray[:,2])
        
        RedArrayLabel = newArrayLabel[:,0]
        RedArrayLabel[RedArrayLabel>0.5] = 1 
        RedArrayLabel = numpy.trim_zeros(RedArrayLabel)
        
        GreenArrayLabel = newArrayLabel[:,1]
        GreenArrayLabel[GreenArrayLabel>0.5] = 1
        GreenArrayLabel = numpy.trim_zeros(GreenArrayLabel)
        
        BlueArrayLabel = numpy.trim_zeros(newArrayLabel[:,2])     
                            
        # Determine the class of this super pixel        
        if(numpy.mean(RedArrayLabel)>numpy.mean(GreenArrayLabel) and numpy.mean(RedArrayLabel)>0.5):
            Label.append(1)
            arrayKmeans.append([numpy.mean(RedArray),numpy.mean(GreenArray),numpy.mean(BlueArray)])
        elif(numpy.mean(GreenArrayLabel)>numpy.mean(RedArrayLabel) and numpy.mean(GreenArrayLabel)>0.5):
            Label.append(2)
            arrayKmeans.append([numpy.mean(RedArray),numpy.mean(GreenArray),numpy.mean(BlueArray)])
del arrayKmeans[0]

# Divide the dataset into
kf = KFold(numpy.size(Label), n_folds=6)
Accuracy = []
for train_index, test_index in kf:
    print("TRAIN:", train_index, "TEST:", test_index)
    X_test =[]
    X_train =[]
    y_test = []
    y_train = []
    for testSampleIndex in range(0,len(test_index)):
        X_test.append(arrayKmeans[test_index[testSampleIndex]])
        y_test.append(Label[test_index[testSampleIndex]])
    
    
    for trainSampleIndex in range(0,len(train_index)):
        X_train.append(arrayKmeans[train_index[trainSampleIndex]])
        y_train.append(Label[train_index[trainSampleIndex]])
                
    clf = SVC()
    clf.fit(X_train, y_train)
    y_predict = clf.predict(X_test)
    errorPercent = float(len(y_test)-numpy.sum(numpy.absolute(y_predict-y_test)))/len(y_test)
    Accuracy.append(errorPercent)
ZAccuracyM[a][b]=numpy.mean(Accuracy)
print(ZAccuracyM[a][b])

fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, Y, ZAccuracyM)
plt.show()

clf2 = SVC()
clf2.fit(arrayKmeans,Label )

pickle.dump( clf2, open( "H_EtextureSVM.p", "wb" ) )

                