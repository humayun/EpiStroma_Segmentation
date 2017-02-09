__author__ = "Sindhu Ghanta, Humayun Irshad"
__copyright__ = "Copyright 2015, Becklab"
__license__ = "Becklab"
__version__ = "1.0.1"
__maintainer__ = "Sindhu Ghanta, Humayun Irshad"
__email__ = "sghanta2@bidmc.harvard.edu, hirshad@bidmc.harvard.edu"
__status__ = "Completed"

''' This code reads all the *.jpg images from a given path and segments the epithelium and stroma pixels.

    Input: Path to the folder
    Output: Binary image files containing Epithelium and stroma
            Color image 
    Example Usage: H_and_E_segmentation.py "CodePath" "SourceDataPath" "DestinationDataPath" "FileExtension"
                 : In Canopy: %run "H:/CodingStuff/Python/EpiStroma_Segmentation/Code/H_and_E_segmentation.py"
                    "H:/CodingStuff/Python/EpiStroma_Segmentation/Code/" 
                    "R:/Beck Lab/Atypia_Andy/ImageProcessing/Pathologist_2/" 
                    "R:/Beck Lab/Atypia_Andy/ImageProcessing/Pathologist_2/EpiStroma/" 
                    "png"

    This code works the best when image resolution is approximately 3 mu m/pixel.
'''
import sys
import glob
import os
from skimage import io, segmentation, util
import numpy as np
import pickle

def ensure_dir(f):
    d = os.path.dirname(f)
    if not os.path.exists(d):
        os.makedirs(d)

def getBackgroundPixel(RedArray,GreenArray,BlueArray):
    PIXEL_INTENSITY_SUM_THRESH = 0.9
    imageGreen_new = GreenArray[RedArray>PIXEL_INTENSITY_SUM_THRESH]
    imageBlue_new = BlueArray[RedArray>PIXEL_INTENSITY_SUM_THRESH]
    imageBlue_new1 = imageBlue_new[imageGreen_new>PIXEL_INTENSITY_SUM_THRESH]
    imageBlue_new2 = imageBlue_new1[imageBlue_new1>PIXEL_INTENSITY_SUM_THRESH]
    return len(imageBlue_new2) 


FRACTION_ONES_THRESH = 0.3  
def processImage(imagePath,clf):
    
    # Read the image  
    color_original = util.img_as_float(io.imread(imagePath))
    segments = segmentation.slic(color_original, n_segments = 150, compactness=25, max_iter=50, sigma = 5, enforce_connectivity=True)
    # Save the dimensions of the image
    imageShape = color_original.shape
    # Extract mean intensity value for each channel
    Epithelium = np.zeros((imageShape[0], imageShape[1]))
    Stroma = np.zeros((imageShape[0], imageShape[1]))
    ColorImage = np.zeros((imageShape[0], imageShape[1], imageShape[2]))
    
    # Calculate the class of each superpixel
    for segmentNum in range(0,segments.max()):
         
        newArray = color_original[segments==segmentNum]
        
        ## Error - Program crash because of remove zero values (from frontend or backend) in one array
        #RedArray = np.trim_zeros(newArray[:,0])
        #GreenArray = np.trim_zeros(newArray[:,1])
        #BlueArray = np.trim_zeros(newArray[:,2])        
        
        RedArray = newArray[:,0]
        GreenArray = newArray[:,1]
        BlueArray = newArray[:,2]
        
        if (np.isnan(np.mean(RedArray)) | np.isnan(np.mean(GreenArray)) | np.isnan(np.mean(BlueArray))):
            print(np.mean(RedArray),np.mean(GreenArray),np.mean(BlueArray), " Skip the superpixel ... ")
            continue

        fractionOnes = float(getBackgroundPixel(RedArray,GreenArray,BlueArray))/len(newArray[:,0]) 
        indexPixels = np.where(segments==segmentNum)
        
        # If fraction of white pixels is too high, assume that it is background or fat and leave it out without processing
        if(fractionOnes<FRACTION_ONES_THRESH):
            classLabel = clf.predict([np.mean(RedArray),np.mean(GreenArray),np.mean(BlueArray)])     
                                 
            if(classLabel == 1): # Its epithelium
                Epithelium[indexPixels[0],indexPixels[1]] = 1
                ColorImage[indexPixels[0],indexPixels[1],0] = 1 
                
            else:  # Its Stroma
                Stroma[indexPixels[0],indexPixels[1]] = 1
                ColorImage[indexPixels[0],indexPixels[1],1] = 1 
                
    return (Epithelium, Stroma, ColorImage)

        
def processFolder(DataPath, FolderPath, WritePath, Extension):
    try:
        ensure_dir(WritePath)
    except:
        print("Error: " + WritePath + " wrong path ... ")

    # Load the SVM parameters
    clf = pickle.load(open(DataPath+"H_EtextureGMM.p"))
    
    files = sorted(glob.glob(FolderPath+'*.'+Extension))

    # Process all the JPEG images in the folder given by user   
    i = 0
    for file in files:
        i = i + 1
        path, filename = os.path.split(file)

        if os.path.isfile(WritePath+filename[0:len(filename)-4]+"_ColorImage.jpg"):
            print("\n" + str(i) + " - Already Processed: " + filename)
            continue
        else: 
            print("\n" + str(i) + " - Processing: " + filename)

        if i == 1219:
            print("\n" + str(i) + " - Damaged Image: " + filename)
            continue
        
        # Call the function to process the image
        [Epithelium, Stroma, ColorImage] = processImage(file,clf)
        
        # Save the segmented image
        io.imsave(WritePath+filename[0:len(filename)-4]+"_Epithelium.jpg",Epithelium)
        io.imsave(WritePath+filename[0:len(filename)-4]+"_Stroma.jpg",Stroma)
        io.imsave(WritePath+filename[0:len(filename)-4]+"_ColorImage.jpg",ColorImage)

                
if __name__ == "__main__":

    # Check the WSIs Path
    if not os.path.exists(os.path.dirname(sys.argv[1])):
        print("Error: " + sys.argv[1] + " directory is not exist ... ")
    else: 
        if not os.path.exists(os.path.dirname(sys.argv[2])):
            print("Error: " + sys.argv[2] + " directory is not exist ... ")
        else:
            processFolder(sys.argv[1],sys.argv[2],sys.argv[3], sys.argv[4])