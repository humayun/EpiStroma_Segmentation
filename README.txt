# Epithelium and Stroma Segmentation Framework

This framework segments H & E stained histopathology images (Breast Cancer) into Hematoxylin (H) - Epithilum and Eison (E) - Stroma regions. This framework is supervised machine learning (based intesity distribution).

This code reads all the images from a given path and segments the epithelium and stroma pixels.
    Input: Path to the folder
    Output: Binary image files containing Epithelium and stroma
            Color image 
    Example Usage: H_and_E_segmentation.py "CodePath" "SourceDataPath" "DestinationDataPath" "FileExtension"
                 : In Canopy: %run "H_and_E_segmentation.py"
                    "EpiStroma_Segmentation/Code/" 
                    "InputFolder/" 
                    "OutputFolder/" 
                    "png"
    This code works the best when image resolution is approximately 3 mu m/pixel.

