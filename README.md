# imgSeg-fcm-fp
Implementation of FCM and FCMFP algorithms for image segmentation. Second part of the work is a calculation of milk estimation on breast of women using thermographic images.

kmeans.py - Kmeans algorithm using the axis information(spacial context), in the latter programs it was changed into color spaces(basically just changing the Euclidean distance calculation)

proj.py FCM algorithm and can calculate milk-estimation by commenting the appropriate lines of code (RGB color space)

fcmfp.py - FCMFP algorithm, calculation of best cluster choice with Xie Beni measure

accuracy.py - calculates the segmentation accuracy

FCM algorithm was tested also with CIELAB colorspace which obtained bad results.
