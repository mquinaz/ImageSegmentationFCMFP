# imgSeg-fcm-fp
Implementation of FCM and FCMFP algorithms for image segmentation. Second part of the work is a calculation of milk estimation on breast of women using thermographic images.

kmeans.py is a Kmeans algorithm using the axis information(spacial context), in the latter programs it was changed into color spaces(basically just changing the Euclidean distance calculation)

proj.py FCM algorithm and can calculate milk-estimation by commenting the appropriate lines of code (RGB color space)

projfp.py FCMFP algorithm, calculation of best cluster choice with Xie Beni measure

projetofp.py FCMFP algorithm for the milk estimation

classifica.py calculates the segmentation accuracy

FCM algorithm was tested also with CIELAB colorspace which obtained bad results.
