--This repository includes 4 parts for achieving face detection and recognition for custom people:

1. The 'dataset_gen.py' is used for generating dataset for training . This code will use 'haarcascade_frontalface_default' for face detection.

2. The 'training.py' is the code which the training model is designed and the structure uses MTCNN and SVM for training on the dataset generated.
This code will give two output files as 'NPZ' and the model is saved as 'PKL' format.

3. The 'fps_demo' is used to improve the FPS of the code runing in real-time in 'main.py'.

4. The 'main.py' is used for runing the real time face detection and recognintion based on the files generated in training part.

Note: the dataset folder should have the following structure:

ex. FOLDER FOR PERSON1 IMAGES
    FOLDER FOR PERSON2 IMAGES
    .
    . 
    .
    FOLDER FOR UNKNOWN IMAGES( This folder you can give 4,5 peoples imges mixed or you can put more people for better accuracy of distinguishing the known from unknown))
