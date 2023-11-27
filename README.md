Team Name: Sivatagi Rohamcsigák
Participants: 
	Bodai Adrián Tibor - OU1G79 
	Hermán Judit - L7D38R
	Kovács Kíra Diána - CXL05B

Name of the project: Landmark recognition based on Kaggle data
Link to Kaggle project: https://www.kaggle.com/competitions/landmark-recognition-2021
The description of the project:



We made both the model and evaluation in jupyter notebook, everything is well commented and documented. Please DO NOT RERUN the notebooks, because we have 100 GB data to work with, and we could not find a suitable location in the cload, to make it perfectly runable on the servers.



Welcome to the fourth Landmark Recognition competition! This year, we introduce a lot more diversity in the challenge’s test images in order to measure global landmark recognition performance in a fairer manner. And following last year’s success, we set this up as a code competition.

Have you ever gone through your vacation photos and asked yourself: What is the name of this temple I visited in China? Who created this monument I saw in France? Landmark recognition can help! This technology can predict landmark labels directly from image pixels, to help people better understand and organize their photo collections. This competition challenges Kagglers to build models that recognize the correct landmark (if any) in a dataset of challenging test images.

Many Kagglers are familiar with image classification challenges like the ImageNet Large Scale Visual Recognition Challenge (ILSVRC), which aims to recognize 1K general object categories. Landmark recognition is a little different from that: it contains a much larger number of classes (there are more than 81K classes in this challenge), and the number of training examples per class may not be very large. Landmark recognition is challenging in its own way.

This year's competition is structured in a synchronous rerun format, where participants need to submit their Kaggle notebooks for scoring. This is similar to the 2020 version of the competition. In older editions (2018 and 2019), submissions had been handled by uploading prediction files to the system.

This challenge is organized in conjunction with the Landmark Retrieval Challenge 2021. Both challenges will be discussed at the Instance-Level Recognition workshop in ICCV 21.


DATA:
ATTENTION: The data we used is 105GB, so we could not upload it anywhere, we did the training and every reformatting locally. 


The description of the data at the Kaggle page:
Dataset Description
In this competition, you are asked to take test images and recognize which landmarks (if any) are depicted in them. The training set is available in the train/ folder, with corresponding landmark labels in train.csv. The test set images are listed in the test/ folder. Each image has a unique id. Since there are a large number of images, each image is placed within three subfolders according to the first three characters of the image id (i.e. image abcdef.jpg is placed in a/b/c/abcdef.jpg).

This is a synchronous rerun code competition. The provided test set is a representative set of files to demonstrate the format of the private test set. When you submit your notebook, Kaggle will rerun your code on the private dataset. Additionally, this competition also has two unique characteristics:

To facilitate recognition-by-retrieval approaches, the private training set contains only a 100k subset of the total public training set. This 100k subset contains all of the training set images associated with the landmarks in the private test set. You may still attach the full training set as an external data set if you wish.
Submissions are given 12 hours to run, as compared to the site-wide session limit of 9 hours. While your commit must still finish in the 9 hour limit in order to be eligible to submit, the rerun may take the full 12 hours.
Dataset
For this year, we introduce a new test set which is sampled from many countries, increasing the diversity in worldwide representation. See our paper for more details.

The training data for this competition comes from a cleaned version of the Google Landmarks Dataset v2 (GLDv2), which is available here. Please refer to the paper for more details on the dataset construction and how to use it. See this code example for an example of a pretrained model.

If you make use of this dataset in your research, please consider citing:

"Google Landmarks Dataset v2 - A Large-Scale Benchmark for Instance-Level Recognition and Retrieval", T. Weyand, A. Araujo, B. Cao and J. Sim, Proc. CVPR'20

and specifically for this year's challenge:

"Towards A Fairer Landmark Recognition Dataset", Z. Kim, A. Araujo, B. Cao, C. Askew, J. Sim, M. Green, N. Yilla and T. Weyand, arxiv:2108.08874
