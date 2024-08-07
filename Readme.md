# Speech Emotion Recognition (SER)

## Overview

he project uses three different datasets: TESS, SAVEE, and RAVDESS. Each of these datasets contains short audio clips of people, with each audio clip labeled according to one of the following emotions: anger, disgust, fear, happiness, pleasant surprise, or sadness. Our goal is to create a model that can recognize the emotions of a person based on the intuition derived from the audio clip, without relying on the spoken text. The algorithms used include:

- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- Decision Tree
- Random Forest
- Neural Network

## Documents

[Explanation of the project](/docsPro.pdf)

## Results

The performance of the classification algorithms is detailed in the results/ directory. This includes a comparison of the algorithms based on 20 runs, with the best accuracy being 86%, achieved with the K-Nearest Neighbors model.

## Datasets

The project uses three combined datasets, each containing audio files of people and labeled according to emotions.
Each recording is labeled according to one of the following emotions: anger, disgust, fear, happiness, pleasant surprise, sadness, and neutral.
1. **Toronto emotional speech set (TESS)**: 
   - Kaggle Link: [tess Dataset](https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess) The dataset consists of audio clips from females only, with a total of 2,800 audio files. 

2. **Surrey Audio-Visual Expressed Emotion (SAVEE)**: 
   - Kaggle Link: [savee Dataset](https://www.kaggle.com/datasets/ejlok1/surrey-audiovisual-expressed-emotion-savee) The dataset consists of audio clips from males only, with a total of 480 audio files.

3. **RAVDESS Emotional speech audio**: 
   - Kaggle Link: [ravdess Dataset](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio) The dataset contains 24 professional actors (12 women, 12 men), with a total of 1,440 audio files.


## Running the Project

Follow these steps to run the project:

1. **Download Datasets**: Download the datasets to a new folder called data (with the names- "TESS","SAVEE" and "RAVDESS").
2. **Feature Extraction**: Execute the feature extraction scripts within the `src/feature` directory.
3. **Training and Evaluation**: Proceed to `src/ml_algorithms` and run the script for the desired machine learning algorithm.

## Contributors

- Ben Dabush
- Naama Shiponi
