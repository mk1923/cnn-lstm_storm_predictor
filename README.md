# Team Yolanda for The Day After Tomorrow Challenge

GitHub respository by Boyang Hu, Manawi Kahie, Zeyi Ke, Tao Lin, Anna Smith, Ardan Suphi, and Peifang Tan

 <p align="left">
  <img src="additional files/ese_msc.png" alt="ese_msc icon">
</p>

## Table of Contents

1. [ Introduction ](#intro)
2. [ User Instructions ](#usage)
3. [ Tour of the Repository ](#repo)
4. [ Reading List ](#resources)

<a name="intro"></a>
## Introduction

### Forecasting tropical cyclone behaviour through Deep Learning

 <p align="left">
  <img src="additional files/360px-FEMA_logo.png" alt="FEMA logo">
</p>

In this GitHub repository, we present our submission to the (fictional) Federal Emergency Management Agency's (FEMA) *The Day After Tomorrow Challenge*. This challenge comes in response to the immense human and economic costs inflicted by tropical cyclones globally, with upwards of 1,000 deaths and $50 billion in damages inflicted by a single cyclone event. Deep Learning methods can help us better predict future outcomes of tropical cyclones, and with our submission we hope to help inform the development of successful impact mitigation and response strategies.

 <p align="left">
  <img src="additional files/cyclone.jpg" alt="cyclone">
</p>

The FEMA *The Day After Tomorrow Challenge* presented us with two tasks:
1. **Task 1:** Given data from a surprise test storm, where some satellite images are available, generate a ML/DL-based solution able to generate 3 future image predictions
2. **Task 2:** Train a network to predict wind speeds using provided satellite image and feature data (10 wind speeds), as well as for samples where satellite images were generated using the Task 1 model (3 wind speeds),

In response, Team Yolanda has developed two Deep Learning models:

1. **Task 1 Solution:** An CNN-ConvLSTM image generation model to predict the evolution of tropical cyclones, based on previously collected satellite image data
2. **Task 2 Solution:** A CNN-LSTM model that can predict the wind speed of a tropical cyclone based on satellite imagery data

The notebooks in our repository document our model architectures, training procedures, and task solutions.


<a name="usage"></a>
## User Instructions

### Cloning the repository

To clone our GitHub repository, run the following code in your terminal:
 
```
$ git clone https://github.com/ese-msc-2023/acds-the-day-after-tomorrow-yolanda.git
```

### `yolanda` conda environment

If you wish to run our notebooks locally, you can use our `yolanda` conda environment. To install, navigate to your cloned repository and run the following code in your terminal:

```
$ cd acds-the-day-after-tomorrow-yolanda
$ conda env create -f environment.yml
$ conda activate yolanda
```

Our notebooks can also be run in Google Colab, in conjunction with the training and test data prodivded in the Google Drive folders below.


### Data Access

The training data used for this project can be found here:
https://drive.google.com/drive/folders/1tFqoQl-sdK6qTY1vNWIoAdudEsG6Lirj?usp=drive_link

The surprise storm (test) data is available here:
https://drive.google.com/drive/folders/1HYHH7kBirVto9640I67Qh53Od_98Zo5z?usp=sharing

<a name="repo"></a>
## Tour of the Repository

Our repository contains the following key folders and files:


- `yolanda_generatedimages` folder contains three .jpg files for our submission for Task 1
- `Yolanda_windpredictions.csv` contains our predicted wind speeds for Task 2
- `Surprise_storm.ipynb`: Our notebook showing the implementation of our models to predict the missing data for the test surprise storm
- `wind_predictor` folder:
    - `Wind_notebook.ipynb`: Notebook demonstrating the training of our wind speed prediction model
- `image_predictor` folder:
    - `Image_notebook.ipynb`: Notebook demonstrating the training of our generative storm image model
- `additional files` folder:
    - `Training_Data_EDA.ipynb`: Notebook documenting our training data EDA
- `references.txt`: Links to external sources consuted in the development of our solution
- `LICENSE.md`: Information regarding the usage of our code and repository


<a name="resources"></a>
## Reading List

[Link to our solution pitch slides](https://docs.google.com/presentation/d/1GiOz1dMnFNclHVwYLGggZjaWkz-DhBHl7aA62n9FFP0/edit?usp=sharing)

[Link to the challenge briefing slides](https://imperiallondon.sharepoint.com/:p:/r/sites/TrialTeam-EA/Shared%20Documents/General/TheDayAfterTomorrow-presentation%202.pptx?d=wdf1d9e0210264eab88858e2353a36242&csf=1&web=1&e=XoU1Am)
