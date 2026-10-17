# Multidimensional Evaluation of Explainable AI for Five-Class Diabetic Retinopathy Grading

A comparative study of XAI techniques for Diabetic Retinopathy detection using deployable detection models such as a ResNet50 and an EfficientNetB0. XAI techniques used include Grad-CAM, LIME and SHAP.

All computations were performed on Hippo at the University of KwaZulu-Natal. All our code was run Python/3.8.6. If you are running your code using UKZN Hippo, run this command: "module load Python/3.8.6-GCCcore-10.2.0". All python notebooks that have been executed have a corresponding batch script that was used to execute it. All batch scripts can be found in the saame directory as their corresponding notebook. If you are runing your code on HIPPO (UKZN), you will need to convert your batch scripts to Unix/Linux format using the command "dos2unix example.sh", and please ensure you set the path for the working directory, virtual enviornment and path to notebook to match your setup. 

Requirements: You will find a requirements.txt that has all the necessary libraries that have to be installed. We recommend you create a virtual enviornment before installing all these libraries. Once you have activated your virtual enviornment, the command "pip install -r requirements.txt" should install most things. Additionally run the command "pip install opencv-python-headless" to install opencv-python-headless. Now you should have all the libraries installed.

Two datasets were used in this study:
- [The APTOS 2019 Blindness Detection dataset](https://www.kaggle.com/competitions/aptos2019-blindness-detection/data)
- [Refined IDRiD dataset](https://zenodo.org/records/17615903)

There are two main stages in our code, that being:
- Model training 
- XAI evaluation

The order of execution of all the notebooks we recommend is as follows (the batch scripts used to execute them is in bracket next to each notebook): 
ResNet.ipynb (ResNet.sh) -> EffNet.ipynb (EffNet.sh) -> IDRiD_preprocessing.ipynb (preprocess.sh) -> FinalLocalFaith.ipynb (finallocalandfaith.sh) ->   
MaxSens.ipynb (maxsens.sh) -> RISEDisplay.ipynb (RISEDisplay.sh) -> confidence.ipynb (confidence.sh)

Additioanlly, to run all the code, you will need the AttributionEngine.py and MemoryManagement.py files that can be found in the xai_lib folder. Ensure that you have the xai_lib folder (containing AttributionEngine.py and MemoryManagement.py) in the same directory as the notebooks. For example, this is how we had it setup using HIPPO (UKZN): 

```text
/home/{name}/
├── xai_lib/
│   ├── AttributionEngine.py
│   └── MemoryManagement.py
├── ResNet.ipynb
├── ResNet.sh
├── EffNet.ipynb
├── EffNet.sh
├── IDRiD_preprocessing.ipynb
├── preprocess.sh
├── FinalLocalFaith.ipynb
├── finallocalandfaith.sh
├── MaxSens.ipynb
├── maxsens.sh
├── RISEDisplay.ipynb
├── RISEDisplay.sh
├── confidence.ipynb
└── confidence.sh
```

Model Training:
Before running model training notebooks (ResNet.ipynb and EffNet.ipynb), if you are using HIPPO, make sure you have the pretrained weights saved in the torche_cache folder in your data directory.The ResNet50 and EfficientNetB0 models start from pretrained weights, which PyTorch downloads and caches under $TORCH_HOME/hub/checkpoints/. On HIPPO the compute nodes may not have internet access, so download the weights once on the login node before submitting the training jobs. Choose a cache directory and set TORCH_HOME. For example, /data/aveera/torch-cache, so replace this with your own path.

To run ResNet.ipynb and EffNet.ipynb, please ensure you set the paths for variables DRIVE_MODEL_DIR, TRAIN_DF_DIR, TRAIN_IMAGES_DIR and CROPPED_IMAGE_DIR with your own paths.

To run IDRiD_preprocessing.ipynb, ensure you set the variables PREPROCESSED_DIR and IDRID_DIR with your own paths.

To run FinalLocalFaith.ipynb, ensure you set the variables PREPROCESSED_DIR, DRIVE_MODEL_DIR, LOCALIZATION_SAVE_DIR, SAVE_DIR, RESULTS_DIR and FIGURES_DIR with your own paths.

To run MaxSens.ipynb, ensure you set the variables IDRID_DIR, drive_model_dir. RESULTS_DIR and CHECKPOINT_DIR with your own paths.

To run RISEDisplay.ipynb, ensure you set the variables RESULTS_DIR to your own path.

To run confidence.ipynb, ensure you set the variables DATA_DIR, OUTPUT_DIR, IROF_FILES, RISE_FILE and MAXSENS_FILE with their own paths and names.

Comments accompanying the variables will tell you what the paths represent.

All results can be found in the github repository: 