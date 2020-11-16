### Code for the COLING 2020 paper *Language-Driven Region Pointer Advancement for Controllable Image Captioning (Lindh, Ross and Kelleher, 2020)*

If you find this project useful or learned something from it, please cite our paper. :)

---

##### DEPENDENCIES
 * Python 3.5.3 or later. (Needs to be a build that supports torchvision.)
 
 * unzip (apt-get install unzip if it's missing)
 
 * Around 13 GB of disk space. (For the training data and model checkpoint files. If you wish to reproduce the preprocessing steps you will need additional space.) 
 
---
 
##### STEPS TO REPRODUCE

The following steps will download the preprocessed features and launch the training and evaluation of the model. See further down this page how to reproduce the features.

1) Download the source code:
    > git clone https://github.com/AnnikaLindh/Controllable_Region_Pointer_Advancement.git

2) (Optional) Create and activate a new virtual environment for this project to ensure none of the requirements clash 
with your other projects.

3) Install the required Python libraries:
    > pip3 install -r requirements.txt   

4) Run the setup script to download data files and the fully trained model:
     > bash setup.sh

5) (Skip this step if you wish to use the pre-trained model under results/CIDEr_model.pth.) Go to the src folder and start the training. If you do not have CUDA, please add --device='cpu:0' to run on the CPU instead. (See src/model/parameter_parsing.py for more settings.)

    This step might take days to finish. You can lower the default number of epochs from 300 via the --num_epochs option. The best CIDEr validation score is expected at around 250 epochs so any training beyond this is not necessary to achieve similar results to the paper.
    The current epoch number will be printed to the console at the start of each epoch.
    > cd src \
    python3 model/main.py --mode=train

6) When training has finished, find the path to the automatically created checkpoint dir where your trained model has been saved:
   > ls ../cg_checkpoints

7) Replace the path after --load_path_cg below and run the inference over the test set to evaluate the model and generate captions for all test examples:
   > python3 model/main.py --mode=test --load_path_cg=../cg_checkpoints/caption_generator_YYYYMMDD_XXXX --use_all_regions=enforced --block_unnecessary_tokens=True --metrics="['BLEU-1', 'BLEU-2', 'BLEU-3', 'BLEU-4', 'ROUGE-L', 'METEOR', 'CIDEr', 'SPICE']"

    The evaluation results on the standard metrics will be printed to the console. The full set of captions can be found in CIDEr_CAPTIONS.json under the same directory as the model file.
    
    To calculate the diversity metrics, see: src/evaluate/diversity.py

---

##### REPRODUCING THE POS TAG STATISTICS, RUNNING PREPROCESSING STEPS AND THE STANDALONE SCRIPT

You will be using the following files from this repository:
 * POS TAGS: chunk_stats.py under src/data_exploration/
 * PREPROCESSING: launch_bua_preprocessing.py and launch_entity_preprocessing.py under src/preprocessing/
 * STANDALONE: model/standalone.py

###### POG TAG CHUNK STATS

For chunk stats you need to pip3 install textblob and download NLTK data. You will also need the Flickr30kEntities Sentences file.

###### PREPROCESSING

For preprocessing you will need to download the flickr30k images and the Entities Annotations and Sentences.
You will need to setup and train the bottom-up net according to the instructions here: https://github.com/peteanderson80/bottom-up-attentionon
and using the splits without data contamination for the Flickr30k splits used in the paper (these splits can be downloaded from: https://drive.google.com/uc?id=1jjaJGsX2q7gZIaPAjuTOe_S0myjA3Wmo).
The bottom-up net instructions includes how to build and install a specific version of caffe which is also needed when loading the bottom-up net
during preprocessing and standalone use.

###### STANDALONE

With the standalone.py script you can test the model on a custom image with custom (pre-defined) regions.
You will need to follow the instructions from PREPROCESSING above to setup and train the bottom-up net and its 
corresponding caffe version.
