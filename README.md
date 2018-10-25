# Noise detection using SVM

## Requirements

```
pip install -r requirements.txt
```

## How to use

### Multiple folders and scripts are availables :


- **fichiersSVD/\*** : all scene files information (zones of each scene, SVD descriptor files information and so on...).
- **fichiersSVD_light/\*** : all scene files information (zones of each scene, SVD descriptor files information and so on...) but here with reduction of information for few scenes. Information used in our case.
- **models/*.py** : all models developed to predict noise in image.
- **data_svm/\*** : folder which will contain all *.train* & *.test* files in order to train model.
- **saved_models/*.joblib** : all scikit learn models saved.
- **models_info/*.md** : all markdown files generated to get quick information about model performance and prediction.

### Scripts for generating data files

Two scripts can be used for generating data in order to fit model :
- **generate_data_svm.py** : zones are specified and stayed fixed for each scene
- **generate_data_svm_random.py** : zones are chosen randomly (just a number of zone is specified)


**Remark** : Note here that all python script have *--help* command.

```
python generate_data_svm.py --help

python generate_data_svm.py --output xxxx --interval 0,20  --kind svdne --scenes "A, B, D" --zones "0, 1, 2" --percent 0.7 --sep : --rowindex 1
```

Parameters explained : 
- **output** : filename of data (which will be split into two parts, *.train* and *.test* relative to your choices).
- **interval** : the interval of data you want to use from SVD vector.
- **kind** : kind of data ['svd', 'svdn', 'svdne']; not normalize, normalize vector only and normalize together.
- **scenes** : scenes choice for training dataset.
- **zones** : zones to take for training dataset.
- **percent** : percent of data amount of zone to take (choose randomly) of zone
- **sep** : output csv file seperator used
- **rowindex** : if 1 then row will be like that 1:xxxxx, 2:xxxxxx, ..., n:xxxxxx

### Train model

This is an example of how to train a model

```python
python models/xxxxx.py --data 'data_svm/xxxxx.train' --output 'model_file_to_save'
```

### Predict image using model

Now we have a model trained, we can use it with an image as input :

```python
python predict_noisy_image_svd_lab.py --image path/to/image.png --interval "x,x" --model saved_models/xxxxxx.joblib --mode 'svdn'
```

The model will return only 0 or 1 :
- 1 means noisy image is detected.
- 0 means image seem to be not noisy.

### Predict scene using model

Now we have a model trained, we can use it with an image as input :

```python
python prediction_scene.py --data path/to/xxxx.csv --model saved_model/xxxx.joblib --output xxxxx --scene xxxx
```
**Remark** : *scene* parameter expected need to be the correct name of the Scene.

## Others scripts

### Test model on all scene data

In order to see if a model well generalized, a bash script is available :

```bash
bash testModelByScene.sh '100' '110' 'saved_models/xxxx.joblib' 'svdne'
```

Parameters list :
- 1 : Begin of interval of data from SVD to use
- 2 : End of interval of data from SVD to use
- 3 : Model we want to test
- 4 : Kind of data input used by trained model


### Get treshold map 

Main objective of this project is to predict as well as a human the noise perception on a photo realistic image. Human threshold is available from training data. So a script was developed to give the predicted treshold from model and compare predicted treshold from the expected one.

```python
python predict_noisy_image.py --interval "x,x" --model 'saved_models/xxxx.joblib' --mode ["svd", "svdn", "svdne"] --limit_detection xx
```

Parameters list :
- **model** : mode file saved to use
- **interval** : the interval of data you want to use from SVD vector.
- **mode** : kind of data ['svd', 'svdn', 'svdne']; not normalize, normalize vector only and normalize together.
- **limit_detection** : number of not noisy images found to stop and return threshold (integer).

### Display model performance information

Another script was developed to display into Mardown format the performance of a model.

The content will be divised into two parts :
- Predicted performance on all scenes
- Treshold maps obtained from model on each scenes

The previous script need to already have ran to obtain and display treshold maps on this markdown file.

```python
python save_model_result_in_md.py --interval "xx,xx" --model saved_models/xxxx.joblib --mode ["svd", "svdn", "svdne"]
```

Parameters list :
- **model** : mode file saved to use
- **interval** : the interval of data you want to use from SVD vector.
- **mode** : kind of data ['svd', 'svdn', 'svdne']; not normalize, normalize vector only and normalize together.


Markdown file is saved using model name into **models_info** folder.

### Others...

All others bash scripts are used to combine and run multiple model combinations...

## How to contribute

This git project uses [git-flow](https://danielkummer.github.io/git-flow-cheatsheet/) implementation. You are free to contribute to it.
