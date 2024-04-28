Welcome to our final year project 

Here i am trying to train the model using Decision Tree (RandomClassifier) Method 

To Train the model simply follow the folling instructions 
1. Create a virtual environment and activate the virtual environment

2. Install the important libraries using `requirements.txt`
```
pip install -r ./requirements.txt
``` 
3. First we need to collect the data or either create the data 

4. Check if the ./data folder is present or not if not man you need to create the data own 
download and unzipe  this [download](https://drive.google.com/file/d/19ijZTMI2btgIV8xDCnRv8jlFKBMZAo8D/view?usp=drive_link)

5. If not present run the command file `data_collection.py`

6. If the program runs perfectly it will create dataset for you if you press 1 it will collect data for the coresspoding images you may manipulate the code to store more images 

7. Run the file it will serialize the dataset and make sure the data consists of equal no of datapoints 

8. After that if everything is correctly followed you might able to see a new file named `data.pickle` serialized dataset 

9. Run the file `train_model.py`  check the terminal and its accuracy 

10. After that run you will able to see a new file `model.p` which is our trained model and to verify our model run the program `test_model.py` you may change labels in this file for specific directory 

Thats all  btw mine accuracy was 98.64 