## BERTopic Topic Extraction Model

### Purpose :
Model to extract meaningful segmentations of a query dataset

### Testing the model deployment :
To run for testing of the model for topic head generation, follow the given below steps:

- Git clone the repo
- Go to current folder location i.e. ``` cd src/topic_modelling/BERTopic ```
- Create docker image file and test the api:
#### (IMP) The input .csv file must have one column having preprocessed text and column name as 'text'
'''
docker build -t testmodel .
docker run -p 8000:8000 testmodel
curl -X POST -F "test.csv"  http://localhost:8000/embed -o output4.csv
'''
