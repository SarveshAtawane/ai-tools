## NER:

### Purpose :

Model to detect

- crops
- pests
- seed type
- email
- time
- phone numbers
- numbers with units
- dates

### Testing the model deployment :

To run for testing just the Hugging Face deployment for grievence recognition, you can follow the following steps :

- Git clone the repo
- Go to current folder location i.e. ``cd /src/ner/agri_ner_akai/local``
- Create docker image file and test the api:

```
docker build -t testmodel .
docker run -p 8000:8000 testmodel
```

### **Request**

```
curl -X POST -H "Content-Type: application/json" -d '{
"text": "What are tomatoes and potaotes that are being attacked by aphids will be treated next monday?",
"type": ["email", "CROP"]
}' http://localhost:8000/
```

```
curl -X POST -H "Content-Type: application/json" -d '{
"text": "What are tomatoes and potaotes that are being attacked by aphids? "
}' http://localhost:8000/
```
