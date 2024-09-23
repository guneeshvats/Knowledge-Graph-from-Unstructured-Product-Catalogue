# Knowledge-Graph-from-Unstructured-Product-Catalogue
In this project I am converting an unstructured text about products into a knowledge graph which you can query also


Task : To build a knowledge graph which has the info of the whole product catalouge availaible on the website and can convert an unstructured text/paragraph about a new product into a structured json format which dscribes the information mabout the characteristics of the product based on a specific schema. You can also inference the model about information of the prodcuts. 


![image](https://github.com/user-attachments/assets/19d32793-fce6-41c9-a57a-84fd1dac9f90)


## Steps to run the project 
0. Make sure you have the synthetic data generated using the `generate_synthetic_data.py` file using the dataset given in json format - `becn_data.json`
1. Using the structured data in filename I converted it into jsonl fomnat and that data is in `training_data_jsonl_format.jsonl`. Passing in the schema of the product catalogue (shcema is in `product_schema.ttl`) as part of the system prompt to prevent - hallucination or catastrophic forgetting.,
2. Then run the `fine_tune.py` code. 

## Folder Structure
```
|
|
```
