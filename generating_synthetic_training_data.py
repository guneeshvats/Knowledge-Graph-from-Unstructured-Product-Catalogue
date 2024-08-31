import json
import chardet

# Function to load JSON data with correct encoding
def load_json_file(file_path):
    with open(file_path, 'rb') as f:
        encoding = chardet.detect(f.read())['encoding']
    with open(file_path, 'r', encoding=encoding) as f:
        return json.load(f)

input_data = load_json_file('C:/Users/Guneesh Vats/PycharmProjects/dataset_building/becn_data_very_small_updated.json')
print("Loaded input data")

# Define the schema to be included in the prompt
schema_definition = """
@prefix ex: <http://proxzar.com/schema/1/> .

schema:Product a rdf:Class .
schema:hasBrand a rdf:Property ; rdfs:domain schema:Product ; rdfs:range schema:Brand .
schema:hasCategory a rdf:Property ; rdfs:domain schema:Product ; rdfs:range schema:Category .
schema:hasDescription a rdf:Property ; rdfs:domain schema:Product ; rdfs:range xsd:string .
schema:hasApplicableStandards a rdf:Property ; rdfs:domain schema:Product ; rdfs:range xsd:string .
schema:hasArea a rdf:Property ; rdfs:domain schema:Product ; rdfs:range xsd:string .
schema:hasBaseMaterial a rdf:Property ; rdfs:domain schema:Product ; rdfs:range xsd:string .
schema:hasDimensions a rdf:Property ; rdfs:domain schema:Product ; rdfs:range schema:Dimensions .
schema:Dimensions a rdf:Class .
schema:hasHeight a rdf:Property ; rdfs:domain schema:Dimensions ; rdfs:range xsd:float .
schema:hasLength a rdf:Property ; rdfs:domain schema:Dimensions ; rdfs:range xsd:float .
schema:hasThickness a rdf:Property ; rdfs:domain schema:Dimensions ; rdfs:range xsd:float .
schema:hasWeight a rdf:Property ; rdfs:domain schema:Dimensions ; rdfs:range xsd:float .
schema:hasWidth a rdf:Property ; rdfs:domain schema:Dimensions ; rdfs:range xsd:float .
schema:MainCategory a rdf:Class .
schema:SubCategory a rdf:Class .

schema:Brand a rdf:Class .
schema:Category a rdf:Class ..
"""

# Generate training examples
examples = []

for product in input_data["data"]:
    system_prompt = f"""
    You are a helpful assistant tasked with converting unstructured text into RDF format using the predefined product schema. The schema specifies the structure and relationships between different entities in the RDF graph. Ensure that the RDF graph you generate adheres to the Schema1 ontology with IRI <http://proxzar.com/schema/1/>:
    
    {schema_definition}

    Below is the unstructured text that needs to be converted to an RDF graph:
    """
    user_text = product["GeneratedDescription"]
    assistant_response = product["RDF"]

    example = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": assistant_response}
        ]
    }
    examples.append(example)

# Save the examples to a JSONL file
with open('training_data.jsonl', 'w') as outfile:
    for example in examples:
        json.dump(example, outfile)
        outfile.write('\n')

print(f"Generated {len(examples)} examples for training.")
