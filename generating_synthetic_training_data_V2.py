
#  Token - yW0aNp__FIGA60M1Lr3AYK9C1zb96PQ1HIheDqLq5920f_2ZuQHkg_xCPUk-7rCywErsk38W3Anh5I9A0x0lu2lM8YktKAPgfeB0jJ-lteCq7fiwOz21BQfZdeKsnWmRnvkr5tpdXZ-0cM-C2qW_Vo4S5BJaC2j2vQ3m2r9qtQDz2A7HixiYzK-uJm8CAIbhUcAZmktrNVeuODXyYYeu93cmMNrJZTtlQEHGH2ctCcPGgSMidUQvAfocy6Pcw748x0pqrLG9GrPcVi9NmfnVQYgBxH3FZkJRJAIaF3MhmFsyfNCULU5ark-dahqE-44wZHR5TJMyb0CD84zBn0Cx9VlRP8fg6iLkuw_VtumJzyud2NJPwwoA-aNA6z5OFjbX34Kb8aTRIk-mb3Af-V0dN-T4AOOu_iHfyRHvk2Ll0PvhwFwhLqexRIsF7xXJqpjGv8AJhqZFPN8k9QulRkIgx8-FTHmtBR_RDfupkP_slQ6zFL8RPfcT7L3NF9JGui1WwkM9t1Tmi7NnNeq5A9fgJg
# Current location : C:/Users/Guneesh Vats/PycharmProjects/dataset_building/becn_data_very_small_updated.json
import json
import chardet
import requests
from tqdm import tqdm
from rdflib import Graph, URIRef, Literal, Namespace
from rdflib.namespace import RDF

# Constants
TOKEN = 'yW0aNp__FIGA60M1Lr3AYK9C1zb96PQ1HIheDqLq5920f_2ZuQHkg_xCPUk-7rCywErsk38W3Anh5I9A0x0lu2lM8YktKAPgfeB0jJ-lteCq7fiwOz21BQfZdeKsnWmRnvkr5tpdXZ-0cM-C2qW_Vo4S5BJaC2j2vQ3m2r9qtQDz2A7HixiYzK-uJm8CAIbhUcAZmktrNVeuODXyYYeu93cmMNrJZTtlQEHGH2ctCcPGgSMidUQvAfocy6Pcw748x0pqrLG9GrPcVi9NmfnVQYgBxH3FZkJRJAIaF3MhmFsyfNCULU5ark-dahqE-44wZHR5TJMyb0CD84zBn0Cx9VlRP8fg6iLkuw_VtumJzyud2NJPwwoA-aNA6z5OFjbX34Kb8aTRIk-mb3Af-V0dN-T4AOOu_iHfyRHvk2Ll0PvhwFwhLqexRIsF7xXJqpjGv8AJhqZFPN8k9QulRkIgx8-FTHmtBR_RDfupkP_slQ6zFL8RPfcT7L3NF9JGui1WwkM9t1Tmi7NnNeq5A9fgJg'
PROXZAR_ID = 'BECNPRODS4'

# Loading JSON data
def load_json_file(file_path):
    with open(file_path, 'rb') as f:
        encoding = chardet.detect(f.read())['encoding']
    with open(file_path, 'r', encoding=encoding) as f:
        return json.load(f)

input_data = load_json_file('C:/Users/Guneesh Vats/PycharmProjects/dataset_building/becn_data_very_small.json')

# Generate Unstructured text using API
def generate_description(product_info):
    input_text = (
        f"Product: {product_info.get('ProductTitle', 'N/A')} "
        f"Brand: {product_info.get('Brand', 'N/A')} "
        f"Category: {product_info.get('MainCategory', 'N/A')} - {product_info.get('SubCategory', 'N/A')} "
        f"Description: {product_info.get('ProductDescription', 'N/A')}"
    )
    prompt = {
        "inputs": f"Please provide a detailed and vivid description of the following product information. Highlight the product's features, brand, category, and any other unique attributes. Ensure the language is engaging and reads as if a knowledgeable salesperson is describing the product to a potential customer. {input_text}",
        "max_new_tokens": 1200
    }

    try:
        response = requests.post("http://164.52.195.209:8610/generate", json=prompt)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"Request failed: {e}")
        return ""

    try:
        result = response.json()
    except json.JSONDecodeError:
        print(f"Failed to decode JSON: {response.text}")
        return ""
    # removing the ending truncated sentences from the generated descriptions
    description = result.get('generated_text', '')
    if description and description[-1] != '.':
        description = description[:description.rfind('.')] + '.'

    return description

# Clean the text using multiple API endpoints
def clean_text(text):
    # Helper function to split text into chunks of 50 words or less
    def split_into_chunks(text, chunk_size=50):
        words = text.split()
        return [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

    cleaned_chunks = []
    chunks = split_into_chunks(text)

    for chunk in chunks:
        # Remove dimension symbols
        url_uom = "https://pf.proxzar.ai/ekgwebforms/api/Proxzar/Members/PreProcessUoM?"
        res = requests.get(url_uom, params={'ProxzarId': PROXZAR_ID, 'SequenceUoM': chunk}, headers={'Authorization': f'Bearer {TOKEN}'})
        cleaned_chunk = res.content.decode('utf-8')

        # Remove stopwords and duplicates
        url_seq_pre = "https://pf.proxzar.ai/ekgwebforms/api/Proxzar/Members/PreprocessTextSequence"
        res = requests.get(url_seq_pre, params={'ProxzarID': PROXZAR_ID, 'Sequence': cleaned_chunk}, headers={'Authorization': f'Bearer {TOKEN}'})
        cleaned_chunk = res.content.decode('utf-8')

        # Spell check
        url_spell_check = "https://pf.proxzar.ai/ekgwebforms/api/Proxzar/Members/SpellCheckSequence"
        res = requests.get(url_spell_check, params={'ProxzarID': PROXZAR_ID, 'Sequence': cleaned_chunk}, headers={'Authorization': f'Bearer {TOKEN}'})
        cleaned_chunk = res.content.decode('utf-8')

        cleaned_chunks.append(cleaned_chunk)

    # Recombine cleaned chunks into a single text
    return ' '.join(cleaned_chunks)

# Extract entities using custom NER model
def extract_entities(text):
    url_ner = "http://216.48.189.230:8535/api/v1/getMatchingNEsForQ"
    json_data = {
        "proxId": PROXZAR_ID,
        "userQ": text
    }
    try:
        response = requests.post(url_ner, json=json_data)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"Error occurred: {e}")
        return {}

# Generate RDF format
EX = Namespace("http://proxzar.com/productcatalog#")

def generate_rdf(product_info, cleaned_description, entities):
    g = Graph()
    product_uri = URIRef(f"http://proxzar.com/productcatalog#{product_info['ProductID']}")
    g.add((product_uri, RDF.type, EX.Product))
    g.add((product_uri, EX.title, Literal(product_info['ProductTitle'])))
    g.add((product_uri, EX.brand, Literal(product_info['Brand'])))
    g.add((product_uri, EX.category, Literal(product_info['MainCategory'])))
    g.add((product_uri, EX.subCategory, Literal(product_info['SubCategory'])))
    g.add((product_uri, EX.description, Literal(cleaned_description)))

    # Add extracted entities to RDF
    for entity in entities.get('entities', []):
        label = entity.get('label', 'unknown')
        value = entity.get('text', '')
        g.add((product_uri, EX[label], Literal(value)))

    return g.serialize(format='turtle')

# Processing each product in the JSON file
for product in tqdm(input_data["data"], desc="Processing Products"):
    description = generate_description(product)
    cleaned_description = clean_text(description)
    entities = extract_entities(cleaned_description)
    product['GeneratedDescription'] = description  # Adding raw generated description to JSON
    product['RDF'] = generate_rdf(product, cleaned_description, entities)

# Save the updated data to a new JSON file
output_file = 'C:/Users/Guneesh Vats/PycharmProjects/dataset_building/becn_data_very_small_updated.json'
with open(output_file, 'w', encoding='utf-8') as outfile:
    json.dump(input_data, outfile, indent=4, ensure_ascii=False)
# ensure ascii = False   shows the encoding string in readable symbols

print(f"Updated product data with generated descriptions, cleaned text, extracted entities, and RDF saved to {output_file}")
