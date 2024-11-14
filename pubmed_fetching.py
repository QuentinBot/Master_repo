import pandas as pd
from metapub import PubMedFetcher
import time
import os
import json

# Prompt the user to enter the path to the CSV file
input_file = "data/12B1_golden.json"
output_file = input_file.replace(".json", "_with_abstracts.csv")
log_file = input_file.replace(".json", "_progress.log")

# Load the CSV file into a DataFrame
df = pd.read_json(input_file)
df.columns = df.columns.str.strip()  # Ensure no extra whitespace in headers

# Initialize the PubMedFetcher
fetch = PubMedFetcher()

# Define a function to fetch the abstract given a PMID
def fetch_abstract(pmid):
    try:
        article = fetch.article_by_pmid(str(pmid))
        return article.abstract if article else "Abstract not found"
    except Exception as e:
        print(f"Error fetching abstract for PMID {pmid}: {e}")
        return "Error fetching abstract"


doc_to_abstract = {}
for val in df["questions"]:
    print(f"Processing {val['body']}")
    i = 0
    for doc in val["documents"]:
        pmid = doc.split("/")[-1]

        abstract = fetch_abstract(pmid)
        doc_to_abstract[doc] = abstract
        
        i += 1
        if i == 5:
            break
    
    time.sleep(0.5)
for key, val in doc_to_abstract.items():
    print(key, val)

json.dump(doc_to_abstract, open("data/doc_to_abstract.json", "w"))