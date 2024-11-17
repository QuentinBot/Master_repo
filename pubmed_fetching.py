import pandas as pd
from metapub import PubMedFetcher
import time
import os
import json

# Prompt the user to enter the path to the CSV file
input_files = ["data/12B1_golden.json", "data/12B2_golden.json", "data/12B3_golden.json", "data/12B4_golden.json"]
doc_to_abstract = json.load(open("data/doc_to_abstract.json", "r"))
print(doc_to_abstract)
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

for file in input_files:

    # Load the CSV file into a DataFrame
    df = pd.read_json(file)
    df.columns = df.columns.str.strip()  # Ensure no extra whitespace in headers

    for val in df["questions"]:
        print(f"Processing {val['body']}")
        i = 0
        for doc in val["documents"]:
            i += 1
            if i == 6:
                break

            if doc in doc_to_abstract:
                continue

            pmid = doc.split("/")[-1]

            abstract = fetch_abstract(pmid)
            doc_to_abstract[doc] = abstract
            
            
        time.sleep(0.5)
    for key, val in doc_to_abstract.items():
        print(key, val)

json.dump(doc_to_abstract, open("data/doc_to_abstract.json", "w"))