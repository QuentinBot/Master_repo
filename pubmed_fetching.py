import pandas as pd
from metapub import PubMedFetcher
import time
import json


def pubmed_fetch():
    # Prompt the user to enter the path to the CSV file
    input_files = ["data/12B1_golden.json", "data/12B2_golden.json", "data/12B3_golden.json", "data/12B4_golden.json"]
    doc_to_abstract = {}
    doc_to_title = {}
    
    # Initialize the PubMedFetcher
    fetch = PubMedFetcher()

    # Define a function to fetch the abstract given a PMID
    def fetch_abstract(pmid):
        try:
            article = fetch.article_by_pmid(str(pmid))
            return (article.abstract, article.title) if article else None
        except Exception as e:
            print(f"Error fetching abstract for PMID {pmid}: {e}")
            return None

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

                response = fetch_abstract(pmid)
                if not response:
                    print(f"Error fetching abstract for doc {doc}")
                    continue
                doc_to_abstract[doc] = response[0]
                doc_to_title[doc] = response[1]
                
                
            time.sleep(0.5)
        

        json.dump(doc_to_abstract, open("data/doc_to_abstract.json", "w"))
        json.dump(doc_to_title, open("data/doc_to_title.json", "w"))


def convert_to_xlsx():
    # Convert the JSON file to an Excel file
    doc_to_abstract = json.load(open("data/doc_to_abstract.json", "r"))
    doc_to_title = json.load(open("data/doc_to_title.json", "r"))
    input_files = ["data/12B1_golden.json", "data/12B2_golden.json", "data/12B3_golden.json", "data/12B4_golden.json"]

    for file in input_files:

        df = pd.read_json(file)
        df.columns = df.columns.str.strip()  # Ensure no extra whitespace in headers
        new_df = pd.DataFrame(columns=["research_question", "paper_1_title", "paper_1_abstract", "paper_2_title", "paper_2_abstract", "paper_3_title", "paper_3_abstract", "paper_4_title", "paper_4_abstract", "paper_5_title", "paper_5_abstract"])

        for val in df["questions"]:
            row = [val["body"], "", "", "", "", "", "", "", "", "", ""]

            print(f"Processing {val['body']}")
            i = 0
            for doc in val["documents"]:
                i += 1
                if i == 6:
                    break

                row[(i*2)-1] = doc_to_title[doc]
                row[i*2] = doc_to_abstract[doc]


            new_df.loc[len(new_df)] = row
        
        new_df.to_excel(f"{file.split('.')[0]}.xlsx", index=False)
        


if __name__ == "__main__":
    # pubmed_fetch()
    convert_to_xlsx()