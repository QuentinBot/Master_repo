from openai import OpenAI
import re
import pandas as pd
import json
import time
from api_keys import ACADEMICCLOUD_KEY


MODELS = ["qwen2.5-72b-instruct", "mistral-large-instruct", "meta-llama-3.1-70b-instruct", "meta-llama-3.1-8b-instruct"]
# "mistral-large-instruct", "meta-llama-3.1-70b-instruct", "meta-llama-3.1-8b-instruct"

DEFAULT_DELAY = 5


def prepare_data():
    filepath = "data/gpt-4_paper-wise.xlsx"
    df = pd.read_excel(filepath)
    df = df.iloc[:, :19]
    df.rename(columns={"research_problem": "research_question"}, inplace=True)

    # map research problems to questions
    problem_to_question_map = json.load(open("data/mapped_questions.json", "r"))
    for index, row in df.iterrows():
        problem = re.sub(r'\s+', ' ', row["research_question"]).strip()
        df.at[index, "research_question"] = problem_to_question_map[problem]

    df.to_excel("data/synthesized_papers.xlsx", index=False)


def synthesis_from_papers(filepath="data/synthesized_papers.xlsx"):
    df = pd.read_excel(filepath)

    base_url = "https://chat-ai.academiccloud.de/v1"
    # models = ["meta-llama-3.1-8b-instruct", "mistral-large-instruct", "meta-llama-3.1-70b-instruct", "qwen2.5-72b-instruct"] 

    # Start OpenAI client
    client = OpenAI(
        api_key = ACADEMICCLOUD_KEY,
        base_url = base_url
    )

    SYSTEM_PROMPT = "You are a scholarly assistant tasked with answering research questions by synthesizing information from the abstracts and titles of scholarly articles."

    for index, row in df.iterrows():
        print("###############################################")
        print(f"Processing row {index}")

        question = row["research_question"]
        paper1_content = f"Title: {row['paper_1_title']}. Abstract: {row['paper_1_abstract']}"
        paper2_content = f"Title: {row['paper_2_title']}. Abstract: {row['paper_2_abstract']}"
        paper3_content = f"Title: {row['paper_3_title']}. Abstract: {row['paper_3_abstract']}"
        paper4_content = f"Title: {row['paper_4_title']}. Abstract: {row['paper_4_abstract']}"
        paper5_content = f"Title: {row['paper_5_title']}. Abstract: {row['paper_5_abstract']}"

        user_prompt = f"Generate a synthesis from the provided papers as content on the research question '{question}' into a concise single paragraph of no more than 200 words. Follow these instructions: \n - Only the titles and abstracts will be provided from up to five scientific papers which are to be used as the content for the synthesis. \n - The objective of this synthesis is to provide a paperwise analysis. Therefore, summarize each paper's contributions individually to the given research question above, noting significant findings or methodologies, or other such salient contribution facets. \n - Support each claim with citations, formatted as (1) or (3, 5) to refer to the respective papers' content, where the numbers correspond to the list of provided papers. \n - Ensure the output is formatted as a single cohesive paragraph without section headings, titles, abstracts, or any paper-like structure. The focus should be on integrating and synthesizing the content into a unified narrative. \n - Focus on essential information, maintaining clarity and precision. \n - Do not include additional information or exceed the specified word count of 200 words and the single paragraph synthesis output requirement.\n\n Papers \n 1. '{paper1_content}' \n\n 2. '{paper2_content}' \n\n 3. '{paper3_content}' \n\n 4. '{paper4_content}' \n\n 5. '{paper5_content}'"

        for model in MODELS:
            # if not pd.isnull(row[f"{model}_synthesis"]):
            #    continue

            print(f"Using model: {model}")

            retries = 0
            max_retries = 5
            delay = 60

            while retries < max_retries:
                try:
                    chat_completion = client.chat.completions.create(
                        messages=[{"role":"system","content": SYSTEM_PROMPT},{"role":"user","content": user_prompt}],
                        model= model,
                    )
                    df.at[index, f"{model}_synthesis"] = chat_completion.choices[0].message.content
                    break
                except Exception as e:
                    retries += 1
                    
                    print(f"Error: {e}. Retrying in {delay} seconds...")
                    time.sleep(delay)
                    delay *= 2
            
            
            time.sleep(DEFAULT_DELAY)
                
        df.to_excel(filepath, index=False)
        break
        

def chain_of_thought_test():
    client = OpenAI(
        api_key = ACADEMICCLOUD_KEY,
        base_url = "https://chat-ai.academiccloud.de/v1"
    )

    SYSTEM_PROMPT = "For each paper in the provided list, perform a structured analysis step by step as follows: \n1. Identify and briefly summarize the paper's main research question and objectives. \n2. Describe the primary methodology or approach used in the study, noting any unique aspects that define the research framework. \n3. Extract the central findings or insights, emphasizing those directly relevant to the user's question. \n4. Summarize the conclusion, discussing the broader significance or implications as highlighted by the authors. \nOnce you have analysed each paper, synthesise the information retrieved into a cohesive, concise answer to the user's research question. Limit the response to 200 words or fewer, and support each claim with citations, formatted as (1) or (3, 5) to refer to the respective papers in the list. Only output the synthesis limited to at most 200 words."      

    filepath = "data/synthesized_papers.xlsx"
    df = pd.read_excel(filepath)

    for index, row in df.iterrows():
        print("###############################################")
        print(f"Processing row {index}")

        question = row["research_question"]
        paper1_content = f"Title: {row['paper_1_title']}. Abstract: {row['paper_1_abstract']}"
        paper2_content = f"Title: {row['paper_2_title']}. Abstract: {row['paper_2_abstract']}"
        paper3_content = f"Title: {row['paper_3_title']}. Abstract: {row['paper_3_abstract']}"
        paper4_content = f"Title: {row['paper_4_title']}. Abstract: {row['paper_4_abstract']}"
        paper5_content = f"Title: {row['paper_5_title']}. Abstract: {row['paper_5_abstract']}"
        
        user_prompt = f"# Research Question: {question} \n# Papers: \n1. {paper1_content} \n2. {paper2_content} \n3. {paper3_content} \n4. {paper4_content} \n5. {paper5_content}"

        chat_completion = client.chat.completions.create(
            messages=[{"role":"system","content": SYSTEM_PROMPT},{"role":"user","content": user_prompt}],
            model= "meta-llama-3.1-8b-instruct",
        )
        print(chat_completion.choices[0].message.content)

        break  


def check_models():
    client = OpenAI(
        api_key = ACADEMICCLOUD_KEY,
        base_url = "https://chat-ai.academiccloud.de/v1"
    )

    models = client.models.list()
    for model in models:
        print(model)  


def check_specific_model():
    client = OpenAI(
        api_key = ACADEMICCLOUD_KEY,
        base_url = "https://chat-ai.academiccloud.de/v1"
    )
    model = "mistral-large-instruct"

    chat_completion = client.chat.completions.create(
        messages=[{"role":"system","content": "You are a helpful assistant."},{"role":"user","content": "Ping"}], model= model,
    )
    print(chat_completion.choices[0].message.content)


def check_df():
    df = pd.read_excel("data/synthesized_papers.xlsx")
    for index, row in df.iterrows():
        print(pd.isnull(row["mistral-large-instruct_synthesis"]))


def test():
    with open("data/doc_to_abstract.json", "r") as f:
        data = json.load(f)
        print(data["http://www.ncbi.nlm.nih.gov/pubmed/30885541"])


if __name__ == "__main__":
    # prepare_data()
    
    files = ["data/12B1_golden.xlsx", "data/12B2_golden.xlsx", "data/12B3_golden.xlsx", "data/12B4_golden.xlsx"]
    for file in files:
        synthesis_from_papers(file)
    
    # check_df()
    # check_models()
    # check_specific_model()
    # chain_of_thought_test()
    
    # test()
    # base_model_access() 