import pandas as pd
from nltk import sent_tokenize, word_tokenize


SPORTS_SENTENCE = "Stephen Curry, LeBron James and Kevin Durant led the Americans to a 98-87 victory against host country France to win gold at the 2024 Paris Olympics on Saturday."
BLOG_SENTENCE = "If you are a Turing fan or follower, you know where this conversation is headed. If you are not, let us save you the trouble."
MODELS = ["qwen2.5-72b-instruct", "mistral-large-instruct", "meta-llama-3.1-70b-instruct", "meta-llama-3.1-8b-instruct"]
CONJUNCTIONS = {"and", "or", "but", "so", "yet", "for", "nor"} # TODO: improve conjunctions list


def adversarial_creation_subtle(filepath):
    # create adversarial examples by making subtle changes
    
    df = pd.read_excel(filepath)
    for index, row in df.iterrows():
        
        for model in MODELS:
            same_domain_sentence = find_similar_entry(index, df, model)
            sentences = sent_tokenize(row[f"{model}_synthesis"])

            # 1. Relevancy
            df.at[index, f"{model}_subtle_relevancy"] = row[f"{model}_synthesis"] + " " + same_domain_sentence
            
            # 2. Correctness
            df.at[index, f"{model}_subtle_correctness"] = row[f"{model}_synthesis"] + " " + same_domain_sentence

            # 3. Completeness
            if len(sentences) > 1:
                df.at[index, f"{model}_subtle_completeness"] = " ".join(sentences[:-1])
            else:
                df.at[index, f"{model}_subtle_completeness"] = row[f"{model}_synthesis"]

            # 4. Informativeness
            df.at[index, f"{model}_subtle_informativeness"] = row[f"{model}_synthesis"] + " " + same_domain_sentence

            # 5. Integration
            new_synth = []
            found = False
            for sent in sentences:
                words = word_tokenize(sent)
                for word in words:
                    if found == False and word in CONJUNCTIONS:
                        found = True
                    else:
                        new_synth.append(word)
                new_synth[-1] += "."
            df.at[index, f"{model}_subtle_integration"] = " ".join(new_synth)
                
            # 6. Cohesion
            if len(sentences) > 1:            
                df.at[index, f"{model}_subtle_cohesion"] = " ".join(sentences[:-2] + [sentences[-1]] + [sentences[-2]])
            else:
                df.at[index, f"{model}_subtle_cohesion"] = row[f"{model}_synthesis"]

            # 7. Coherence
            df.at[index, f"{model}_subtle_coherence"] = row[f"{model}_synthesis"] + " " + same_domain_sentence

            # 8. Readability


            # 9. Conciseness


def adversarial_creation_extreme(filepath):
    # create adversarial examples by making extreme changes
    pass


def find_similar_entry():
    # find next entry of same category and return sentence
    pass


def main():
    original_bioasq = "data/bioasq_dataset_synthesis.xlsx"
    original_llm4syn = "data/llm4syn_dataset_synthesis.xlsx"
    print("Hello World!")


if __name__ == "__main__":
    main()