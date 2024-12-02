import pandas as pd
import numpy as np
from bert_score import BERTScorer, score
# from moverscore import get_idf_dict, word_mover_score
from moverscore_v2 import get_idf_dict, word_mover_score
from word_mover_distance import model
from transformers import AutoModel, AutoTokenizer
from adapters import AutoAdapterModel
import nltk
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.nist_score import sentence_nist
from nltk.translate.meteor_score import single_meteor_score
from rouge_score import rouge_scorer
from jiwer import wer
from collections import defaultdict


MODELS = ["qwen2.5-72b-instruct", "mistral-large-instruct", "meta-llama-3.1-70b-instruct", "meta-llama-3.1-8b-instruct"]


def bert_score(scorer, cands, refs):
    # refs = ["the cat was found under the bed"], cands = ["the cat was under the bed"]
    # uses Roberta-large
    P, R, F1 = scorer.score(cands, refs)
    return P, R, F1


def mover_score(cands, refs):
    # refs = ["the cat was found under the bed"], cands = ["the cat was under the bed"]
    # uses DistilBert
    idf_dict_cand = defaultdict(lambda: 1.)
    idf_dict_ref = defaultdict(lambda: 1.)
    return word_mover_score(refs, cands, idf_dict_ref, idf_dict_cand, n_gram=2)


def bleu_score(cands, refs):
    # refs = ["I", "am", "a", "bot"], cands = ["I", "am", "a", "chatbot"]
    return sentence_bleu([refs], cands)


def rouge_score(cands, refs):
    # refs = "the cat was found under the bed", cands = "the cat was under the bed"
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(cands, refs)
    return scores


def nist_score(cands, refs):
    # refs = ["I", "am", "a", "bot"], cands = ["I", "am", "a", "chatbot"]
    return sentence_nist([refs], cands)


def meteor_score(cands, refs):
    # refs = ["I", "am", "a", "bot"], cands = ["I", "am", "a", "chatbot"]
    return single_meteor_score(refs, cands)


def wer_score(cands, refs):
    # refs = "the cat was found under the bed", cands = "the cat was under the bed"
    return wer(refs, cands)


def generate_embeddings(word_list, model, tokenizer):
    emb_dict = {}

    for word in word_list:
        inputs = tokenizer(word, padding=True, truncation=True, return_tensors="pt", return_token_type_ids=False, max_length=512)
        output = model(**inputs)
        embedding = output.last_hidden_state[:, 0, :].detach().numpy()
        emb_dict[word] = embedding

    return emb_dict

def word_mover_distance(cands, refs):
    text_batch = []
    for text in cands + refs:
        for word in text.split():
            text_batch.append(word.lower())

    # SPECTER model
    spec_tokenizer = AutoTokenizer.from_pretrained("allenai/specter2_base")
    spec_model = AutoAdapterModel.from_pretrained("allenai/specter2_base")
    spec_model.load_adapter("allenai/specter2", source="hf", load_as="proximity", set_active=True)

    emb_dict = generate_embeddings(text_batch, spec_model, spec_tokenizer)
    my_model = model.WordEmbedding(model=emb_dict)

    specter_results = my_model.wmdistance(cands[-1].lower().split(), refs[-1].lower().split())

    # SciBERT model
    scibert_tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
    scibert_model = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased")

    emb_dict = generate_embeddings(text_batch, scibert_model, scibert_tokenizer)
    my_model = model.WordEmbedding(model=emb_dict)

    scibert_results = my_model.wmdistance(cands[-1].lower().split(), refs[-1].lower().split())

    return specter_results, scibert_results


def main(file_path):
    cands = ["hello there general kenobi the tree is flying", "Obama speaks to the media in Chicago"]
    refs = ["Hello there generalo kenobi the tree is dying", "The president spoke to the press in Chicago"]
    
    scorer = BERTScorer(lang="en")

    df = pd.read_excel(file_path)
    for index, row in df.iterrows():
        for ref_model in MODELS:
            for cand_model in MODELS:
                if ref_model == cand_model:
                    continue
                
                # bert-score
                value = bert_score(scorer, [row[f"{cand_model}_synthesis"]], [row[f"{ref_model}_synthesis"]])
                df.at[index, f"bertscoreP_{ref_model}_{cand_model}"] = value[0].item()
                df.at[index, f"bertscoreR_{ref_model}_{cand_model}"] = value[1].item()
                df.at[index, f"bertscoreF1_{ref_model}_{cand_model}"] = value[2].item()

                # mover-score
                value = mover_score([row[f"{cand_model}_synthesis"]], [row[f"{ref_model}_synthesis"]])
                df.at[index, f"moverscore_{ref_model}_{cand_model}"] = value


    
    df.to_excel(file_path, index=False)



    # bert = bert_score(cands, refs)
    # mover = mover_score(cands, refs)
    # wmd = word_mover_distance(cands, refs)
    # bleu = bleu_score(cands[0].lower().split(), refs[0].lower().split())
    # rouge = rouge_score(cands[0], refs[0])
    # nist = nist_score(cands[0].lower().split(), refs[0].lower().split())
    # meteor = meteor_score(cands[0].lower().split(), refs[0].lower().split())
    # wer = wer_score(cands[0], refs[0])

    # print(bert)
    # print(mover)
    # print(wmd)
    # print(bleu)
    # print(rouge)
    # print(nist)
    # print(meteor)
    # print(wer)


if __name__ == "__main__":
    # TODO: use cased or uncased models?
    main("data/BioASQ_dataset_5_sentences.xlsx")
    
    # nltk.download('wordnet')