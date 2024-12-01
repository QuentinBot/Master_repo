import pandas as pd
import numpy as np
from bert_score import score
from moverscore import get_idf_dict, word_mover_score
from word_mover_distance import model
from transformers import AutoModel, AutoTokenizer
from adapters import AutoAdapterModel


def bert_score(cands, refs):
    P, R, F1 = score(cands, refs, lang='en', verbose=True)
    return P, R, F1


def mover_score(cands, refs):
    idf_dict_hyp = get_idf_dict(cands)
    idf_dict_ref = get_idf_dict(refs)
    return word_mover_score(cands, refs, idf_dict_ref, idf_dict_hyp)


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


def main():
    cands = ["hello there", "general kenobi", "Obama speaks to the media in Chicago"]
    refs = ["Hello there", "generalo kenobi", "The president spoke to the press in Chicago"]

    bert = bert_score(cands, refs)
    mover = mover_score(cands, refs)
    wmd = word_mover_distance(cands, refs)

    print(bert)
    print(mover)
    print(wmd)


if __name__ == "__main__":
    # TODO: use cased or uncased models?
    main()