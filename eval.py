import pandas as pd
import numpy as np
from bert_score import score
from moverscore import get_idf_dict, word_mover_score
from word_mover_distance import model
from transformers import AutoModel, AutoTokenizer
from adapters import AutoAdapterModel
import nltk
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.nist_score import sentence_nist
from nltk.translate.meteor_score import single_meteor_score
from rouge_score import rouge_scorer
from jiwer import wer


def bert_score(cands, refs):
    P, R, F1 = score(cands, refs, lang='en', verbose=True)
    return P, R, F1


def mover_score(cands, refs):
    idf_dict_hyp = get_idf_dict(cands)
    idf_dict_ref = get_idf_dict(refs)
    return word_mover_score(cands, refs, idf_dict_ref, idf_dict_hyp)


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


def main():
    cands = ["hello there general kenobi the tree is flying", "Obama speaks to the media in Chicago"]
    refs = ["Hello there generalo kenobi the tree is dying", "The president spoke to the press in Chicago"]

    # bert = bert_score(cands, refs)
    # mover = mover_score(cands, refs)
    # wmd = word_mover_distance(cands, refs)
    # bleu = bleu_score(cands[0].lower().split(), refs[0].lower().split())
    # rouge = rouge_score(cands[0], refs[0])
    # nist = nist_score(cands[0].lower().split(), refs[0].lower().split())
    # meteor = meteor_score(cands[0].lower().split(), refs[0].lower().split())
    wer = wer_score(cands[0], refs[0])

    # print(bert)
    # print(mover)
    # print(wmd)
    # print(bleu)
    # print(rouge)
    # print(nist)
    # print(meteor)
    print(wer)


if __name__ == "__main__":
    # TODO: use cased or uncased models?
    main()
    
    # nltk.download('wordnet')