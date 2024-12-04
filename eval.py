import pandas as pd
import numpy as np
from bert_score import BERTScorer, score
# from moverscore import get_idf_dict, word_mover_score
from moverscore_v2 import get_idf_dict, word_mover_score
from word_mover_distance import model
from transformers import AutoModel, AutoTokenizer
from adapters import AutoAdapterModel
import nltk
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
from nltk.translate.nist_score import sentence_nist
from nltk.translate.meteor_score import single_meteor_score
from nltk.tokenize import word_tokenize, sent_tokenize
from rouge_score import rouge_scorer
from jiwer import wer
from collections import defaultdict
from sacrebleu import BLEU


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


def sentence_bleu_score(cands, refs):
    # refs = ["I", "am", "a", "bot"], cands = ["I", "am", "a", "chatbot"]
    return sentence_bleu([refs], cands)


def corpus_bleu_score(cands, refs):
    # refs = ["I am a bot.", "I like logic."], cands = ["I am a chatbot"]
    refs = [[ref] for ref in refs]
    return corpus_bleu(refs, cands)


def sacrebleu_score(cands, refs):
    # refs = ["I am a bot.", "I like logic."], cands = ["I am a chatbot"]
    bleu = BLEU()
    return bleu.corpus_score(cands, [refs]).score


def sacrebleu_sent_score(cands, refs):
    # refs = ["I am a bot. I like logic."], cands = "I am a chatbot"
    bleu = BLEU()
    return bleu.sentence_score(cands, [refs]).score


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

def word_mover_distance(cand_words, ref_words, vocab):
    # refs = "the cat was found under the bed", cands = "the cat was under the bed"

    # SPECTER model
    spec_tokenizer = AutoTokenizer.from_pretrained("allenai/specter2_base")
    spec_model = AutoAdapterModel.from_pretrained("allenai/specter2_base")
    spec_model.load_adapter("allenai/specter2", source="hf", load_as="proximity", set_active=True)

    emb_dict = generate_embeddings(vocab, spec_model, spec_tokenizer)
    my_model = model.WordEmbedding(model=emb_dict)
    specter_results = my_model.wmdistance(cand_words, ref_words)

    # SciBERT model
    scibert_tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
    scibert_model = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased")

    emb_dict = generate_embeddings(vocab, scibert_model, scibert_tokenizer)
    my_model = model.WordEmbedding(model=emb_dict)

    scibert_results = my_model.wmdistance(cand_words, ref_words)

    return specter_results, scibert_results


def main(file_path):    
    scorer = BERTScorer(lang="en")

    df = pd.read_excel(file_path)
    for index, row in df.iterrows():
        for ref_model in MODELS:
            ref_words = word_tokenize(row[f"{ref_model}_synthesis"].lower())
            ref_sents = sent_tokenize(row[f"{ref_model}_synthesis"].lower())
            ref_fullsent = row[f"{ref_model}_synthesis"].lower()

            for cand_model in MODELS:
                if ref_model == cand_model:
                    continue
                
                cand_words = word_tokenize(row[f"{cand_model}_synthesis"].lower())
                cand_sents = sent_tokenize(row[f"{cand_model}_synthesis"].lower())
                cand_fullsent = row[f"{cand_model}_synthesis"].lower()
                vocab = set(ref_words + cand_words)

                # bert-score
                value = bert_score(scorer, [cand_fullsent], [ref_fullsent])
                df.at[index, f"bertscoreP_{ref_model}_{cand_model}"] = value[0].item()
                df.at[index, f"bertscoreR_{ref_model}_{cand_model}"] = value[1].item()
                df.at[index, f"bertscoreF1_{ref_model}_{cand_model}"] = value[2].item()

                # mover-score
                value = mover_score([cand_fullsent], [ref_fullsent])
                df.at[index, f"moverscore_{ref_model}_{cand_model}"] = value

                # word-mover-distance
                specter_value, scibert_value = word_mover_distance(cand_words, ref_words, vocab)
                df.at[index, f"wmd_specter_{ref_model}_{cand_model}"] = specter_value
                df.at[index, f"wmd_scibert_{ref_model}_{cand_model}"] = scibert_value

                # corpus-bleu-score
                value = sacrebleu_score(cand_sents, ref_sents)
                df.at[index, f"sacrebleu_corpus_{ref_model}_{cand_model}"] = value

                # sentence-bleu-score
                value = sacrebleu_sent_score(cand_fullsent, ref_fullsent)
                df.at[index, f"sacrebleu_sent_{ref_model}_{cand_model}"] = value

                # rouge-score
                value = rouge_score(cand_fullsent, ref_fullsent)
                df.at[index, f"rouge1P_{ref_model}_{cand_model}"] = value["rouge1"].precision
                df.at[index, f"rouge1R_{ref_model}_{cand_model}"] = value["rouge1"].recall
                df.at[index, f"rouge1F1_{ref_model}_{cand_model}"] = value["rouge1"].fmeasure
                df.at[index, f"rouge2P_{ref_model}_{cand_model}"] = value["rouge2"].precision
                df.at[index, f"rouge2R_{ref_model}_{cand_model}"] = value["rouge2"].recall
                df.at[index, f"rouge2F1_{ref_model}_{cand_model}"] = value["rouge2"].fmeasure
                df.at[index, f"rougeLP_{ref_model}_{cand_model}"] = value["rougeL"].precision
                df.at[index, f"rougeLR_{ref_model}_{cand_model}"] = value["rougeL"].recall
                df.at[index, f"rougeLF1_{ref_model}_{cand_model}"] = value["rougeL"].fmeasure

                # nist-score
                value = nist_score(cand_words, ref_words)
                df.at[index, f"nist_{ref_model}_{cand_model}"] = value

                # meteor-score
                value = meteor_score(cand_words, ref_words)
                df.at[index, f"meteor_{ref_model}_{cand_model}"] = value

                # wer-score
                value = wer_score(cand_fullsent, ref_fullsent)
                df.at[index, f"wer_{ref_model}_{cand_model}"] = value              

        break
    
    df.to_excel(file_path, index=False)


def test():
    cands = ["Hello there general kenobi, the tree is withering. I loved that tree."] #, "Obama speaks to the media in Chicago"]
    refs = ["Hello there general kenobi, the tree is dying. I liked that tree. It smelled good."] #, "The president spoke to the press in Chicago"]
    cands_sents = sent_tokenize(cands[0])
    refs_sents = sent_tokenize(refs[0])
    # bert = bert_score(BERTScorer(lang="en"), cands, refs)
    # mover = mover_score(cands, refs)
    # wmd = word_mover_distance(refs[0].lower().split(), cands[0].lower().split(), set(refs[0].lower().split() + cands[0].lower().split()))
    # sent_bleu = sentence_bleu_score(cands[0].lower().split(), refs[0].lower().split())
    # corp_bleu = corpus_bleu_score(cands_sents, refs_sents)
    sacrebleu_corpus = sacrebleu_score(cands_sents, refs_sents)
    sacrebleu_sentC = sacrebleu_score(cands, refs)
    sacrebleu_sent = sacrebleu_sent_score(cands[0], refs[0])
    # rouge = rouge_score(cands[0], refs[0])
    # nist = nist_score(cands[0].lower().split(), refs[0].lower().split())
    # meteor = meteor_score(cands[0].lower().split(), refs[0].lower().split())
    # wer = wer_score(cands[0], refs[0])

    # print(bert)
    # print(mover)
    # print(wmd)
    # print(sent_bleu)
    # print(corp_bleu)
    print(sacrebleu_corpus)
    print(sacrebleu_sentC)
    print(sacrebleu_sent)
    # print(rouge)
    # print(nist)
    # print(meteor)
    # print(wer)


if __name__ == "__main__":
    # TODO: use cased or uncased models?
    # TODO: BLEU -> use entire answer as one reference string or split up into sentences?
    # TODO: WMD -> generate word embeddings even though they use contextualised embeddings?

    # main("data/BioASQ_dataset_5_sentences.xlsx")
    test()

    # nltk.download('wordnet')
    # nltk.download("punkt_tab")