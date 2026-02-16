import json
from SentenceTokenizer import LitcoinSentenceTokenizer

TYPE_MAP = {
    "Disease": "DiseaseOrPhenotypicFeature",
    "Gene": "GeneOrGeneProduct",
    "Chemical": "ChemicalEntity",
}

def read_jsonl_to_list(file_path):
    data_list = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if line:
                data_list.append(json.loads(line))
    return data_list

def build_sentence_lookup(data):
    sentence_lookup = {}
    tokenizer = LitcoinSentenceTokenizer()
    for entry in data:
        abstract_id = entry['document_id']
        title_text = entry['title']
        abstract_text = entry['abstract']

        tokenizer = LitcoinSentenceTokenizer()

        if title_text: 
            tokenized_title = tokenizer.sentence_tokenize(title_text)
        else:
            tokenized_title = []
        
        if abstract_text:
            tokenized_abstract = tokenizer.sentence_tokenize(abstract_text)
        else:   
            tokenized_abstract = []

        tokenized_sentences = tokenized_title + tokenized_abstract
        local_lookup = {sent:i for i, sent in enumerate(tokenized_sentences)}

        sentence_lookup[abstract_id] = local_lookup
    
    return sentence_lookup

def main():
    id_to_pmid = {} 

    for path in ["./abstracts_train_pmids.jsonl", "./abstracts_test_pmids.jsonl"]:
        data = read_jsonl_to_list(path)
        for item in data:
            id = item['abstract_id']
            pmid = item['pmid']

            if id in id_to_pmid:
                assert id_to_pmid[id] == pmid, f"Conflict for id {id}: {id_to_pmid[id]} vs {pmid}"
            else:
                id_to_pmid[id] = item['pmid']

    with open("./Annotated_Tina_RD1.json") as f:
        data = json.load(f)

    with open("../litcoin_600/All.json") as f: 
        all_data = json.load(f)

    sentence_lookup = build_sentence_lookup(all_data)

    for item in data:
        id = str(item['abstract_id'])
        if id in id_to_pmid:
            pmid = id_to_pmid[id]
        else:
            pmid = None 

        item['pmid'] = pmid
        if pmid is not None and pmid in sentence_lookup:
            sentences_map = sentence_lookup[pmid]
            sentence_text = item['sentence']
            if sentence_text in sentences_map:
                sentence_id = sentences_map[sentence_text]
                item['sentence_id'] = sentence_id
            else:
                item['sentence_id'] = None
        else:
            item['sentence_id'] = None

        if item['entity_a_type'] in TYPE_MAP:
            item['entity_a_type'] = TYPE_MAP[item['entity_a_type']]
        if item['entity_b_type'] in TYPE_MAP:
            item['entity_b_type'] = TYPE_MAP[item['entity_b_type']]

    with open("./Annotated_Tina_RD1_with_pmids.json", "w") as f:
        json.dump(data, f, indent=2)


if __name__ == "__main__":
    main()