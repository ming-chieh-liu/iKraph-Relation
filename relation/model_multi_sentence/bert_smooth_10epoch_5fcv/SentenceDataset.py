import torch
from torch.utils.data import Dataset
import pandas as pd

class SentenceDataset(Dataset):
    def __init__(self, list_of_dataframes, tokenizer, config):
        """
        config: a dictionary of
            max_len: int, tokenizer maximium length
            is_train: Bool, if it's for training. When training, remove duplicated cases
            transform_method: str in 
                ["entity mask", "entity marker", "entity marker punkt", "typed entity marker", "typed entity marker punct"]
                how sentence is processed
            move_entities_to_start: if True, additionally add entities to start when processing
            model_type: str in ["cls", "triplet"]
                cls: only use the cls token
                triplet: use not only the cls token, but also the positions of entities
            label_column_name: str, the column name in the dataframe that contains the labels
            no_relation_file: str, optional csv file that each row indicates a pair doesn't have relations
            remove_cellline_organismtaxon: bool, remove all cellline and organismtaxon entities
        """
        label_list = ["NOT", "Association", "Positive_Correlation", "Negative_Correlation", "Bind", "Cotreatment", "Comparison", "Drug_Interaction", "Conversion"]
        self.LABEL_DICT = {idx: val for idx, val in enumerate(label_list)}
        self.ENTITY_LIST = ["CellLine", "ChemicalEntity", "DiseaseOrPhenotypicFeature", "GeneOrGeneProduct", "OrganismTaxon", "SequenceVariant"]

        og_dataframe = pd.concat(list_of_dataframes, ignore_index=True)
        self.tokenizer = tokenizer
        self.config = config
        self.dataframe = self._process_dataframe(og_dataframe)
        self.no_relation_mat = self._process_no_rel()
        special_tokens = self._get_special_tokens()
        self.tokenizer.add_tokens(special_tokens, special_tokens=True)

    def get_processed_dataframe(self):
        return self.dataframe
    
    def get_f1_true_labels(self):
        return list(range(1, len(self.LABEL_DICT)))
        
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, index):
        if self.config["model_type"] == "cls":
            return self._get_item_cls(index)
        elif self.config["model_type"] == "triplet":
            return self._get_item_triplet(index)
        else:
            raise NotImplementedError(f"{self.config['model_type']} is no supported!")

    def _process_dataframe(self, dataframe):
        dataframe = dataframe.copy(deep=True)
        if self.config["is_train"] == True:
            # dataframe = dataframe[dataframe["duplicated_flag"]==False]
            dataframe = dataframe.reset_index(drop=True)
            if self.config["remove_cellline_organismtaxon"] is True:
                remove_list = ["CellLine", "OrganismTaxon"]
                keep_flag = [True for _ in range(len(dataframe))]
                for idx, (_, entry) in enumerate(dataframe.iterrows()):
                    ent1, ent2 = entry["entity_a"][0][2], entry["entity_b"][0][2]
                    if ent1 in remove_list or ent2 in remove_list:
                        keep_flag[idx] = False
                dataframe = dataframe[keep_flag]
                dataframe = dataframe.reset_index(drop=True)
        
        dataframe["label"] = dataframe[self.config["label_column_name"]]
        dataframe["label"] = dataframe["label"].replace("TBD", 0)
        for key, label in self.LABEL_DICT.items():
            dataframe['label'] = dataframe['label'].replace(label, key)

        texts, new_ent1s, new_ent2s = [], [], []
        for _, entry in dataframe.iterrows():
            new_text, new_ent1, new_ent2 = self._transform_sentence(entry)
            texts.append(new_text)
            new_ent1s.append(new_ent1)
            new_ent2s.append(new_ent2)
        dataframe["processed_text"] = texts
        dataframe["processed_ent1"] = new_ent1s
        dataframe["processed_ent2"] = new_ent2s
        return dataframe
    
    def _transform_sentence(self, entry):
        transform_method = self.config["transform_method"]

        sent = entry["text"]
        entity_a_mentions = entry["entity_a"]
        entity_b_mentions = entry["entity_b"]

        # Collect all mentions with their entity group (A or B) and index
        all_mentions = []
        for idx, mention in enumerate(entity_a_mentions):
            all_mentions.append(('A', idx, mention[0], mention[1], mention[2]))
        for idx, mention in enumerate(entity_b_mentions):
            all_mentions.append(('B', idx, mention[0], mention[1], mention[2]))

        # Sort by position in reverse order (right to left) to avoid offset issues
        all_mentions.sort(key=lambda x: x[2], reverse=True)

        # Create unique temporary markers for each mention
        tmp_markers = {}
        for entity_group, idx, start, end, ent_type in all_mentions:
            marker_key = f"{entity_group}_{idx}"
            tmp_marker = f"<##TMP_{marker_key}$$>"
            assert tmp_marker not in sent
            tmp_markers[marker_key] = tmp_marker

        # Replace each mention with markers (right to left)
        for entity_group, idx, start, end, ent_type in all_mentions:
            mention_text = sent[start:end]
            marker_key = f"{entity_group}_{idx}"
            tmp_marker = tmp_markers[marker_key]

            # Determine pre/post markers and replacement text based on transform method
            if transform_method == "entity_mask":
                pre = ''
                replacement = ent_type
                post = ''
            elif transform_method == "entity_marker":
                pre = f'[E{1 if entity_group == "A" else 2}]'
                replacement = mention_text
                post = f'[/E{1 if entity_group == "A" else 2}]'
            elif transform_method == "entity_marker_punkt":
                pre = '@' if entity_group == "A" else '#'
                replacement = mention_text
                post = '@' if entity_group == "A" else '#'
            elif transform_method == "typed_entity_marker":
                pre = f'[{ent_type}]'
                replacement = mention_text
                post = f'[/{ent_type}]'
            elif transform_method == "typed_entity_marker_punct":
                pre = f'@ * {ent_type} *' if entity_group == "A" else f'# ^ {ent_type} ^'
                replacement = mention_text
                post = '@' if entity_group == "A" else '#'
            else:
                raise NotImplementedError(f"{transform_method} is not implemented!")

            # Insert markers into sentence
            sent = sent[0:start] + ' ' + pre + ' ' + tmp_marker + replacement + tmp_marker + ' ' + post + ' ' + sent[end:]

        # Add first mentions to start if configured
        if self.config["move_entities_to_start"] == True:
            first_ent_a = entry["text"][entity_a_mentions[0][0]:entity_a_mentions[0][1]]
            first_ent_b = entry["text"][entity_b_mentions[0][0]:entity_b_mentions[0][1]]
            sent = first_ent_a + ', ' + first_ent_b + ', ' + sent

        sent = ' '.join(sent.split())  # remove multiple spaces in a row
        sent = sent.replace("LPA1 '3", "LPA1    '3")  # only one special case where it's multiple spaces in the test dataste
        sent = sent.replace("ERK1 '2", "ERK1    '2")

        # First, find all marker positions before removing any
        all_marker_positions = []
        for idx in range(len(entity_a_mentions)):
            marker_key = f"A_{idx}"
            tmp_marker = tmp_markers[marker_key]
            ent_type = entity_a_mentions[idx][2]
            start_pos = sent.find(tmp_marker)
            end_pos = sent.find(tmp_marker, start_pos + len(tmp_marker))
            all_marker_positions.append(('A', idx, start_pos, end_pos, len(tmp_marker), ent_type))

        for idx in range(len(entity_b_mentions)):
            marker_key = f"B_{idx}"
            tmp_marker = tmp_markers[marker_key]
            ent_type = entity_b_mentions[idx][2]
            start_pos = sent.find(tmp_marker)
            end_pos = sent.find(tmp_marker, start_pos + len(tmp_marker))
            all_marker_positions.append(('B', idx, start_pos, end_pos, len(tmp_marker), ent_type))

        # Sort by position (right to left) so we can remove without affecting earlier positions
        all_marker_positions.sort(key=lambda x: x[2], reverse=True)

        # Remove markers from right to left and track final positions
        entity_a_final = {}
        entity_b_final = {}
        cumulative_shift = 0  # Track how much text we've removed

        for entity_group, idx, start_pos, end_pos, marker_len, ent_type in all_marker_positions:
            # Remove second marker first (since we're going right to left within each mention)
            sent = sent[:end_pos] + sent[end_pos+marker_len:]
            # Remove first marker
            sent = sent[:start_pos] + sent[start_pos+marker_len:]

            # Calculate final positions accounting for already-removed markers to the right
            final_start = start_pos - cumulative_shift
            final_end = end_pos - marker_len - cumulative_shift

            # Update cumulative shift (we removed 2 markers)
            cumulative_shift += 2 * marker_len

            if entity_group == 'A':
                entity_a_final[idx] = [final_start, final_end, ent_type]
            else:
                entity_b_final[idx] = [final_start, final_end, ent_type]

        # Convert back to lists in original order
        new_entity_a_positions = [entity_a_final[i] for i in range(len(entity_a_mentions))]
        new_entity_b_positions = [entity_b_final[i] for i in range(len(entity_b_mentions))]

        return sent, new_entity_a_positions, new_entity_b_positions

    def _get_special_tokens(self):
        special_tokens = []
        transform_method = self.config["transform_method"]

        if transform_method == "entity_mask":
            special_tokens += self.ENTITY_LIST
        elif transform_method == "entity_marker":
            special_tokens += ['[E1]', '/[E1]', '[E2]', '[/E2]']
        elif transform_method == "entity_marker_punkt":
            pass  # not adding @ or #
        elif transform_method == "typed_entity_marker":
            special_tokens += [f'[{this_type}]' for this_type in self.ENTITY_LIST]
            special_tokens += [f'[/{this_type}]' for this_type in self.ENTITY_LIST]
        elif transform_method == "typed_entity_marker_punct":
            special_tokens += self.ENTITY_LIST
        else:
            raise NotImplementedError(f"{transform_method} is not implemented!")
        if hasattr(self.tokenizer, "do_lower_case") and self.tokenizer.do_lower_case:
            return sorted([elem.lower() for elem in set(special_tokens)])
        else:
            return []

    def _tokenize_fn(self, sentence, add_special_tokens=True, padding=True):
        sentence = sentence.strip()
        return self.tokenizer(
            sentence,
            add_special_tokens=add_special_tokens,
            padding='max_length' if padding else False,
            truncation=True,
            max_length=self.config["max_len"],
            return_token_type_ids=False,
            return_attention_mask=True,
            return_tensors='pt',
        )

    def _get_item_cls(self, index):
        sentence = self.dataframe.loc[index, "processed_text"]
        label = self.dataframe.loc[index, "label"]
        ent_a_type = self.dataframe.loc[index, "entity_a"][0][2]
        ent_a_idx = self.ENTITY_LIST.index(ent_a_type)
        ent_b_type = self.dataframe.loc[index, "entity_b"][0][2]
        ent_b_idx = self.ENTITY_LIST.index(ent_b_type)
        encoding = self._tokenize_fn(sentence)

        return {
            'sentence': sentence,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'relation_mask': self.no_relation_mat[ent_a_idx, ent_b_idx, :],
            'label': torch.tensor(label, dtype=torch.long)
        }

    def _get_item_triplet(self, index):
        sentence = self.dataframe.loc[index, "processed_text"]
        ent1_positions = self.dataframe.loc[index, "processed_ent1"]
        ent2_positions = self.dataframe.loc[index, "processed_ent2"]
        # Use first mention of each entity for triplet model
        ent1_start, ent1_end, ent1_type = ent1_positions[0]
        ent2_start, ent2_end, ent2_type = ent2_positions[0]
        ent_a_idx = self.ENTITY_LIST.index(ent1_type)
        ent_b_idx = self.ENTITY_LIST.index(ent2_type)

        label = self.dataframe.loc[index, "label"]

        encoding_1 = self._tokenize_fn(sentence[0: ent1_start], padding=False, add_special_tokens=False)
        encoding_2 = self._tokenize_fn(sentence[0: ent2_start], padding=False, add_special_tokens=False)
        encoding = self._tokenize_fn(sentence)

        input_ids_1, attention_mask_1 = encoding_1['input_ids'].flatten(), encoding_1['attention_mask'].flatten()
        input_ids_2, attention_mask_2 = encoding_2['input_ids'].flatten(), encoding_2['attention_mask'].flatten()

        if len(input_ids_2) >= self.config["max_len"]:
            raise ValueError(f"Tokenization length > {self.config['max_len']} before the appearance of entity2, cannot truncate because entity2 would be discarded!")

        return {
            'sentence': sentence,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'relation_mask': self.no_relation_mat[ent_a_idx, ent_b_idx, :],
            'positions': torch.tensor([0, len(input_ids_1)+1, len(input_ids_2)+1]),
            'label': torch.tensor(label, dtype=torch.long)
        }
    def  _process_no_rel(self):
        ret = torch.zeros((len(self.ENTITY_LIST), len(self.ENTITY_LIST), len(self.LABEL_DICT)))
        if self.config["no_relation_file"] == "":
            return ret
        no_rel_file = pd.read_csv(self.config["no_relation_file"])
        for _, entry in no_rel_file.iterrows():
            type_a, type_b, relation = entry["type_a"], entry["type_b"], entry["relation"]
            idx_a = self.ENTITY_LIST.index(type_a)
            idx_b = self.ENTITY_LIST.index(type_b)
            idx_rel = list(self.LABEL_DICT.values()).index(relation)
            ret[idx_a, idx_b, idx_rel] = -999
            ret[idx_b, idx_a, idx_rel] = -999
        return ret

if __name__ == "__main__":
    from transformers import BertTokenizer, RobertaTokenizer
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    # for transform_method in ["entity_mask", "entity_marker", "entity_marker_punkt", "typed_entity_marker", "typed_entity_marker_punct"]:
    for transform_method in ["typed_entity_marker"]:
        config = {
            "model_type": "triplet",
            "max_len": 384,
            "transform_method": transform_method,
            "label_column_name": "annotated_type",
            "move_entities_to_start": False,
            "no_relation_file": "no_rel.csv",
            "is_train": True,
            "remove_cellline_organismtaxon": True
        }
        df = pd.read_json("annotation_data/new_train_splits/split_0/data.json", orient="table")
        train_dataset = SentenceDataset([df], tokenizer, config)
        first_elem = train_dataset[1]
        sent = first_elem["sentence"]
        tokens = train_dataset.tokenizer.convert_ids_to_tokens(first_elem["input_ids"])
        print(sent, first_elem["positions"])
        for idx in first_elem["positions"]:
            print(tokens[idx-1], tokens[idx], tokens[idx+1])
        input()
        print(first_elem)