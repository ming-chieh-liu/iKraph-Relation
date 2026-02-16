# Litcoin 164 - Biomedical Relation Extraction Data Processing

This directory contains scripts for processing and augmenting biomedical relation extraction data from the Litcoin challenge dataset (version 164).

## Overview

The pipeline processes annotated biomedical literature data to:
1. Map abstract IDs to PubMed IDs (PMIDs)
2. Assign sentence IDs using consistent sentence tokenization
3. Generate additional training data by filtering and reformatting annotations

## Data Files

### Input Files
- `Annotated_Tina_RD1.json` - Original annotated relation data
- `abstracts_train_pmids.jsonl` - Training set abstracts with PMID mappings
- `abstracts_test_pmids.jsonl` - Test set abstracts with PMID mappings
- `../litcoin_600/All.json` - Complete document corpus for sentence lookup

### Output Files
- `Annotated_Tina_RD1_with_pmids.json` - Annotated data enriched with PMIDs and sentence IDs
- `annotated_train_extra_pd.json` - Additional training data in standardized format
- `annotated_data.csv` - Processed annotations in CSV format

## Scripts

### 1. add_abstract_sentence_ids.py

**Purpose**: Enriches annotated data with PubMed IDs and sentence-level identifiers.

**Functionality**:
- Maps abstract IDs to PMIDs from train/test JSONL files
- Builds sentence-level lookup using `LitcoinSentenceTokenizer`
- Assigns sentence IDs by matching annotation text to tokenized sentences
- Normalizes entity type names using TYPE_MAP:
  - `Disease` ’ `DiseaseOrPhenotypicFeature`
  - `Gene` ’ `GeneOrGeneProduct`
  - `Chemical` ’ `ChemicalEntity`

**Dependencies**:
- `SentenceTokenizer.py` (LitcoinSentenceTokenizer class)
- Input: `Annotated_Tina_RD1.json`, `abstracts_{train,test}_pmids.jsonl`, `../litcoin_600/All.json`
- Output: `Annotated_Tina_RD1_with_pmids.json`

### 2. gen_input_data.py

**Purpose**: Generates additional training data by filtering and reformatting annotations.

**Functionality**:
- Loads annotated data with PMID mappings
- Filters for documents not already in the training set (uses validation IDs)
- Creates deduplicated relation entries using composite keys: (pmid, entity_a_id, entity_b_id, sentence_id)
- Reformats data into standardized JSON structure with fields:
  - `abstract_id`: PubMed ID
  - `relation_id`: Unique identifier (format: `True.{pmid}.{relation_id}.sentence{sentence_id}.combination{combination_id}`)
  - Entity information: IDs, types, spans
  - `text`: Sentence text
  - `annotated_type`: Relation type annotation
  - `duplicated_flag`: Always False

**Dependencies**:
- pandas, json
- Input: `Annotated_Tina_RD1_with_pmids.json`, `All.json`, `/data/mliu/iKraph/relation/original_data/abstracts_train.csv`
- Output: `annotated_train_extra_pd.json`

### 3. SentenceTokenizer.py

**Purpose**: Custom sentence tokenizer for biomedical text.

**Key Features**:
- Extends NLTK Punkt tokenizer with biomedical abbreviations (e.g., i.v., i.p., i.m., e.g., i.e.)
- Handles edge cases common in scientific text:
  - Parenthetical expressions
  - Unit measurements (e.g., "24 h.")
  - Chemical/gene nomenclature (e.g., "CACNG8c.*6819A>T")
- Returns sentences with character-level offsets
- Post-processing includes custom split and merge rules

**Main Classes**:
- `LitcoinSentenceTokenizer`: Primary tokenizer class
  - `sentence_tokenize(text)`: Returns list of sentences
  - `sentence_tokenize_with_offsets(text)`: Returns sentences with start/end offsets
  - `add_extra_abbreviations(abbreviations)`: Add custom abbreviations

## Usage

Run scripts in the following order:

### Step 1: Add PMID and Sentence IDs
```bash
python add_abstract_sentence_ids.py
```

This will create `Annotated_Tina_RD1_with_pmids.json` with enriched metadata.

### Step 2: Generate Additional Training Data
```bash
python gen_input_data.py
```

This will create `annotated_train_extra_pd.json` with filtered training examples.

## Data Schema

### Annotated_Tina_RD1_with_pmids.json
```json
{
  "abstract_id": "123456",
  "pmid": "123456",
  "sentence_id": 5,
  "relation_id": "R1",
  "sentence": "Aspirin inhibits COX2.",
  "anno_type": "inhibits",
  "entity_a_id": "MESH:D001241",
  "entity_a_span": [0, 7],
  "entity_a_type": "ChemicalEntity",
  "entity_b_id": "HGNC:9605",
  "entity_b_span": [17, 21],
  "entity_b_type": "GeneOrGeneProduct"
}
```

### annotated_train_extra_pd.json
```json
{
  "abstract_id": 123456,
  "relation_id": "True.123456.R1.sentence5.combination0",
  "entity_a_id": "MESH:D001241",
  "entity_b_id": "HGNC:9605",
  "text": "Aspirin inhibits COX2.",
  "annotated_type": "inhibits",
  "entity_a": [0, 7, "ChemicalEntity"],
  "entity_b": [17, 21, "GeneOrGeneProduct"],
  "duplicated_flag": false
}
```

## Notes

- The pipeline ensures consistent sentence tokenization across all annotations
- Duplicate relations within the same sentence are tracked using combination IDs
- Only documents not in the original training set are included in extra training data
- Entity type normalization follows Litcoin challenge ontology conventions