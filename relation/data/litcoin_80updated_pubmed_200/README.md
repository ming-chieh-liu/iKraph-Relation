# Litcoin 80 Updated PubMed 200 - Biomedical Relation Extraction Data Processing

This directory contains scripts for processing manually annotated biomedical relation data from the updated Litcoin dataset, focusing on 80 updated records and 200 PubMed documents.

## Overview

The pipeline enriches manually annotated relation data with entity IDs from a reference dataset and converts it into a standardized format for training:
1. Add entity IDs to annotated data by matching with reference dataset
2. Generate standardized training data with entity type normalization

## Data Files

### Input Files
- `combined_for_annotation_sentence_level.json` - Reference dataset containing entity ID mappings and sentence-level annotations
- `DS/` - Directory containing gold standard annotated relation files (multiple batches)
  - Format: `GS_for_PreLabels_RE_batch{X}_{Y}.json`
  - Contains manually annotated relation data without entity IDs

### Intermediate Files
- `DS_with_id/` - Directory containing annotated files enriched with entity IDs
  - Same file structure as `DS/` but with added `id_1` and `id_2` fields

### Output Files
- `new_annotated_train_80updated_pubmed200_pd.json` - Final training data in pandas-compatible JSON format

## Scripts

### 1. add_entity_id.py

**Purpose**: Enriches gold standard annotations with entity IDs from the reference dataset.

**Functionality**:
- Loads reference dataset (`combined_for_annotation_sentence_level.json`) to build entity lookup by Name
- Processes all JSON files in `DS/` directory
- For each annotation entry:
  - Matches by `Name` field to find corresponding reference entry
  - Validates entity consistency (text, type, spans)
  - Adds `id_1` and `id_2` fields from reference data
- Outputs enriched files to `DS_with_id/` directory

**Validation Checks**:
- PMID must match between annotated and reference data
- Number of relation annotations must match
- Entity names, types, and spans must match exactly

**Dependencies**:
- json, pandas, pathlib
- Input: `combined_for_annotation_sentence_level.json`, `DS/*.json`
- Output: `DS_with_id/*.json`

### 2. gen_input_data.py

**Purpose**: Generates standardized training data from entity-enriched annotations.

**Functionality**:
- Reads all files from `DS_with_id/` directory
- Extracts batch and sentence IDs from filename pattern: `NER_batch{X}_{Y}_sent_{Z}`
- Normalizes entity types:
  - `Gene` → `GeneOrGeneProduct`
  - `Chemical` → `ChemicalSubstance`
  - `Disease` → `DiseaseOrPhenotypicFeature`
- Creates relation entries for all entity span combinations using `itertools.product`
- Generates unique relation IDs: `True.{pmid}.{id_1}.{id_2}.{batch_id}.{sent_id}.combination{N}`
- Tracks duplicate combinations within same sentence using composite keys

**Output Format**:
- Pandas-compatible JSON with standardized fields:
  - `abstract_id`: PubMed ID (integer)
  - `relation_id`: Unique identifier string
  - `entity_a_id`, `entity_b_id`: Entity identifiers
  - `text`: Sentence text
  - `annotated_type`: Relation type from gold standard
  - `entity_a`, `entity_b`: [start, end, type] tuples
  - `duplicated_flag`: Always False

**Dependencies**:
- json, pandas, pathlib, itertools
- Input: `DS_with_id/*.json`
- Output: `new_annotated_train_80updated_pubmed200_pd.json`

## Usage

Run scripts in the following order:

### Step 1: Add Entity IDs
```bash
python add_entity_id.py
```

This enriches annotations in `DS/` with entity IDs and saves to `DS_with_id/`.

### Step 2: Generate Training Data
```bash
python gen_input_data.py
```

This creates `new_annotated_train_80updated_pubmed200_pd.json` with standardized training data.

## Data Schema

### combined_for_annotation_sentence_level.json (Reference)
```json
{
  "Name": "NER_batch1_1_sent_1",
  "PMID": "11773892",
  "text": "End-stage renal disease (ESRD) after...",
  "Relation_Annotation": [
    {
      "entity_1": "calcineurin",
      "type_1": "Gene",
      "id_1": "5530",
      "entity_1_spans_list": [[83, 94]],
      "entity_2": "End-stage renal disease",
      "type_2": "Disease",
      "id_2": "D007676",
      "entity_2_spans_list": [[0, 23], [25, 29]]
    }
  ]
}
```

### DS/ Files (Input - Without IDs)
```json
{
  "Name": "NER_batch1_1_sent_1",
  "PMID": "11773892",
  "text": "End-stage renal disease (ESRD) after...",
  "Relation_Annotation": [
    {
      "entity_1": "calcineurin",
      "type_1": "Gene",
      "entity_1_spans_list": [[83, 94]],
      "entity_2": "End-stage renal disease",
      "type_2": "Disease",
      "entity_2_spans_list": [[0, 23], [25, 29]],
      "gold_standard": {
        "relation_type": "NOT",
        "direction": "No Direction"
      }
    }
  ]
}
```

### DS_with_id/ Files (Intermediate - With IDs)
Same as DS/ files but with added `id_1` and `id_2` fields in each relation annotation.

### new_annotated_train_80updated_pubmed200_pd.json (Output)
```json
{
  "abstract_id": 11773892,
  "relation_id": "True.11773892.5530.D007676.1.1.combination0",
  "entity_a_id": "5530",
  "entity_b_id": "D007676",
  "text": "End-stage renal disease (ESRD) after...",
  "document_level_type": "",
  "document_level_novel": "",
  "annotated_type": "NOT",
  "annotated_relation_words": [],
  "entity_a": [83, 94, "GeneOrGeneProduct"],
  "entity_b": [0, 23, "DiseaseOrPhenotypicFeature"],
  "duplicated_flag": false
}
```

## Notes

- Entity IDs are matched by the `Name` field (e.g., `NER_batch1_1_sent_1`)
- Multiple entity spans generate multiple training instances via Cartesian product
- Combination IDs track multiple instances within the same sentence context
- Entity type normalization ensures consistency with Litcoin ontology standards
- The pipeline validates data consistency at each step with assertions
