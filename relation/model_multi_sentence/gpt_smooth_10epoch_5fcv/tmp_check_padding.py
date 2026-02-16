from transformers import AutoTokenizer

# 1. Llama 3.1 QLora checkpoint
path1 = "meta-llama_llama-3.1-8b-instruct/runs_litcoin_600_typed_entity_marker_punct_qlora_bs16_lr3e-05_ls0.02/split_0/checkpoint-best"
tok1 = AutoTokenizer.from_pretrained(path1, trust_remote_code=True)
print(f"Llama QLora checkpoint padding_side: {tok1.padding_side}")

# 2. GPT-OSS-20B DeepSpeed checkpoint
path2 = "openai_gpt-oss-20b/runs_litcoin_600_typed_entity_marker_punct_deepspeed_mxfp4_bs16_lr3e-05_ls0.02/split_0/checkpoint-best"
tok2 = AutoTokenizer.from_pretrained(path2, trust_remote_code=True)
print(f"GPT-OSS-20B DeepSpeed checkpoint padding_side: {tok2.padding_side}")
