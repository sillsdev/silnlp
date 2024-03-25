from transformers import AutoTokenizer, AutoConfig, AutoModelForSeq2SeqLM

from hftrim.hftrim.ModelTrimmers import M2M100Trimmer
from hftrim.hftrim.TokenizerTrimmer import TokenizerTrimmer

data = []
with open("test_S/MT/scripture/en-NIV11.txt", encoding="utf-8") as f:
    data += f.readlines()[:100]
with open("test_S/MT/scripture/es-RVR1960.txt", encoding="utf-8") as f:
    data += f.readlines()[:100]

# load pretrained config, tokenizer and model
config = AutoConfig.from_pretrained("facebook/nllb-200-distilled-600M") # facebook/m2m100_418M
tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M", use_fast=False)
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")

# trim tokenizer
tt = TokenizerTrimmer(tokenizer)
tt.make_vocab(data)
trimmed_tokenizer = tt.make_tokenizer()
print(trimmed_tokenizer)

# trim model
mt = M2M100Trimmer(model, config, trimmed_tokenizer)
mt.make_weights(tt.trimmed_vocab_ids)
trimmed_model = mt.make_model()
print(trimmed_model)

trimmed_tokenizer.save_pretrained("zzz/tokenizer")
trimmed_model.save_pretrained("zzz/model", safe_serialization=True)

rl_tok = AutoTokenizer.from_pretrained("zzz/tokenizer")
rl_model = AutoModelForSeq2SeqLM.from_pretrained("zzz/model")