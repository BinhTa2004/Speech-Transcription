from transformers import PegasusTokenizer, PegasusForConditionalGeneration

model_name = "google/pegasus-xsum"
tokenizer = PegasusTokenizer.from_pretrained(model_name)
model = PegasusForConditionalGeneration.from_pretrained(model_name)

text = "summarize: This is a long text ..."

inputs = tokenizer(text, return_tensors="pt", truncation=True)
summary = model.generate(**inputs, max_length=80)
print(tokenizer.decode(summary[0], skip_special_tokens=True))
