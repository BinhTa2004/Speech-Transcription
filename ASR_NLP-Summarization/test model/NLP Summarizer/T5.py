from transformers import T5ForConditionalGeneration, T5Tokenizer

model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

text = "summarize: This is a long text ..."
inputs = tokenizer(text, return_tensors="pt")

summary = model.generate(**inputs, max_length=80)
print(tokenizer.decode(summary[0], skip_special_tokens=True))
