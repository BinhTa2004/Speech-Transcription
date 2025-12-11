from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class TextSummarizer:
    def __init__(self, model_name="t5-small"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def summarize(self, text: str, max_length=80) -> str:
        input_ids = self.tokenizer(
            "summarize: " + text,
            return_tensors="pt",
            truncation=True
        ).input_ids

        output = self.model.generate(
            input_ids,
            max_length=max_length,
            min_length=20,
            length_penalty=1.5,
            num_beams=4
        )

        return self.tokenizer.decode(output[0], skip_special_tokens=True)
