from flask import Flask, render_template, request
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

app = Flask(__name__)

# Load the trained model
model_path = "models/model.pt"
model_state_dict = torch.load(model_path, map_location=torch.device("cpu"))
model = T5ForConditionalGeneration.from_pretrained("t5-base")
model.load_state_dict(model_state_dict)

# Load the tokenizer
tokenizer_path = "models/tokenizer"
tokenizer = T5Tokenizer.from_pretrained(tokenizer_path, do_lower_case=True)

# Set the model to evaluation mode
model.eval()

# Paraphrase function
def paraphrase(text):
    input_ids = tokenizer.encode(text, return_tensors="pt")
    outputs = model.generate(
        input_ids,
        max_length=1000,
        num_beams=10,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        top_k=50,
        temperature=0.7,
        early_stopping=True
    )
    paraphrased_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return paraphrased_text



@app.route("/")
def home():
    return render_template("index.html")

@app.route("/paraphrase", methods=["POST"])
def paraphrase_text():
    original_text = request.form.get("input-text")
    paraphrased_text = paraphrase(original_text)
    return render_template("index.html", paraphrased_text=paraphrased_text)

if __name__ == "__main__":
    app.run(debug=True)
