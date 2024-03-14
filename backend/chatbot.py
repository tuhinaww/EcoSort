from flask import Flask, request, jsonify
import os
from embedchain import App
from flask_cors import CORS
from IndicTransTokenizer import IndicProcessor, IndicTransTokenizer
from transformers import AutoModelForSeq2SeqLM
import torch

os.environ["HUGGINGFACE_ACCESS_TOKEN"] ="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"
app_embedchain = App.from_config("/home/karthik/Desktop/IndicTrans2/huggingface_interface/mistral.yaml")

tokenizer = IndicTransTokenizer(direction="en-indic")
ip = IndicProcessor(inference=True)
model = AutoModelForSeq2SeqLM.from_pretrained("ai4bharat/indictrans2-en-indic-dist-200M", trust_remote_code=True)

flask_app = Flask(__name__)  # Fix: Correct the Flask object creation
CORS(flask_app)

languages = {
    "asm_Beng": "Assamese",
    "kas_Arab": "Kashmiri (Arabic)",
    "pan_Guru": "Punjabi",
    "ben_Beng": "Bengali",
    "kas_Deva": "Kashmiri (Devanagari)",
    "san_Deva": "Sanskrit",
    "brx_Deva": "Bodo",
    "mai_Deva": "Maithili",
    "sat_Olck": "Santali",
    "doi_Deva": "Dogri",
    "mal_Mlym": "Malayalam",
    "snd_Arab": "Sindhi (Arabic)",
    "eng_Latn": "English",
    "mar_Deva": "Marathi",
    "snd_Deva": "Sindhi (Devanagari)",
    "gom_Deva": "Konkani",
    "mni_Beng": "Manipuri (Bengali)",
    "tam_Taml": "Tamil",
    "guj_Gujr": "Gujarati",
    "mni_Mtei": "Manipuri (Meitei)",
    "tel_Telu": "Telugu",
    "hin_Deva": "Hindi",
    "npi_Deva": "Nepali",
    "urd_Arab": "Urdu",
    "kan_Knda": "Kannada",
    "ory_Orya": "Odia", 
}

# Language support list
@flask_app.route('/languages', methods=['GET'])
def get_languages():
    return jsonify(languages)

@flask_app.route('/translated_query_mistral', methods=['POST'])
def translated_query_mistral():
    try:
        data = request.get_json()
        input_text =  data.get('input_text', '')
        src_lang = data.get('src_lang', 'hin_Deva') 
        tgt_lang = 'eng_Latn'

        # Translate input text to English
        batch = ip.preprocess_batch([input_text], src_lang=src_lang, tgt_lang=tgt_lang)
        batch = tokenizer(batch, src=True, return_tensors="pt")

        with torch.inference_mode():
            translated_input = model.generate(**batch, num_beams=5, num_return_sequences=1, max_length=256)

        translated_input = tokenizer.batch_decode(translated_input, src=False)[0]

        # Query Mistral with translated input
        app_embedchain.add("https://en.wikipedia.org/wiki/Waste_management")  # Fix: Use app_embedchain instead of embedchain_app_mistral
        text = "Just give me a brief answer, just the answer part for: "
        prompt = text + translated_input
        result = app_embedchain.query(prompt)
        answer_index = result.find("Answer")
        if answer_index != -1:
            embedchain_result = result[answer_index + len("Answer: "):]
        else:
            embedchain_result = result

        # Translate Mistral result to the original source language
        batch = ip.preprocess_batch([embedchain_result], src_lang=tgt_lang, tgt_lang=src_lang)
        batch = tokenizer(batch, src=True, return_tensors="pt")

        with torch.inference_mode():
            translated_result = model.generate(**batch, num_beams=5, num_return_sequences=1, max_length=256)

        translated_result = tokenizer.batch_decode(translated_result, src=False)[0]

        # Postprocess the translated result
        translated_result = ip.postprocess_batch([translated_result], lang=src_lang)

        return jsonify({'translated_result': translated_result})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    flask_app.run(debug=True)
