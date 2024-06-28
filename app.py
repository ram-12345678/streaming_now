from flask import Flask, request, jsonify
from transformers import AutoProcessor, SeamlessM4Tv2Model
import torchaudio
import numpy as np

app = Flask(__name__)

processor = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large")
model = SeamlessM4Tv2Model.from_pretrained("facebook/seamless-m4t-v2-large")

@app.route('/translate', methods=['POST'])
def translate():
    data = request.json
    mode = data.get('mode')
    src_lang = data.get('src_lang')
    tgt_lang = data.get('tgt_lang')
    input_text = data.get('input_text')
    input_audio_url = data.get('input_audio_url')

    if mode == 'text_to_speech':
        text_inputs = processor(text=input_text, src_lang=src_lang, return_tensors="pt")
        audio_array_from_text = model.generate(**text_inputs, tgt_lang=tgt_lang)[0].cpu().numpy().squeeze()
        return jsonify(audio_array_from_text.tolist())
    elif mode == 'speech_to_speech':
        audio, orig_freq = torchaudio.load(input_audio_url)
        audio = torchaudio.functional.resample(audio, orig_freq=orig_freq, new_freq=16_000)
        audio_inputs = processor(audios=audio, return_tensors="pt")
        audio_array_from_audio = model.generate(**audio_inputs, tgt_lang=tgt_lang)[0].cpu().numpy().squeeze()
        return jsonify(audio_array_from_audio.tolist())
    elif mode == 'speech_to_text':
        audio, orig_freq = torchaudio.load(input_audio_url)
        audio_inputs = processor(audios=audio, return_tensors="pt")
        output_tokens = model.generate(**audio_inputs, tgt_lang=tgt_lang, generate_speech=False)
        translated_text_from_audio = processor.decode(output_tokens[0].tolist()[0], skip_special_tokens=True)
        return jsonify(translated_text_from_audio)
    elif mode == 'text_to_text':
        text_inputs = processor(text=input_text, src_lang=src_lang, return_tensors="pt")
        output_tokens = model.generate(**text_inputs, tgt_lang=tgt_lang, generate_speech=False)
        translated_text_from_text = processor.decode(output_tokens[0].tolist()[0], skip_special_tokens=True)
        return jsonify(translated_text_from_text)
    else:
        return jsonify({"error": "Invalid mode"}), 400

if __name__ == '__main__':
    app.run(debug=True)
