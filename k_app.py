import os
from flask import Flask, request, render_template
import torch
import torchaudio
from speechbrain.pretrained import EncoderClassifier
from sklearn.cluster import SpectralClustering
import numpy as np
from spleeter.separator import Separator
import whisper  # Importing Whisper for transcription

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'static/outputs'
app.config['SAMPLE_RATE'] = 16000


# Load pretrained speaker embedding model
embedding_model = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb", 
    savedir="pretrained_models/spkrec-ecapa-voxceleb"
)

# Load Whisper model
whisper_model = whisper.load_model("base")
import shutil

# Function to clear the output directory
def clear_output_folder():
    folder = app.config['OUTPUT_FOLDER']
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)  # Remove the file
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)  # Remove directory
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')

# Function to load and preprocess audio
def load_audio(file_path):
    audio, sr = torchaudio.load(file_path)
    if sr != app.config['SAMPLE_RATE']:
        audio = torchaudio.transforms.Resample(orig_freq=sr, new_freq=app.config['SAMPLE_RATE'])(audio)
    return audio

# Separate vocals and accompaniment
def separate_vocals(audio_path):
    audio, sr = torchaudio.load(audio_path)
    if audio.shape[0] == 1:
        audio = torch.cat([audio, audio], dim=0)
    audio_np = audio.transpose(0, 1).numpy()

    separator = Separator('spleeter:2stems')
    separation_result = separator.separate(audio_np)
    vocals = separation_result['vocals']
    accompaniment = separation_result['accompaniment']
    return vocals, accompaniment

# Generate embeddings
def generate_embeddings(audio, chunk_size=16000):
    embeddings = []
    num_chunks = audio.shape[1] // chunk_size
    for i in range(num_chunks):
        start = i * chunk_size
        end = start + chunk_size
        audio_chunk = audio[:, start:end]
        
        embedding = embedding_model.encode_batch(audio_chunk).squeeze(0).detach().cpu().numpy()
        embeddings.append(embedding)
    
    embeddings = np.array(embeddings)
    if embeddings.ndim > 2:
        embeddings = embeddings.reshape(embeddings.shape[0], -1)
    return embeddings

# Apply Spectral Clustering
def apply_spectral_clustering(embeddings, n_clusters):
    spectral = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors', random_state=0)
    labels = spectral.fit_predict(embeddings)
    return labels

# Separate speakers based on clustering labels
def separate_speakers(audio, labels, chunk_size=16000):
    separated_audio = {i: [] for i in range(app.config['NUM_SPEAKERS'])}
    num_chunks = audio.shape[1] // chunk_size

    for i in range(num_chunks):
        start = i * chunk_size
        end = start + chunk_size
        label = labels[i]
        separated_audio[label].append(audio[:, start:end])

    for speaker in separated_audio:
        separated_audio[speaker] = torch.cat(separated_audio[speaker], dim=1)
    return separated_audio

# Transcribe audio to text using Whisper
def audio_to_text_whisper(audio_path):
    result = whisper_model.transcribe(audio_path)
    return result["text"]

# Route to render the main page and handle file upload
@app.route("/", methods=["GET", "POST"])
def index():
    transcript = None
    if request.method == "POST":
        # Clear previous output files
        clear_output_folder()

        # Set number of speakers from form data, with a default if not provided
        num_speakers = request.form.get("num_speakers")
        app.config['NUM_SPEAKERS'] = int(num_speakers) if num_speakers else 2

        # Handle file upload and processing
        file = request.files.get("file")
        if file and file.filename.endswith(".mp3"):
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)

            # Process audio and perform clustering/transcription
            audio = load_audio(file_path)
            embeddings = generate_embeddings(audio)
            labels = apply_spectral_clustering(embeddings, app.config['NUM_SPEAKERS'])
            separated_audio = separate_speakers(audio, labels)

            # Generate transcript data
            transcript = {f"Speaker {i+1}": [] for i in range(app.config['NUM_SPEAKERS'])}
            for speaker_id, speaker_audio in separated_audio.items():
                audio_output_path = os.path.join(app.config['OUTPUT_FOLDER'], f"speaker_{speaker_id+1}_output.wav")
                torchaudio.save(audio_output_path, speaker_audio, app.config['SAMPLE_RATE'])

                # Transcribe audio for each speaker
                text = audio_to_text_whisper(audio_output_path)
                transcript[f"Speaker {speaker_id+1}"].append({
                    "audio_url": f"outputs/speaker_{speaker_id+1}_output.wav",
                    "text": text
                })

        return render_template("index.html", transcript=transcript)

    return render_template("index.html", transcript=transcript)

# Run Flask app
if __name__ == "__main__":
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)
    app.run(host="0.0.0.0", port=5000)
