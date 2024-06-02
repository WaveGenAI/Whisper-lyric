# Whisper-lyric

Codebase to finetune whisper for music transcription.

## Installation

To install the required dependencies, run the following command:

```bash
pip install -r requirements.txt
```

## Dataset download

To download the dataset, run the following command:

```bash
python download_dataset.py --num_images 1000
```

The dataset will be downloaded to the `data` directory.
The format of the dataset is as follows:

```
dataset
├── audio
│   ├── 0.wav
│   ├── 1.wav
│   ├── ...
└── lyrics
    ├── 0.txt
    ├── 1.txt
    ├── ...
```

where `0.wav` corresponds to the audio file and `0.txt` corresponds to the lyrics transcription of the audio file.

## Process the dataset

To process the dataset, run the following command:

```bash
python process_dataset.py --clean
```

The process will split the audio in chunks of 32 seconds and split the lyrics.

## Test the model

Here is an example of how to test the model:

```py
import librosa
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor, pipeline


model_name = "Jour/whisper-small-lyric-finetuned"
audio_file = "PATH_TO_AUDIO_FILE"

device = "cuda:0" if torch.cuda.is_available() else "cpu"
processor = WhisperProcessor.from_pretrained("openai/whisper-small")
model = WhisperForConditionalGeneration.from_pretrained(model_name)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    chunk_length_s=30,
    device=device,
)

sample, _ = librosa.load(audio_file, sr=processor.feature_extractor.sampling_rate)

prediction = pipe(sample.copy(), batch_size=8)["text"]
print(prediction)
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.