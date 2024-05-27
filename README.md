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

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.