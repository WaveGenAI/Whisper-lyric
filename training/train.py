"""
This module contains the Trainer class which is responsible for training whisper on predicting lyrics.
"""

import warnings
from functools import partial

import evaluate
import librosa
import numpy as np
import torch
from datasets import Dataset
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from transformers.models.whisper.english_normalizer import BasicTextNormalizer

from training.collator import DataCollatorSpeechSeq2SeqWithPadding

METRIC = evaluate.load("wer")

NORMALIZER = BasicTextNormalizer()

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


class Trainer:
    """
    A class that represents the trainer for the whisper model.
    """

    def __init__(
        self,
        dataset=None,
        model_name="openai/whisper-small",
    ):
        """
        The constructor for the Trainer class.
        The dataset is optional and can be added later with the method process_dataset.
        The dataset should be formated and already mapped to the columns "audio" and "lyrics" and ready for training.
        :param dataset: The dataset to train the model on.
        """
        self.processor = WhisperProcessor.from_pretrained(model_name, language="en", task="transcribe")
        self.model = WhisperForConditionalGeneration.from_pretrained(model_name)
        self.dataset = dataset
        self.data_collator = DataCollatorSpeechSeq2SeqWithPadding(self.processor)
        self.prepare_tokenizer()

    def prepare_tokenizer(self) -> None:
        """
        A method that adds special tokens i.e. tags to the tokenizer.
        :return: None
        """
        special_tokens_to_add = []
        for i in range(1, 5):
            special_tokens_to_add.append(f"[VERSE {i}]")
        special_tokens_to_add.append("[CHORUS]")
        special_tokens_to_add.append("[BRIDGE]")
        self.processor.tokenizer.add_special_tokens(
            {"additional_special_tokens": special_tokens_to_add}
        )
        self.model.resize_token_embeddings(len(self.processor.tokenizer))

    def process_dataset(self, dataset, chunk_id) -> Dataset:
        """
        A method that processes the dataset.
        :return: None
        """

        def prepare_dataset(example):
            target_sr = self.processor.feature_extractor.sampling_rate
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=FutureWarning)
                warnings.filterwarnings("ignore", category=UserWarning)
                audio, sr = librosa.load(example["audio"], sr=None)
            audio = librosa.resample(
                np.asarray(audio),
                orig_sr=sr,
                target_sr=target_sr,
            )

            example = self.processor(
                audio=audio,
                sampling_rate=target_sr,
                text=example["lyrics"],
            )

            # compute input length of audio sample in seconds
            example["input_length"] = len(audio) / sr

            return example
        if chunk_id == -1:
            last_chunk_size = len(dataset) % 1000
            small_dataset = Dataset.from_dict(dataset[-last_chunk_size:])
        else:
            small_dataset = Dataset.from_dict(dataset[chunk_id*1000:chunk_id*1000+1000])
        self.dataset = small_dataset.map(
            prepare_dataset,
            remove_columns=small_dataset.column_names,
            num_proc=1,
        )

        max_input_length = 30.0

        def is_audio_in_length_range(length):
            return length < max_input_length

        self.dataset = self.dataset.filter(
            is_audio_in_length_range,
            input_columns=["input_length"],
        )
        return self.dataset

    def compute_metrics(self, pred):
        """
        A method that computes the metrics.
        :param pred: The predictions of the model.
        :return: The metrics.
        """
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # replace -100 with the pad_token_id
        label_ids[label_ids == -100] = self.processor.tokenizer.pad_token_id

        # we do not want to group tokens when computing the metrics
        pred_str = self.processor.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = self.processor.batch_decode(label_ids, skip_special_tokens=True)

        # compute orthographic wer
        wer_ortho = 100 * METRIC.compute(predictions=pred_str, references=label_str)

        # compute normalised WER
        pred_str_norm = [NORMALIZER(pred) for pred in pred_str]
        label_str_norm = [NORMALIZER(label) for label in label_str]
        # filtering step to only evaluate the samples that correspond to non-zero references:
        pred_str_norm = [
            pred_str_norm[i]
            for i in range(len(pred_str_norm))
            if len(label_str_norm[i]) > 0
        ]
        label_str_norm = [
            label_str_norm[i]
            for i in range(len(label_str_norm))
            if len(label_str_norm[i]) > 0
        ]

        wer = 100 * METRIC.compute(predictions=pred_str_norm, references=label_str_norm)

        return {"wer_ortho": wer_ortho, "wer": wer}

    def train(self):
        """
        A method that trains the model.
        :return:
        """

        self.model.generate = partial(
            self.model.generate, language="en", task="transcribe", use_cache=True
        )
        training_args = Seq2SeqTrainingArguments(
            output_dir="./train",
            per_device_train_batch_size=10,
            per_device_eval_batch_size=8,
            num_train_epochs=3,
            learning_rate=1e-5,
            lr_scheduler_type="linear",
            warmup_steps=50,
            gradient_checkpointing=False,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            bf16_full_eval=torch.cuda.is_bf16_supported(),
            fp16_full_eval=not torch.cuda.is_bf16_supported(),
            evaluation_strategy="steps",
            eval_steps=75,
            optim="adamw_8bit",
            predict_with_generate=True,
            generation_max_length=350,
            logging_steps=25,
            metric_for_best_model="wer",
            greater_is_better=False,
        )

        trainer = Seq2SeqTrainer(
            args=training_args,
            model=self.model,
            train_dataset=self.dataset["train"],
            eval_dataset=self.dataset["test"],
            data_collator=self.data_collator,
            compute_metrics=self.compute_metrics,
            tokenizer=self.processor,
        )
        return trainer.train()

    def save_model(self, path: str) -> None:
        """
        A method that saves the model.
        :param path: The path to save the model.
        :return: None
        """

        self.model.save_pretrained(path)
        self.processor.save_pretrained(path)
