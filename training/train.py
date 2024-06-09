"""
This module contains the Trainer class which is responsible for training whisper on predicting lyrics.
"""

import evaluate
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperTokenizer,
    logging,
)

from training.collator import DataCollatorSpeechSeq2SeqWithPadding

logging.set_verbosity_warning()


class Trainer:
    """
    A class that represents the trainer for the whisper model.
    """

    def __init__(
        self,
        model_name="openai/whisper-tiny",
        task="transcribe",
        output_dir="./whisper-finetuned",
    ):
        """Function to initialize the Trainer class.

        Args:
            model_name (str, optional): _description_. Defaults to "openai/whisper-tiny".
            task (str, optional): _description_. Defaults to "transcribe".
            output_dir (str, optional): _description_. Defaults to "./whisper-finetuned".
        """

        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)
        self.tokenizer = WhisperTokenizer.from_pretrained(model_name, task=task)

        self.processor = WhisperProcessor.from_pretrained(model_name, task=task)

        self.model = WhisperForConditionalGeneration.from_pretrained(model_name)
        self.model.generation_config.task = task

        self.model.generation_config.forced_decoder_ids = None
        self.metric = evaluate.load("wer")

        self.data_collator = DataCollatorSpeechSeq2SeqWithPadding(
            processor=self.processor,
            decoder_start_token_id=self.model.config.decoder_start_token_id,
        )

        self._ouput_dir = output_dir

    def _compute_metrics(self, pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # replace -100 with the pad_token_id
        label_ids[label_ids == -100] = self.tokenizer.pad_token_id

        # we do not want to group tokens when computing the metrics
        pred_str = self.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = self.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        print(pred_str[0])
        print(label_str[0])

        wer = 100 * self.metric.compute(predictions=pred_str, references=label_str)

        return {"wer": wer}

    def train(self, dataset):
        """
        A method that trains the model.
        :return:
        """

        training_args = Seq2SeqTrainingArguments(
            output_dir=self._ouput_dir,  # change to a repo name of your choice
            per_device_train_batch_size=8,
            gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
            learning_rate=1e-5,
            warmup_steps=500,
            max_steps=4000,
            gradient_checkpointing=True,
            fp16=True,
            eval_strategy="steps",
            per_device_eval_batch_size=8,
            predict_with_generate=True,
            generation_max_length=225,
            save_steps=80,
            eval_steps=80,
            logging_steps=25,
            report_to=["tensorboard"],
            load_best_model_at_end=True,
            metric_for_best_model="wer",
            greater_is_better=False,
            push_to_hub=False,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )

        trainer = Seq2SeqTrainer(
            args=training_args,
            model=self.model,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            data_collator=self.data_collator,
            compute_metrics=self._compute_metrics,
            tokenizer=self.processor.feature_extractor,
        )

        self.processor.save_pretrained(training_args.output_dir)

        trainer.train()
