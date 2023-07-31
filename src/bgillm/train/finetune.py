import logging
import math
import os
import sys

import datasets
import evaluate
import transformers
from transformers import Trainer, TrainingArguments, default_data_collator, set_seed
from transformers.trainer_utils import get_last_checkpoint

from bgillm.data.data import Data, DataTrainingArguments
from bgillm.model.model import CausalLMModel, ModelArguments


logger = logging.getLogger(__name__)


class Finetune:
    def __init__(self, model_args:ModelArguments, data_args:DataTrainingArguments, training_args:TrainingArguments, *args, **kwargs):
        self.model_args = model_args
        self.data_args = data_args
        self.training_args = training_args

        # Setup logging
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            handlers=[logging.StreamHandler(sys.stdout)],
        )

        # if training_args.should_log:
        #     transformers.utils.logging.set_verbosity_info()

        log_level = training_args.get_process_log_level()
        logger.setLevel(log_level)
        datasets.utils.logging.set_verbosity(log_level)
        transformers.utils.logging.set_verbosity(log_level)
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()

        # Log on each process the small summary:
        logger.warning(
            f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
            + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
        )
        logger.info(f"Training/evaluation parameters {training_args}")

        # Detecting last checkpoint.
        last_checkpoint = None
        if (
            os.path.isdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
        ):
            last_checkpoint = get_last_checkpoint(training_args.output_dir)
            if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
                raise ValueError(
                    f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                    "Use --overwrite_output_dir to overcome."
                )
            elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
                logger.warning(
                    f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                    "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
                )
        self.last_checkpoint = last_checkpoint
        # Set seed before initializing model.
        set_seed(training_args.seed)

    def train(self, model: CausalLMModel, data: Data):
        training_args = self.training_args
        model_args = self.model_args
        data_args = self.data_args

        train_dataset = data.train_dataset
        tokenizer = model.tokenizer

        if training_args.do_eval:
            eval_dataset = data.eval_dataset

            def preprocess_logits_for_metrics(logits, labels):
                if isinstance(logits, tuple):
                    # Depending on the model and config, logits may contain extra tensors,
                    # like past_key_values, but logits always come first
                    logits = logits[0]
                return logits.argmax(dim=-1)

            metric = evaluate.load("accuracy")

            def compute_metrics(eval_preds):
                preds, labels = eval_preds
                # preds have the same shape as the labels, after the argmax(-1) has been calculated
                # by preprocess_logits_for_metrics but we need to shift the labels
                labels = labels[:, 1:].reshape(-1)
                preds = preds[:, :-1].reshape(-1)
                return metric.compute(predictions=preds, references=labels)

        # Initialize our Trainer
        trainer = Trainer(
            model=model.basemodel,
            args=self.training_args,
            train_dataset=train_dataset if training_args.do_train else None,
            eval_dataset=eval_dataset if training_args.do_eval else None,
            tokenizer=tokenizer,
            # Data collator will default to DataCollatorWithPadding, so we change it.
            data_collator=default_data_collator,
            compute_metrics=compute_metrics if training_args.do_eval else None,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics if training_args.do_eval else None,
        )

        # Training
        if training_args.do_train:
            checkpoint = None
            last_checkpoint = self.last_checkpoint
            if training_args.resume_from_checkpoint is not None:
                checkpoint = training_args.resume_from_checkpoint
            elif last_checkpoint is not None:
                checkpoint = last_checkpoint

            train_result = trainer.train(resume_from_checkpoint=checkpoint)

            if model_args.use_lora:
                trainer.model.save_pretrained(training_args.output_dir)

            trainer.save_model()  # Saves the tokenizer too for easy upload

            metrics = train_result.metrics

            max_train_samples = (
                data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
            )
            metrics["train_samples"] = min(max_train_samples, len(train_dataset))

            trainer.log_metrics("train", metrics)
            trainer.save_metrics("train", metrics)
            trainer.save_state()

        # Evaluation
        if training_args.do_eval:
            logger.info("*** Evaluate ***")

            metrics = trainer.evaluate()

            max_eval_samples = (
                data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
            )
            metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
            try:
                perplexity = math.exp(metrics["eval_loss"])
            except OverflowError:
                perplexity = float("inf")
            metrics["perplexity"] = perplexity

            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", metrics)

        kwargs = {
            "finetuned_from": model_args.model_name_or_path,
            "tasks": "bgillm-train",
        }
        if data_args.dataset_name_or_path is not None:
            kwargs["dataset_tags"] = data_args.dataset_name_or_path
            if data_args.dataset_config_name is not None:
                kwargs["dataset_args"] = data_args.dataset_config_name
                kwargs["dataset"] = f"{data_args.dataset_name_or_path} {data_args.dataset_config_name}"
            else:
                kwargs["dataset"] = data_args.dataset_name_or_path

        trainer.create_model_card(**kwargs)
