import logging
from itertools import chain

import torch
from transformers import TrainingArguments

from bgillm.data.data import Data, DataTrainingArguments


logger = logging.getLogger(__name__)

LLAMA_EOS_TOKEN="</s>"

class QAData(Data):
    def __init__(
        self,
        data_args: DataTrainingArguments,
        training_args: TrainingArguments,
        tokenizer,
        *args,
        **kwargs,
    ):
        super().__init__(data_args)
        self.__preprocess()

        self.tokenized_datasets = self.__tokenized(data_args, training_args, tokenizer)

        # for pretraining not sft
        if not data_args.not_group:
            self.tokenized_datasets = self.__group_texts(data_args, training_args, tokenizer)

        if training_args.do_train:
            if "train" not in self.tokenized_datasets:
                raise ValueError("--do_train requires a train dataset")
            train_dataset = self.tokenized_datasets["train"]
            if data_args.max_train_samples is not None:
                max_train_samples = min(len(train_dataset), data_args.max_train_samples)
                train_dataset = train_dataset.select(range(max_train_samples))
            train_dataset = train_dataset.filter(
                lambda x: len(x["input_ids"]) <= tokenizer.model_max_length,
                num_proc=data_args.preprocessing_num_workers,
            )
            self.train_dataset = train_dataset

        if training_args.do_eval:
            if "validation" not in self.tokenized_datasets:
                raise ValueError("--do_eval requires a validation dataset")
            eval_dataset = self.tokenized_datasets["validation"]
            if data_args.max_eval_samples is not None:
                max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
                eval_dataset = eval_dataset.select(range(max_eval_samples))
            eval_dataset = eval_dataset.filter(
                lambda x: len(x["input_ids"]) <= tokenizer.model_max_length,
                num_proc=data_args.preprocessing_num_workers,
            )
            self.eval_dataset = eval_dataset
        # print(self.train_dataset[31254])
        # exit()
    # pre processs
    def __preprocess(self):

        def add_eos_token(s):
            if not s.endswith(LLAMA_EOS_TOKEN):
                s += LLAMA_EOS_TOKEN
            return s

        if "Dahoas/rm-static" in self.data_args.dataset_name_or_path:
            self.clean_datasets = self.clean_datasets.rename_column("prompt", "input")
            for key in self.clean_datasets.keys():
                target_column = self.clean_datasets[key]["chosen"]
                output_column = [add_eos_token(target) for target in target_column]
                self.clean_datasets[key] = self.clean_datasets[key].add_column("output", output_column)

        elif "Zhihu/Zhihu-KOL" in self.data_args.dataset_name_or_path:
            self.clean_datasets = self.clean_datasets.remove_columns(["SOURCE", "METADATA"])

        elif "PromptCBLUE" in self.data_args.dataset_name_or_path:
            for key in self.clean_datasets.keys():
                target_column = self.clean_datasets[key]["target"]
                output_column = [add_eos_token(target) for target in target_column]
                self.clean_datasets[key] = self.clean_datasets[key].add_column("output", output_column)

                # for text
                input_column = self.clean_datasets[key]["input"]
                task_type_tolumn = self.clean_datasets[key]["task_type"]
                text_column = [
                    f"Write a response that appropriately completes the Input.\n\nInput:\n{input}\n\nResponse:\n{target}{LLAMA_EOS_TOKEN}"
                    for input, target, task_type in zip(input_column, target_column, task_type_tolumn)
                ]
                self.clean_datasets[key] = self.clean_datasets[key].add_column("TEXT", text_column)

        else:
            return

    # tokenize
    def __tokenized(self, data_args, training_args, tokenizer):
        # Preprocessing the datasets.
        # First we tokenize all the texts.
        if training_args.do_train:
            column_names = list(self.clean_datasets["train"].features)
        else:
            column_names = list(self.clean_datasets["validation"].features)

        logger.info(f"column_names {column_names}")

        max_seq_length = tokenizer.model_max_length

        logger.info(f"max_seq_length = {max_seq_length}")

        def all_tokenize_function(examples):
            output = tokenizer(examples["TEXT"])
            output["labels"] = output["input_ids"].copy()
            return output

        def tokenize_function(examples):
            input_token = tokenizer(examples["input"], return_attention_mask=False)
            output_token = tokenizer(examples["output"], return_attention_mask=False, add_special_tokens=False)

            all_input_ids = []
            all_attention_mask = []
            all_labels = []
            for s, t in zip(input_token["input_ids"], output_token["input_ids"]):
                input_ids = torch.LongTensor(s + t)[:max_seq_length]
                labels = torch.LongTensor([-100] * len(s) + t)[:max_seq_length]
                attention_mask=torch.LongTensor([1] * (len(s) + len(t)))[:max_seq_length]

                assert len(input_ids) == len(labels)
                assert len(input_ids) == len(attention_mask)

                all_input_ids.append(input_ids)
                all_attention_mask.append(attention_mask)
                all_labels.append(labels)

            output = {"input_ids": all_input_ids, "attention_mask": all_attention_mask, "labels": all_labels}
            return output

        with training_args.main_process_first(desc="dataset map tokenization"):
            if not data_args.streaming:
                tokenized_datasets = self.clean_datasets.map(
                    tokenize_function if data_args.output_loss else all_tokenize_function,
                    batched=True,
                    num_proc=data_args.preprocessing_num_workers,
                    remove_columns=column_names,
                    load_from_cache_file=not data_args.overwrite_cache,
                    desc="Running tokenizer on dataset",
                )
            else:
                tokenized_datasets = self.clean_datasets.map(
                    tokenize_function,
                    batched=True,
                    remove_columns=column_names,
                )

        return tokenized_datasets

    # group texts
    def __group_texts(self, data_args, training_args, tokenizer):
        # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
        def group_texts(examples):
            # Concatenate all texts.
            concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            # We drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict.
            # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
            total_length = (total_length // block_size) * block_size
            # Split by chunks of max_len.
            result = {
                k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
                for k, t in concatenated_examples.items()
            }
            result["labels"] = result["input_ids"].copy()
            return result

        if data_args.block_size is None:
            block_size = tokenizer.model_max_length
            if block_size > 2048:
                logger.warning(
                    "The chosen tokenizer supports a `model_max_length` that is longer than the default `block_size` value"
                    " of 2048. If you would like to use a longer `block_size` up to `tokenizer.model_max_length` you can"
                    " override this default with `--block_size xxx`."
                )
                block_size = 2048
        else:
            if data_args.block_size > tokenizer.model_max_length:
                logger.warning(
                    f"The block_size passed ({data_args.block_size}) is larger than the maximum length for the model"
                    f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
                )
            block_size = min(data_args.block_size, tokenizer.model_max_length)

        with training_args.main_process_first(desc="grouping texts together"):
            if not data_args.streaming:
                lm_datasets = self.tokenized_datasets.map(
                    group_texts,
                    batched=True,
                    num_proc=data_args.preprocessing_num_workers,
                    load_from_cache_file=not data_args.overwrite_cache,
                    desc=f"Grouping texts in chunks of {block_size}",
                )
            else:
                lm_datasets = self.tokenized_datasets.map(
                    group_texts,
                    batched=True,
                )

        return lm_datasets
