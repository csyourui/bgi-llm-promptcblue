import logging

import transformers
from transformers import TrainingArguments

from bgillm.data.data import Data, DataTrainingArguments


logger = logging.getLogger(__name__)


class RMData(Data):
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

        if training_args.do_train:
            if "train" not in self.tokenized_datasets:
                raise ValueError("--do_train requires a train dataset")
            train_dataset = self.tokenized_datasets["train"]
            if data_args.max_train_samples is not None:
                max_train_samples = min(len(train_dataset), data_args.max_train_samples)
                train_dataset = train_dataset.select(range(max_train_samples))
            self.train_dataset = train_dataset

        if training_args.do_eval:
            if "validation" not in self.tokenized_datasets:
                raise ValueError("--do_eval requires a validation dataset")
            eval_dataset = self.tokenized_datasets["validation"]
            if data_args.max_eval_samples is not None:
                max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
                eval_dataset = eval_dataset.select(range(max_eval_samples))
            self.eval_dataset = eval_dataset

    # pre processs
    def __preprocess(self):
        if "Dahoas/rm-static" in self.data_args.dataset_name_or_path:
            for key in self.clean_datasets.keys():
                prompt_column = self.clean_datasets[key]["prompt"]
                chosen_column = self.clean_datasets[key]["chosen"]
                rejected_column = self.clean_datasets[key]["rejected"]

                text_positive = []
                text_negative = []
                for prompt, chosen, reject in zip(prompt_column, chosen_column, rejected_column):
                    text_positive.append(prompt + chosen + "</s>")
                    text_negative.append(prompt + reject + "</s>")

                self.clean_datasets[key] = self.clean_datasets[key].add_column("TEXT_POSTIVE", text_positive)
                self.clean_datasets[key] = self.clean_datasets[key].add_column("TEXT_NEGATIVE", text_negative)

        elif "Zhihu/Zhihu-KOL" in self.data_args.dataset_name_or_path:
            self.clean_datasets = self.clean_datasets.remove_columns(["SOURCE", "METADATA"])
        else:
            raise NotImplementedError(f'Unsupported dataset "{self.data_args.dataset_name_or_path}"')

    # tokenize
    def __tokenized(self, data_args, training_args, tokenizer):
        if training_args.do_train:
            column_names = list(self.clean_datasets["train"].features)
        else:
            column_names = list(self.clean_datasets["validation"].features)

        # since this will be pickled to avoid _LazyModule error in Hasher force logger loading before tokenize_function
        transformers.utils.logging.get_logger("transformers.tokenization_utils_base")

        def tokenize_function(examples):
            output = {
                "input_ids_pos": [],
                "attention_mask_pos": [],
                "input_ids_neg": [],
                "attention_mask_neg": [],
            }

            for text_pos, text_neg in zip(examples["TEXT_POSTIVE"], examples["TEXT_NEGATIVE"]):
                pos = tokenizer(text_pos)
                neg = tokenizer(text_neg)
                output["input_ids_pos"].append(pos["input_ids"])
                output["attention_mask_pos"].append(pos["attention_mask"])
                output["input_ids_neg"].append(neg["input_ids"])
                output["attention_mask_neg"].append(neg["attention_mask"])

            return output

        with training_args.main_process_first(desc="dataset map tokenization"):
            if not data_args.streaming:
                tokenized_datasets = self.clean_datasets.map(
                    tokenize_function,
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
