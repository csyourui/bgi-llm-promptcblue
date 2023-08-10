import copy
import logging
from dataclasses import dataclass, field
from typing import Optional

from datasets import load_dataset
from transformers.utils.versions import require_version


logger = logging.getLogger(__name__)

DEFAULT_MODEL_MAX_LENGTH = 2048


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name_or_path: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    streaming: bool = field(default=False, metadata={"help": "Enable streaming mode"})

    not_group: bool = field(default=False, metadata={"help": "Enable group data mode"})

    block_size: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Optional input sequence length after tokenization. "
                "The training dataset will be truncated in block of this size for training. "
                "Default to the model max input length for single sentence inputs (take into account special tokens)."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    keep_linebreaks: bool = field(
        default=True, metadata={"help": "Whether to keep line breaks when using TXT files or not."}
    )

    output_loss: bool = field(
        default=False, metadata={"help": "only use output as loss"}
    )


    def __post_init__(self):
        if self.streaming:
            require_version("datasets>=2.0.0", "The streaming feature requires `datasets>=2.0.0`")

        if self.dataset_name_or_path is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, a json or a txt file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`validation_file` should be a csv, a json or a txt file."


class Data:
    """
    Wrapper for Huggingface datasets

    Parameters
    ------------

    kwargs = {
        'data_args': data_args,
        'model_args': model_args,
        'training_args': training_args,
        'tokenizer': tokenizer,
    }
    """

    def __init__(
        self,
        data_args: DataTrainingArguments,
        *args,
        **kwargs,
    ):
        self.data_args = data_args
        self.raw_datasets = self.__get_raw_datasets(data_args)
        self.clean_datasets = copy.deepcopy(self.raw_datasets)
        self.train_dataset = None

    def __get_raw_datasets(self, data_args):
        if data_args.dataset_name_or_path is not None:
            # Downloading and loading a dataset from the hub.
            raw_datasets = load_dataset(
                data_args.dataset_name_or_path,
                data_args.dataset_config_name,
                # cache_dir=model_args.cache_dir,
                # use_auth_token=True if model_args.use_auth_token else None,
                streaming=data_args.streaming,
            )
            if "validation" not in raw_datasets.keys():
                raw_datasets["validation"] = load_dataset(
                    data_args.dataset_name_or_path,
                    data_args.dataset_config_name,
                    split=f"train[:{data_args.validation_split_percentage}%]",
                    # cache_dir=model_args.cache_dir,
                    # use_auth_token=True if model_args.use_auth_token else None,
                    streaming=data_args.streaming,
                )
                raw_datasets["train"] = load_dataset(
                    data_args.dataset_name_or_path,
                    data_args.dataset_config_name,
                    split=f"train[{data_args.validation_split_percentage}%:]",
                    # cache_dir=model_args.cache_dir,
                    # use_auth_token=True if model_args.use_auth_token else None,
                    streaming=data_args.streaming,
                )
        else:
            data_files = {}
            dataset_args = {}
            if data_args.train_file is not None:
                data_files["train"] = data_args.train_file
            if data_args.validation_file is not None:
                data_files["validation"] = data_args.validation_file
            extension = (
                data_args.train_file.split(".")[-1]
                if data_args.train_file is not None
                else data_args.validation_file.split(".")[-1]
            )
            if extension == "txt":
                extension = "text"
                dataset_args["keep_linebreaks"] = data_args.keep_linebreaks
            raw_datasets = load_dataset(
                extension,
                data_files=data_files,
                # cache_dir=model_args.cache_dir,
                # use_auth_token=True if model_args.use_auth_token else None,
                **dataset_args,
            )
            # If no validation data is there, validation_split_percentage will be used to divide the dataset.
            if "validation" not in raw_datasets.keys():
                raw_datasets["validation"] = load_dataset(
                    extension,
                    data_files=data_files,
                    split=f"train[:{data_args.validation_split_percentage}%]",
                    # cache_dir=model_args.cache_dir,
                    # use_auth_token=True if model_args.use_auth_token else None,
                    **dataset_args,
                )
                raw_datasets["train"] = load_dataset(
                    extension,
                    data_files=data_files,
                    split=f"train[{data_args.validation_split_percentage}%:]",
                    # cache_dir=model_args.cache_dir,
                    # use_auth_token=True if model_args.use_auth_token else None,
                    **dataset_args,
                )

        return raw_datasets
