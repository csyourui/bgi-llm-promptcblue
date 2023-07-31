import os
import sys


sys.path.append("./src")
from bgillm.data import DataTrainingArguments, QAData
from bgillm.model.model import BaichuanModel, ModelArguments
from bgillm.train.finetune import Finetune
from bgillm.utils.utils import set_random_seed
from transformers import HfArgumentParser, TrainingArguments


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Initialization
    job = Finetune(model_args=model_args, data_args=data_args, training_args=training_args)

    model = BaichuanModel(model_args=model_args)
    tokenizer = model.tokenizer

    data = QAData(data_args=data_args, training_args=training_args, tokenizer=tokenizer)

    job.train(model=model, data=data)


if __name__ == "__main__":
    main()
