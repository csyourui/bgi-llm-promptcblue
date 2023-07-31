import json
import logging
import math
import os
import random
import sys
from dataclasses import dataclass, field
from typing import List, Optional, Union

import deepspeed
import numpy as np
import torch
from datasets import Dataset, load_dataset
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, HfArgumentParser
from transformers.deepspeed import HfDeepSpeedConfig


logger = logging.getLogger('generate')

@dataclass
class GenerateArguments:
    """
    Arguments GenerateArguments.
    """
    ### model
    model_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Finetuned model path"
            )
        },
    )

    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )

    load_in_8bit: Optional[bool] = field(
        default=False,
        metadata={
            "help": (
                "whether use 8 bit to inference"
            )
        },
    )

    deepspeed: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Enable deepspeed and pass the path to deepspeed json config file (e.g. ds_config.json) or an already"
                " loaded json file as a dict"
            )
        },
    )

    ### tokenizer
    add_eos_token: Optional[bool] = field(
        default=False,
        metadata={
            "help": (
                "whether add eos token"
            )
        },
    )
    use_fast: Optional[bool] = field(
        default=False,
        metadata={
            "help": (
                "whether use fast tokenizer"
            )
        },
    )

    ### data
    data_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Question data file path"
            )
        },
    )
    random_shuffle: Optional[bool] = field(
        default=False,
        metadata={
            "help": (
                "shuffle in dataloader"
            )
        },
    )

    ### generate
    output_path_name: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Output file path and name"
            )
        },
    )
    local_rank: int = field(
        default=-1,
        metadata={"help": "For distributed training: local_rank"
        }
    )

    max_new_tokens: Optional[int] = field(
        default=20,
        metadata={
            "help": (
                "The maximum new tokens to generate"
            )
        },
    )

    inference_batch_size_per_device: Optional[int] = field(
        default=1,
        metadata={
            "help": (
                "every device will infer {inference_batch_size_per_device}"
                " samples in parallel. The inferred results will be concatenaed"
                " with inputs and attach a reward."
            ),
        },
    )

    max_generate_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The maximum generate sample."
            )
        },
    )

    temperature: Optional[float] = field(
        default=None,
        metadata={
            "help": (
                "The temperature of generate"
            )
        },
    )

    debug: bool = field(
        default=False,
        metadata={
            "help": (
                "For debug"
            )
        },
    )

    def __post_init__(self):
        if self.model_path is None or self.data_path is None:
            raise ValueError(
                "--model_path ??? --data_path ???"
            )

def set_random_seed(seed: int):
    """
    Set the random seed for `random`, `numpy`, `torch`, `torch.cuda`.

    Parameters
    ------------
    seed : int
        The default seed.

    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def batchlize(examples: list, batch_size: int, random_shuffle: bool):
    size = 0
    dataloader = []
    length = len(examples)
    if (random_shuffle):
        random.shuffle(examples)
    while size < length:
        if length - size > batch_size:
            dataloader.append(examples[size : size+batch_size])
            size += batch_size
        else:
            dataloader.append(examples[size : size+(length-size)])
            size += (length - size)
    return dataloader




class Generator():
    def __init__(self, gen_args: GenerateArguments, ds_config = None):
        self.gen_args=gen_args


        self.local_rank = int(os.getenv("LOCAL_RANK", "0"))
        self.world_size = int(os.getenv("WORLD_SIZE", "1"))
        torch.cuda.set_device(self.local_rank)  # NOTE: cpu-only machine will have error

        # batch size has to be divisible by world_size, but can be bigger than world_size
        train_batch_size = self.gen_args.inference_batch_size_per_device * self.world_size

        self.gen_args.minibatch_size = train_batch_size




        # TODO: dsconfig
        self.ds_config = None

        # data
        raw_data = load_dataset(gen_args.data_path)['test']
        self.raw_data = raw_data

        input_column = self.raw_data["input"]
        newinput_column = [f"Write a response that appropriately completes the Input.\n\nInput:\n{input}\n\nResponse:\n" for input in input_column]
        self.raw_data= self.raw_data.add_column("newinput", newinput_column)

        logger.info(f"raw_data: {self.raw_data}")

        # token
        tokenizer_kwargs = {
            "add_eos_token" : gen_args.add_eos_token,
            "use_fast": False,
            "trust_remote_code":True,
        }
        tokenizer = AutoTokenizer.from_pretrained(gen_args.model_path, **tokenizer_kwargs)
        tokenizer.padding_side = "left"  # necessary for llama, gpt2 and other decoder models
        self.tokenizer = tokenizer
        ## tokenize data
        def tokenize_function(examples):
            output = tokenizer(examples["newinput"], truncation=True)
            return output

        tokenized_datasets = self.raw_data.map(
            tokenize_function,
            batched=True,
        )
        self.tokenized_datasets = tokenized_datasets
        logger.info(f"tokenized_dataset: {tokenized_datasets}")

        # model
        config = AutoConfig.from_pretrained(gen_args.model_path, trust_remote_code=True)
        # TODO: model can be remove for not use (only engine!)
        dschf = HfDeepSpeedConfig(ds_config)
        assert dschf is not None

        model = AutoModelForCausalLM.from_pretrained(
            gen_args.model_path,
            config=config,
            torch_dtype=gen_args.torch_dtype,
            load_in_8bit=gen_args.load_in_8bit,
            trust_remote_code=True,
        )
        self.model = model
        deepspeed.init_distributed()
        self.engine = deepspeed.initialize(model=self.model, config_params=ds_config)[0]
        self.engine.module.eval()

    def run_inference(self):
        gen_args = self.gen_args
        tokenized_datasets = self.tokenized_datasets

        if gen_args.max_generate_samples is not None and gen_args.max_generate_samples < len(tokenized_datasets):
            tokenized_datasets = tokenized_datasets.select(range(gen_args.max_generate_samples))


        splice_number = math.ceil(len(tokenized_datasets) / 8)
        index = range(gen_args.local_rank*splice_number, min(len(tokenized_datasets), (gen_args.local_rank+1)*splice_number))
        tokenized_datasets = tokenized_datasets.select(index)
        print(len(tokenized_datasets))


        # begin inference
        dataloader, data_size = self.__create_dataloader(tokenized_datasets)
        print(f"temp:{self.gen_args.temperature}")
        target = []
        for batch_index, batch in tqdm(enumerate(dataloader), total=data_size):
            current_batch = batch[0]
            inputs = torch.tensor([current_batch["input_ids"]]).to(device=self.local_rank)
            outputs = self.__generate(inputs)
            raw_out = self.__batch_decode(outputs, skip_special_tokens=True)
            # 增加处理
            text_out = [self.__postprocess(raw) for raw in raw_out]
            text_out = [text.strip() for text in text_out]
            #text_out = self.__batch_postprocess(raw_out)
            target.append(text_out[0])

        tokenized_datasets = tokenized_datasets.remove_columns(column_names=["input_ids", "attention_mask", "target"])
        tokenized_datasets = tokenized_datasets.add_column(name="target", column=target)
        dataframe = tokenized_datasets.to_pandas()
        dataframe  = dataframe [[ 'input', 'target', 'answer_choices', 'task_type', 'task_dataset', 'sample_id']]
        tokenized_datasets = Dataset.from_pandas(dataframe)
        tokenized_datasets.to_json(gen_args.output_path_name + "_" + str(gen_args.local_rank)+ ".json", force_ascii=False)


    def __batch_encode(self, input: Union[str, List[str]], *args, **kwargs ) -> Union[List[int], List[List[int]]]:
        if isinstance(input, list):
            return self.tokenizer(text=input, *args, **kwargs)
        elif isinstance(input, str):
            return self.tokenizer.encode(text=input, *args, **kwargs)
        else:
            raise NotImplementedError(f'type "{type(input)}" cannot be encoded')

    def __batch_decode(self, input, *args, **kwargs ) -> Union[str, List[str]]:
        if isinstance(input, List):
            input=torch.tensor(input)
        if input.dim()==2:
            return self.tokenizer.batch_decode(input, *args, **kwargs)
        else:
            return self.tokenizer.decode(input, *args, **kwargs)

    def __generate(self, inputs, *args, **kwargs):

        with torch.no_grad():
            outputs = self.engine.module.generate(
                input_ids=inputs,
                synced_gpus=True,
                max_new_tokens=self.gen_args.max_new_tokens,
                temperature=self.gen_args.temperature,
                *args,
                **kwargs
            )
        return outputs

    def __create_dataloader(self, dataset: Dataset):
        input = [ example["newinput"] for example in dataset]
        input_ids = [ example["input_ids"] for example in dataset]

        dataset_size = len(input)
        dataset_buf = []
        for idx in range(dataset_size):
            dataset_buf.append({
                "input": input[idx],
                "input_ids": input_ids[idx],
                "input_idx": idx
            })
        dataloader = batchlize(
            dataset_buf,
            batch_size=self.gen_args.minibatch_size,
            random_shuffle=self.gen_args.random_shuffle,
        )
        print(f"Successfully create dataloader with size {len(dataloader)},batch_size {self.gen_args.minibatch_size}.")

        return dataloader, dataset_size

    def __postprocess(self, s):
        index = s.rfind("Response:\n")
        result = s
        if index != -1:
            result = s[index + len("Response:\n"):]
        return result

    def __batch_postprocess(self, texts):

        if isinstance(texts, List):
            res = []
            for text in texts:
                res.append(self.__postprocess(text))
            return res
        else:
            return self.__postprocess(texts)



def main():
    set_random_seed(3407)
    # args
    parser = HfArgumentParser(GenerateArguments)
    gen_args:GenerateArguments = parser.parse_args_into_dataclasses()[0]
    with open (gen_args.deepspeed, "r") as f:
        ds_config = json.load(f)

    logger.info(f"args: {gen_args}")


    generator = Generator(gen_args, ds_config=ds_config)
    generator.run_inference()

if __name__ == "__main__":

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO)

    main()
