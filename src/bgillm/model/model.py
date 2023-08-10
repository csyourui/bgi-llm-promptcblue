import logging
from dataclasses import dataclass, field
from typing import List, Optional

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from trl import AutoModelForCausalLMWithValueHead


logger = logging.getLogger(__name__)

MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )

    reward_model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Reward model"
            )
        },
    )

    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override some existing default config settings when a model is trained from scratch. Example: "
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
            )
        },
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained config name or path if not the same as model_name"},
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"},
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Use self reomte code)."
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

    low_cpu_mem_usage: bool = field(
        default=False,
        metadata={
            "help": (
                "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded."
                "set True will benefit LLM loading time and RAM consumption."
            )
        },
    )

    ds_config: Optional[str] = field(
        default=None,
        metadata={"help": ("Deepspeed conifg json file, details in https://www.deepspeed.ai/docs/config-json/")},
    )

    finetune_type: Optional[str] = field(
        default="supervised_finetuning",
        metadata={
            "help": ("finetune type"),
            "choices": ["supervised_finetuning", "reward_modeling", "test"],
        },
    )

    use_lora: bool = field(
        default=False,
        metadata={"help": "Whether to lora."},
    )

    lora_r: int = field(
        default=8,
        metadata={"help": "the rank of the lora parameters. The smaller lora_r is , the fewer parameters lora has."},
    )

    lora_alpha: int = field(
        default=32,
        metadata={
            "help": "Merging ratio between the fine-tuned model and the original. This is controlled by a parameter called alpha in the paper."
        },
    )

    lora_dropout: float = field(
        default=0.1,
        metadata={"help": "The dropout rate in lora.linear."},
    )

    lora_target_modules: List[str] = field(
        default=None,
        metadata={"help": ("List of module names or regex expression of the module names to replace with Lora.")},
    )

    lora_modules_to_save: List[str] = field(
        default=None,
        metadata={"help": ("List of modules apart from LoRA layers to be set as trainable and saved in the final checkpoint.")},
    )

    def __post_init__(self):
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )


class CausalLMModel:
    """
    Wrapper for CausalLMModel.
    """

    def __init__(self, model_args: ModelArguments, *args, **kwargs):
        # load config
        config_kwargs = {
            "cache_dir": model_args.cache_dir,
            "revision": model_args.model_revision,
            "use_auth_token": True if model_args.use_auth_token else None,
            "trust_remote_code": model_args.trust_remote_code
        }
        if model_args.config_name:
            config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
        elif model_args.model_name_or_path:
            config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
        else:
            config = CONFIG_MAPPING[model_args.model_type]()
            logger.warning("You are instantiating a new config instance from scratch.")
            if model_args.config_overrides is not None:
                logger.info(f"Overriding config: {model_args.config_overrides}")
                config.update_from_string(model_args.config_overrides)
                logger.info(f"New config: {config}")
        self.config = config

        # load tokenizer
        # add_eos_token auto add </s> to end
        tokenizer_kwargs = {
            "use_fast": False,
            "add_eos_token" : False,
            "trust_remote_code": model_args.trust_remote_code
        }

        if model_args.tokenizer_name:
            tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
        elif model_args.model_name_or_path:
            tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
        else:
            raise ValueError(
                "You are instantiating a new tokenizer from scratch. This is not supported by this script."
                "You can do it from another script, save it, and load it from here, using --tokenizer_name."
            )
        self.tokenizer = tokenizer

        torch_dtype = (
            model_args.torch_dtype
            if model_args.torch_dtype in ["auto", None]
            else getattr(torch, model_args.torch_dtype)
        )

        # peft config
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS
            if model_args.finetune_type == "reward_modeling"
            else TaskType.CAUSAL_LM,
            inference_mode=False,
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
            target_modules=model_args.lora_target_modules,
            modules_to_save=model_args.lora_modules_to_save,
        )

        # language model for generation
        # sequence classification for rewarding
        if model_args.finetune_type == "supervised_finetuning":
            model = AutoModelForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                torch_dtype=torch_dtype,
                trust_remote_code=model_args.trust_remote_code,
            )
        elif model_args.finetune_type == "reward_modeling":
            model = AutoModelForSequenceClassification.from_pretrained(
                model_args.model_name_or_path,
                num_labels=1,
                config=config,
                torch_dtype=torch_dtype,
            )
        elif model_args.finetune_type == "reinforcement Learning":
            model = AutoModelForCausalLMWithValueHead.from_pretrained(
                config.model_name,
                peft_config=peft_config if model_args.use_lora else None,
            )
        else:
            return # for test

        # peft initialize
        if model_args.use_lora and model_args.finetune_type in ["supervised_finetuning", "reward_modeling"]:
            model.enable_input_require_grads()
            model = get_peft_model(model, peft_config)
            model.print_trainable_parameters()

        # Since we could load a larger tokenizer
        logger.warning(type(model))
        embedding_size = model.get_input_embeddings().weight.shape[0]
        if len(self.tokenizer) > embedding_size:
            model.resize_token_embeddings(len(self.tokenizer))
            logger.warning(f"You are using a larger tokennizer size :{len(self.tokenizer)} ")
        self.basemodel = model

class LLaMAModel(CausalLMModel):
    """
    Wrapper for llama.
    """

    def __init__(self, model_args, *args, **kwargs):
        super().__init__(model_args, *args, **kwargs)

        self.tokenizer.padding_side = "left"  # necessary for llama, gpt2 and other decoder models


class BaichuanModel(CausalLMModel):
    """
    Wrapper for baichuan.
    """

    def __init__(self, model_args, *args, **kwargs):
        model_args.trust_remote_code = True
        super().__init__(model_args, *args, **kwargs)
