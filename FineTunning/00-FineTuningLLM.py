# Huggingface Fine Tuning

## Token taking from HF

HF (https://huggingface.co) Get Token from HF

- Token issue address: https://huggingface.co/docs/hub/security-tokens

## Reference Model Llama 3 

To use it, you need to install the huggingface_hub package in Python. (https://huggingface.co/docs/huggingface_hub/installation)

# !pip install -qU huggingface_hub
# Save the issued token in the `.env` file as `HUGGINGFACEHUB_API_TOKEN` and proceed to the next step.

# Load `HUGGINGFACEHUB_API_TOKEN`.
# Set up LangSmith tracking. https://smith.langchain.com
# !pip install langchain-teddynote
from langchain_teddynote import logging

# Enter your project name.
logging.langsmith("")
# Login to the Hugging Face model hub
from huggingface_hub import login

import os

login()
# download the model from the hub

from huggingface_hub import snapshot_download

# Download the model
local_dir = snapshot_download(
    repo_id="
    cache_dir="
)
# open llam.cpp

## cd llam.cpp ( Git Bash)

cd C:\Users\Desktop\System\LLM\langchain-kr\04-Model\llama.cpp

python convert_hf_to_gguf.py \
    "C:\Users\Desktop\System\LLM\langchain-kr\04-Model\local_model\SETBOX" \
    --outfile "C:\Users\Desktop\System\LLM\langchain-kr\04-Model\local_model\GGUF\Q8\Llma3-HMGICS-SETBOX-Q8" \
    --model-name "Llma3-SETBOX-Q8" \
    --outtype q8_0

go to the path 
C:\Users\HMGICS\Desktop\System\LLM\langchain-kr\04-Model\local_model\GGUF\Q8

ollama create Llma3-HMGICS-SETBOX-Q8 -f Modelfile

ollama create Llma3-HMGICS-SETBOX-tq2 -f Modelfile

from huggingface_hub import HfApi, HfFolder

# Get your Hugging Face token
token = HfFolder.get_token()

# Create an instance of HfApi
api = HfApi()

# Define the path to your GGUF model
gguf_model_path = r""

# Define the repository name where you want to push the model
repo_name = "stlee9048/Llma3-7B-"

# Push the model to the Hugging Face Hub
api.upload_folder(
    folder_path=gguf_model_path, repo_id=repo_name, repo_type="model", token=token
)
# Bring the data set from locally 
import os
import json

# Print the current working directory path
current_path = os.getcwd()
print(f"Current working directory: {current_path}")

# Set file path
file_path = os.path.join(current_path, "qa_pair_SETBOXRACK2.jsonl")

# Read file
with open(file_path, "r", encoding="utf-8") as file:
    for line in file:
        data = json.loads(line)
        print(data)
import os
import pandas as pd

# Print the current working directory path
current_path = os.getcwd()
print(f"Current working directory: {current_path}")

# JSONL file path
jsonl_file = "./qa_pair_SETBOXRACK.jsonl"

# Check if the file exists
if not os.path.exists(jsonl_file):
    raise FileNotFoundError(f"File not found: {jsonl_file}")

# Read JSONL file
df = pd.read_json(jsonl_file, lines=True)

# Create dataset
from datasets import Dataset
dataset = Dataset.from_pandas(df)

print(dataset)
# data set push to HF
from huggingface_hub import HfApi
import certifi
import requests

# Create HfApi instance
api = HfApi()

response = requests.get("https://huggingface.co", verify=False)
# Repository name to upload the dataset
repo_name = "stlee9048/Llama3.1"

# Push the dataset to the hub
dataset.push_to_hub(repo_name, token="")
# check the torch version in your local machine
# !pip install torch
import torch

# Get the major and minor version of the CUDA device
major_version, minor_version = torch.cuda.get_device_capability()
major_version, minor_version
import torch

print(torch.__file__)  # Print PyTorch installation path
import torch

print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"CUDA Version: {torch.version.cuda}")
print(
    f"Device Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU'}"
)
# Model Optimization: Quantization Settings
# Model lightweight: Quantization settings
from transformers import BitsAndBytesConfig
import torch

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)
from transformers import BitsAndBytesConfig
import torch

quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,  # Set to lower specs using 8-bit quantization
    bnb_8bit_compute_dtype=torch.float16,  # Use 16-bit floating point
    bnb_8bit_use_double_quant=True,  # Use double quantization
    bnb_8bit_quant_type="nf4",  # Set quantization type
)
# model optimization: Lora configuration
from peft import LoraConfig

peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0,
    r=16,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj",
        "v_proj",
        "k_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
)
# loading the model llama 3 8b with quantization and lora configuration
from transformers import AutoModelForCausalLM

base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3-8B",
    quantization_config=quantization_config,
    device_map={"": 0},
)

print("Model loaded successfully")
# Tokenizer configuration
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
tokenizer.add_special_tokens({"pad_token": "<|reserved_special_token_250|>"})
base_model.config.pad_token_id = tokenizer.pad_token_id
# Prompt/Response Format setting
EOS_TOKEN = tokenizer.eos_token

def convert_to_alpaca_format(prompt, completion):
    alpaca_format_str = f"""Below is an prompt that describes a task. Write a completion that appropriately completes the request.\
    \n\n### prompt:\n{prompt}\n\n### Response:\n{completion}\
    """

    return alpaca_format_str

def prompt_formatting_func(examples):
    prompt = examples["prompt"]
    completion = examples["completion"]
    texts = []
    for prompt, completion in zip(prompt, completion):
        alpaca_formatted_str = convert_to_alpaca_format(prompt, completion) + EOS_TOKEN
        texts.append(alpaca_formatted_str)
    return {
        "text": texts,
    }
def convert_to_alpaca_format(prompt, completion):
    alpaca_format_str = f"""Below is an prompt that describes a task. Write a completion that appropriately completes the request.\
    \n\n### prompt:\n{prompt}\n\n### Response:\n{completion}\
    """

    return alpaca_format_str

convert_to_alpaca_format( "What is Halo Bulk Order Reception?", "It refers to receiving bulk order information from the Halo system. Used in SFaaS.HIMS.DM.HO01 and Kafka.DMHO01P01T02and using this function in the main page") 
# Q-A pair without input

from datasets import load_dataset

# Load dataset
dataset = load_dataset("stlee9048/Llama3.1", split="train")

# Shuffle dataset
dataset = dataset.shuffle(seed=42)

# Map dataset
mapped_dataset = dataset.map(prompt_formatting_func, batched=True)

# Split dataset into train and test
split_dataset = mapped_dataset.train_test_split(test_size=0.01, seed=42)

train_dataset = split_dataset["train"]
test_dataset = split_dataset["test"]

print(f"Train dataset size: {len(train_dataset)}")
print(f"Test dataset size: {len(test_dataset)}")
# include input data

# Dataset Load
from datasets import load_dataset

dataset = load_dataset("stlee9048/Llama3.1", split="train")

dataset = dataset.shuffle(seed=42)
no_input_dataset = dataset.filter(lambda example: example["input"] == "")
mapped_dataset = no_input_dataset.map(prompt_formatting_func, batched=True)
split_dataset = mapped_dataset.train_test_split(test_size=0.01, seed=42)

train_dataset = split_dataset["train"]
test_dataset = split_dataset["test"]

print(f"Train dataset size: {len(train_dataset)}")
print(f"Test dataset size: {len(test_dataset)}")
# Data Collator setting
from trl import DataCollatorForCompletionOnlyLM

data_collator_param = {}
response_template = "### Response:\n"
collator = DataCollatorForCompletionOnlyLM(
    response_template=response_template, tokenizer=tokenizer, mlm=False
)
data_collator_param["data_collator"] = collator
# local output dir setting
local_output_dir = "/content/Llama3-8B-SETBOX"
!mkdir {local_output_dir}
# tensorboard setting
%load_ext tensorboard
%tensorboard --logdir '{local_output_dir}/runs'
# Training setup
from trl import SFTTrainer
from transformers import TrainingArguments

training_arguments = TrainingArguments(
    output_dir=local_output_dir,
    report_to="tensorboard",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,
    warmup_steps=50,
    max_steps=100,
    eval_steps=10,
    save_steps=50,
    evaluation_strategy="steps",
    save_strategy="steps",
    learning_rate=1e-4,
    logging_steps=1,
    optim="adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="constant_with_warmup",
    seed=42,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": True},
)

trainer = SFTTrainer(
    model=base_model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=2048,
    dataset_num_proc=2,
    packing=False,
    args=training_arguments,
    **data_collator_param
)

import os

print(os.getcwd())
train_stats = trainer.train()
# push your train model to the HF // change your_model_name to your model name
trainer.push_to_hub("stlee9048/Llama3.1")
print(train_stats)
# TensorBoard setting
%load_ext tensorboard
%tensorboard --logdir '{local_output_dir}/runs'
# model test 

## compare base model and finetuning model
 
# token
# HF token setting
from huggingface_hub import notebook_login

notebook_login()
# Model lightweight: Quantization settings
from transformers import BitsAndBytesConfig
import torch

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)
# Fine tune model load
from transformers import AutoModelForCausalLM

fine_tuned_model_path = AutoModelForCausalLM.from_pretrained(
    "stlee9048/fine_tune_output",
    quantization_config=quantization_config,
    device_map={"": 0},
)
# base model load
base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3-8B",
    quantization_config=quantization_config,
    device_map={"": 0},
)
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model and tokenizer from Hugging Face
model_name = "meta-llama/Meta-Llama-3-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
export CUDA_LAUNCH_BLOCKING=1
# Prompt/Response Format related setting
EOS_TOKEN = tokenizer.eos_token

def convert_to_alpaca_format(instruction, response):
    alpaca_format_str = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.\
    \n\n### Instruction:\n{instruction}\n\n### Response:\n{response}\
    """

    return alpaca_format_str
def test_model(instruction_str, model):
    inputs = tokenizer(
        [
            convert_to_alpaca_format(
                instruction_str,
                "",
            )
        ],
        return_tensors="pt",
    ).to("cuda")
    outputs = model.generate(
        **inputs, max_new_tokens=128, use_cache=True, temperature=0.05, top_p=0.95
    )
    return tokenizer.batch_decode(outputs)[0]
questions = [
    "What information does the FACILITY_CLAS_APPLY_L table store?",
    "How to figure out the 1F issue on Setbox?",
]
answers_dict = { #"base_model_answers": [],
                "fine_tuned_model_answers": []}
for idx, question in enumerate(questions):
    print(f"Processing EXAMPLE {idx}==============")
    # base_model_output = test_model(question, base_model)
    # answers_dict["base_model_answers"].append(base_model_output)
    fine_tuned_model_output = test_model(question, fine_tuned_model_path)
    answers_dict["fine_tuned_model_answers"].append(fine_tuned_model_output)
def simple_format(text, width=120):
    return "\n".join(
        line[i : i + width]
        for line in text.split("\n")
        for i in range(0, len(line), width)
    )

for idx, question in enumerate(questions):
    print(f"EXAMPLE {idx}==============")
    print(f"Question: {question}")

    print("<<Base Model Answer>>")
    base_model_answer = answers_dict["base_model_answers"][idx].split("### Response:")[
        1
    ]
    print(simple_format(base_model_answer))
    print("---")
    print("<<Fine Tuning Model Answer>>")
    fine_tuned_model_answer = answers_dict["fine_tuned_model_answers"][idx].split(
        "### Response:"
    )[1]
    print(simple_format(fine_tuned_model_answer))
    print()
base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3-8B",
    quantization_config=quantization_config,
    device_map={"": 0},
)
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

adapter_model_name = "stlee9048/Llama3-8B-SETBOX"
peft_config = PeftConfig.from_pretrained(adapter_model_name)
model = PeftModel.from_pretrained(base_model, adapter_model_name)

# Merge LoRA weights into the base model
peft_config = peft_config.merge_and_unload()
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

# Example usage of model and tokenizer
input_text = "how to resolve the setbox rack issue in HMGICS ?"
inputs = tokenizer(input_text, return_tensors="pt")
outputs = model.generate(**inputs)
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(generated_text)