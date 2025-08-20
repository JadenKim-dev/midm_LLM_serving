import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

model_name = "K-intelligence/Midm-2.0-Mini-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
generation_config = GenerationConfig.from_pretrained(model_name)

prompt = "트랜스포머 모델의 작동 원리를 설명해줘"

messages = [
    {"role": "system", 
     "content": "Mi:dm(믿:음)은 KT에서 개발한 AI 기반 어시스턴트이다."},
    {"role": "user", "content": prompt}
]

input_ids = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt"
)

output = model.generate(
    input_ids.to("cuda"),
    generation_config=generation_config,
    eos_token_id=tokenizer.eos_token_id,
    max_new_tokens=128,
    do_sample=False,
)
print(tokenizer.decode(output[0]))