
from transformers import AutoTokenizer


model_id='DataPilot/ArrowPro-7B-KUJIRA'
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

messages = [
    {"role": "system", "content": "**設定**"},
    {"role": "user", "content": "**質問**"},
    {"role": "assistant", "content": "**回答**"},
    {"role": "user", "content": "**質問2**"},
]  # NOTE:: phi-3は*デフォルトでは*`system`ロールに対応しない

prompt = tokenizer.apply_chat_template(messages, tokenize=False)
print(prompt)
