
from transformers import AutoTokenizer


model_id='mistralai/Mistral-7B-Instruct-v0.2'
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
# tokenizer.chat_template = '''{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'] %}{% elif true == true and not '<<SYS>>' in messages[0]['content'] %}{% set loop_messages = messages %}{% set system_message = 'You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\\n\\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don\\'t know the answer to a question, please don\\'t share false information.' %}{% else %}{% set loop_messages = messages %}{% set system_message = false %}{% endif %}{% for message in loop_messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if loop.index0 == 0 and system_message != false %}{% set content = '<<SYS>>\\n' + system_message + '\\n<</SYS>>\\n\\n' + message['content'] %}{% else %}{% set content = message['content'] %}{% endif %}{% if message['role'] == 'user' %}{{ bos_token + '[INST] ' + content.strip() + ' [/INST]' }}{% elif message['role'] == 'system' %}{{ '<<SYS>>\\n' + content.strip() + '\\n<</SYS>>\\n\\n' }}{% elif message['role'] == 'assistant' %}{{ ' '  + content.strip() + ' ' + eos_token }}{% endif %}{% endfor %}'''

# messages = [
#     {"role": "system", "content": "**設定**"},
#     {"role": "user", "content": "**質問**"},
#     {"role": "assistant", "content": "**回答**"},
#     {"role": "user", "content": "**質問2**"},
# ]  # NOTE:: phi-3は*デフォルトでは*`system`ロールに対応しない

messages = [
    {"role": "user", "content": "**質問**"},
    {"role": "assistant", "content": "**回答**"},
    {"role": "user", "content": "**質問2**"},
]  # NOTE:: phi-3は*デフォルトでは*`system`ロールに対応しない


prompt = tokenizer.apply_chat_template(messages, tokenize=False)
print(prompt)
