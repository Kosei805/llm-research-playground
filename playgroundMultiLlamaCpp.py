from llama_cpp import Llama
from transformers import AutoTokenizer

model_path = 'models/DataPilot-ArrowPro-7B-KUJIRA-IQ4_NL.gguf'

model_id='mistralai/Mistral-7B-Instruct-v0.2'
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

llm = Llama(
  model_path=model_path,
  n_gpu_layers=-1,
)

sys_msg = "あなたは日本語を話す優秀なアシスタントです。回答には必ず日本語で答えてください。"

messages = [
  {"role": "system", "content": sys_msg},
]


while True:
  user_query = input(':')
  if "++stop" in user_query:
    break
  messages.append({"role": "user", "content": user_query})
  prompt = tokenizer.apply_chat_template(messages, tokenize=False)
  output = llm(
    prompt=prompt,
    max_tokens=512,
    stop=["[INST]","\n"],
    echo=False,
  )
  print(output['choices'][0]['text'])
  messages.append({"role":" assistant", "content": output['choices'][0]['text']})