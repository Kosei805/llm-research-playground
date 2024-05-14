from llama_cpp import Llama

# model_path = 'models/DataPilot-ArrowPro-7B-KUJIRA-IQ4_NL.gguf'
# model_path = 'DataPilot-ArrowPro-7B-KUJIRA-Q4_K_S.gguf'
model_path = 'DataPilot-ArrowPro-7B-KUJIRA-Q4_K_M.gguf'
# model_path = 'DataPilot-ArrowPro-7B-KUJIRA-Q8_0.gguf'

llm = Llama(
  model_path=model_path,
  n_gpu_layers=-1,
)

sys_msg = "あなたは日本語を話す優秀なアシスタントです。回答には必ず日本語で答えてください。"

history = ""



user_query = input(':')
prompt = """[INST] <<SYS>>
{}
<</SYS>>

{}[/INST]
""".format(sys_msg,user_query)

output = llm(
  prompt=prompt,
  max_tokens=512,
  stop=["[INST]","\n"],
  echo=False,
)

print(output['choices'][0]['text'])