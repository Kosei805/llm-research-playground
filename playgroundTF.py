import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class ChatModel():
  def __init__(self,modelPath,setting):
    self.tokenizer = AutoTokenizer.from_pretrained(modelPath)
    self.model = AutoModelForCausalLM.from_pretrained(
      modelPath,
      torch_dtype="auto",
    )
    self.model.eval()
    self.setting = setting


    if torch.cuda.is_available():
      model = model.to("cuda")
    
  def makePrompt(self,user_query):
    template = """[INST] <<SYS>>
{}
<</SYS>>

{}[/INST]"""
    return template.format(self.setting,user_query)
  
  def makeChat(self,user_query):
    prompt = self.makePrompt(user_query)
    input_ids = self.tokenizer.encode(
      prompt,
      add_special_tokens=True,
      return_tensors='pt'
    )
    tokens = self.model.generate(
    input_ids.to(device=self.model.device),
    max_new_tokens=500,
    temperature=1,
    top_p=0.95,
    do_sample=True,
    )
    out = self.tokenizer.decode(tokens[0][input_ids.shape[1]:],skip_special_tokens=True).strip()
    return out

setting_str = '''
あなたはYouTUberです。名前はHanaです。
'''
params = {
  'modelPath': 'DataPilot/ArrowPro-7B-KUJIRA',
  'setting' : setting_str,
}
chatModel = ChatModel(**params)
user_query = input(':')
print(chatModel.makeChat(user_query=user_query))