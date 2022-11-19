import torch
import whisper

class Transcriber:
  def __init__(self):
    torch.set_default_tensor_type('torch.cuda.FloatTensor') 
    self.model = whisper.load_model('base')

  def transcribe(self, filename):
    result = self.model.transcribe(filename) 
    #print(result["text"][:240])
    
    file = open(filename + ".txt", "w", encoding="utf-8") 
    file.write(result["text"])
    file.close()