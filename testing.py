
import openai
from dotenv import load_dotenv

load_dotenv()
openai.api_key = "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"

response = openai.Completion.create(
  model="text-davinci-003",
  prompt='Hi, how are you',
  temperature=0.6
)

print(response['choices'][0]['text'])