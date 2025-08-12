from openai import OpenAI

client = OpenAI()

batch_job = client.batches.retrieve("batch_689b6aa509ec8190bd6d681ce14df250")
print(batch_job)