import ollama

def query(task: str):
    response = ollama.chat(model="gemma:2b", messages=[
    {
        'role': 'user',
        'content': task,
    },
    ])
    print(f"ollama took {response['eval_duration'] }nano seconds, {response['eval_duration']/1000000000} seconds")
    return response["message"]["content"]


with open("./06042024v1_atomic_actiontest.txt", 'r', encoding='utf-8') as file:
    task1 = file.read()
    print(task1)

for i in range(10):
    query(task1)