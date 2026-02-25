
import joblib
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import requests


df = joblib.load('embeddings.joblib')

def create_embedding(text_list):
    r = requests.post("http://localhost:11434/api/embed", json= {
        "model" : "bge-m3",
        "input" : text_list
    })

    embedding = r.json()["embeddings"]
    return embedding

def inference(prompt):
    r = requests.post("http://localhost:11434/api/generate", json= {
        #"model" : "deepseek-r1:latest",
        "model" : "llama3.2",
        "prompt" : prompt,
        "stream" : False
    })

    response = r.json()
    print(response)
    return response



incoming_query = input("Enter your Question:")
question_embedding = create_embedding([incoming_query])[0]


similarities = cosine_similarity(np.vstack(df['embedding']), [question_embedding]).flatten()
#print(similarities)
top_results = 3
max_indx = similarities.argsort()[::-1][0:top_results]
#print(max_indx)
new_df = df.loc[max_indx]
#print(new_df[['number','title','text']])

prompt = f''' I am learning web development using Sigma Web development course.Here are video containing video title, video number, start time in seconds, end time in seconds, the text at that time:

{new_df[['title','number','start','end','text']].to_json(orient="records")}

------------------------------------------------------
"{incoming_query}"

user asked this question related to the video chunks, you have to answer in a human way (dont mention in above foramt) where and how much content is taught in which video (in which video and timestamp) and guide the user to go to that particular video, If user asks unrelated questions, tell him that you can only answer questions related to the course.

'''

with open("prompt.txt","w") as f:
    f.write(prompt)

response = inference(prompt)["response"]
print(response)

with open("response.txt","w") as f:
    f.write(response)


#for index,item in new_df.iterrows():
#    print(index,item["title"],item["number"],item["text"],item["start"],item["end"])
