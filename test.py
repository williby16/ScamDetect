from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.llms import Ollama
import streamlit as st
import tensorflow as tf


# Initialize the Text Classification Model
tokenizer = AutoTokenizer.from_pretrained("pippinnie/scam_text_classifier")
model = TFAutoModelForSequenceClassification.from_pretrained("pippinnie/scam_text_classifier")

# Initialize the Ollama model
llm = Ollama(model="llama3")

# Define a prompt template for the chatbot
prompt_format = '''[<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a scam text expert who explains things in a clear and concise manner that is easy to understand.
The first input from the user is a text they are suspicious of being a scam and is sent for a text classifier who provides a label, either scam or safe and the likelihood.
You love teaching the user to understand why a text is likely or unlikely to be a scam, and teach them how to identify such texts.

Analyze the provided text based on the label and the likelihood, and present your analysis in the following format:

| Label | Possibility |
|---|---|
| {label} | {likelihood:.1%} |

Explanation

Answer any follow-up questions the user may have based on the following chat history.
If they don't have any questions, remind them to open a new chat room for another suspicious text.
{chat_history}
<|eot_id|><|start_header_id|>user<|end_header_id|>
{input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>'''

prompt = ChatPromptTemplate.from_template(prompt_format)

# Create a chain that combines the prompt and the Ollama model
chain = prompt|llm

def clasify_scam(text):
    inputs = tokenizer(text, return_tensors="tf")
    logits = model(**inputs).logits

    predicted_class_id = int(tf.math.argmax(logits, axis=-1)[0])
    label = model.config.id2label[predicted_class_id]

    # Apply softmax to get probabilities
    probabilities = tf.nn.softmax(logits, axis=-1)

    # Get the probability of the label
    likelihood = probabilities[0][predicted_class_id].numpy()

    return (label, likelihood)

# format ["scam or legit", """message""", [empty list for results]]
potential_scams = [["SCAM", """The USPS package has arrived at the warehouse and cannot be delivered due to incomplete address information. Please confirm your address in the link within 12 hours.

https://usps(.)postcarexpress(.)com

(Please reply to Y, then exit the SMS, open the SMS activation link again, or copy the link to Safari browser and open it)

The US Postal team wishes you a wonderful day""", []],
["SCAM", """Hi, I'm Mia with Indeed and we have remote online part-time/full-time jobs! Your background and resume have caught the attention of several online recruiters, so we would like to offer you a job that you can do from home in your free time. This job is easy and has no time constraints. Daily pay ranges from $200 to $5,000 and is paid on the same day.
Join us and be a part of America's booming job market and start a career you can be proud of. (Age Requirement: Adult) If interested, please contact us via WhatsApp at +12022704251""", []],
["SCAM", """Texas Tolls Services, our records show that your vehicle has an outstanding toll charge. To prevent further fees totaling $117.50, please settle the due amount of $11.75 at rmatollservices(.)com""", []],
["SCAM", """Hello, my name is Julia, I got your contact information from the job market. We noticed that your resume is very good, so we would like to offer you a part-time/full-time job that you can do in your free time. The work is simple, can be done from home, and there is no time limit. You only need a smartphone or computer to do the work. The daily salary ranges from $220 to $965, all paid on the same day. If you are interested, please contact the employer via WhatsApp +16462866312

(Requirements: 22 years old and above)""", []],
["SCAM", """The USPS package has arrived at the warehouse. Due to the incorrect address information, it cannot be delivered to you in time. Please click the link to fill in your correct address information.

https://usps(.)alus-packages(.)com

(Please reply Y, then exit the text message and open it again to activate the link, or copy the link and open it in your Safari browser).

The USPS team wishes you a wonderful day!""", []]
 ]

for i in potential_scams:
    i[2] = clasify_scam(i[1])
    
wrong = 0
right = 0
wrongPercent = 0
rightPercent = 0
for i in potential_scams:
    if (i[0] != i[2][0]):
        wrong += 1
        wrongPercent += i[2][1]
    else:
        right += 1
        rightPercent += i[2][1]

# the accuraccy of certainty the bot provides
wrongPercent /= wrong
rightPercent /= right

print("Bot Accuracy: ", (right/(right+wrong))*100)
print("Bot Certinty: ")
print("Right: ", rightPercent*100)
print("Wrong: ", wrongPercent*100)


