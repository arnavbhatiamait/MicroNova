# import streamlit as st 
# from langchain_ollama import ChatOllama,OllamaLLM
# from langchain_core.prompts import ChatPromptTemplate

# if "messages" not in st.session_state:
#     st.session_state.messages = []
# system_prompt="Analyze the meeting transcript and suggest better marketing strategy in order to obtain sales. analyze the customers interest and suggest the user for the best statergy to convert that lead and better marketing lines to please the customer, there is conversation between human and user where your role is to assist the user in order to be better at sales you will get human message, user message as inputs"
# llm=OllamaLLM(model="llama3.2")
# prompt=ChatPromptTemplate.from_messages([("system","{system_prompt}"),("human","{client_input}"),("user","{user_input}")])
# chain=prompt|llm

# for msg in st.session_state.messages:
#     with st.chat_message(msg["role"]):
#         st.write(msg["content"])
# client_input=st.chat_input("Give Client Message")
# user_input = st.chat_input("Give User messgae")
# if st.button("Generate"):
#     st.session_state.messages.append({"role": "client", "content": client_input})
#     st.session_state.messages.append({"role": "user", "content": user_input})
#     with st.chat_message("Client"):
#         st.write(client_input)
#     with st.chat_message("user"):
#         st.write(user_input)

#     response=chain.invoke(
#         {"system_prompt":system_prompt,
#         "client_input":client_input,
#         "user_input":user_input}
#         # prev_mess=st.session_state.messages
#     )
#     st.session_state.messages.append({"role": "assistant", "content": response})
#     with st.chat_message("assistant"):
#         st.write(response)




# # out=llm.invoke(prompt)
# # st.write(out)
import streamlit as st
from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
import os
from dotenv import load_dotenv
load_dotenv()
# add_logo()
st.set_page_config(page_title="Stride",page_icon="./Stride_logo.jpg")
# st.logo("./stride.png",size="large",icon_image="./stride.png")

st.image("./Stride_logo.jpg")
# st.markdown()
st.title("👨🏻‍💼 Stride 📈")
# Initialize LLM
# llm = ChatOllama(model="llama3.2")
try:
    api_key=os.getenv("GROQ_API_KEY")
    print(api_key)
except Exception as e:
    print("Exception ", e)
if api_key==None:
    st.text_input("Enter Groq API Key ")
llm=ChatGroq(model="llama-3.1-8b-instant",api_key=api_key)

# System prompt (fixed role instructions)
system_prompt = SystemMessage(
    content="Analyze the meeting transcript and suggest better marketing strategies to increase sales. Analyze the customer's interest and suggest how the user can convert the lead. Provide better marketing lines to please the customer. You are an assistant helping the user improve sales by analyzing the conversation between human and user."
)

# Chat prompt template with a placeholder for the message history
prompt = ChatPromptTemplate.from_messages(
    [
        system_prompt,
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{client_input}"),
        ("user", "{user_input}")
    ]
)

# Create chain
chain = prompt | llm

# Initialize session history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User inputs
# client_input = st.chat_input("Client (Customer) message:")
client_input = st.text_input("Client (Customer) message:")

user_input = st.text_input("Your (Salesperson) response:", key="user_input")
# if client_input:
if user_input and client_input and st.button("Analyze"):
    # Add current messages to history
    st.session_state.messages.append({"role": "human", "content": client_input})
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Prepare chat history for prompt
    chat_history = []
    for m in st.session_state.messages:
        if m["role"] == "human":
            chat_history.append(HumanMessage(content=m["content"]))
        elif m["role"] == "user":
            chat_history.append(HumanMessage(content=f"[Salesperson] {m['content']}"))
        elif m["role"] == "assistant":
            chat_history.append(AIMessage(content=m["content"]))

    # Get response from LLM
    response = chain.invoke({
        "chat_history": chat_history,
        "client_input": client_input,
        "user_input": user_input
    })

    # Store and display assistant message
    st.session_state.messages.append({"role": "assistant", "content": response.content})
    with st.chat_message("assistant"):
        st.markdown(response.content)
