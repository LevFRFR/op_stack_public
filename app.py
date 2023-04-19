import os, openai, pinecone, time
from dotenv import load_dotenv
import streamlit as st
# from langchain.vectorstores import Pinecone
# from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain.chains.qa_with_sources import load_qa_with_sources_chain
# from langchain.chat_models import ChatOpenAI
# from langchain.prompts import PromptTemplate


DEBUG=False
EMBEDDING_MODEL = "text-embedding-ada-002"
CHAT_MODEL = "gpt-3.5-turbo"

# defining a prompt and a prompt template
PROMPT_TEMPLATE = '''You are a helpful AI assistant. Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say you don't know. DO NOT try to make up an answer.
If the question is not related to the context, politely respond that you are tuned to only answer questions that are related to the context.
Do not round numbers, only return the exact value.

{context}

Question: {question}
Helpful answer:'''

# Initializations
load_dotenv('.env')

openai.api_key = os.getenv("OPENAI_API_KEY")

if 'num_times_inited' not in st.session_state:
    st.session_state['num_times_inited'] = 0

@st.cache_resource()
def init_pinecone():
    pinecone.init(
        api_key=os.getenv('PINECONE_API_KEY'),
        environment=os.getenv('PINECONE_ENV')  # find next to API key in console
    )

    if os.getenv('PINECONE_INDEX') is not None and os.getenv('PINECONE_INDEX').strip():
        return pinecone.Index(os.getenv('PINECONE_INDEX'))
    else:
        raise SystemError("Couldn't find `.env` file.")

pinecone_index = init_pinecone()

def request_chat(context, question, MODEL=CHAT_MODEL, role="user"):
    completion = openai.ChatCompletion.create(
    model=MODEL,
    messages=[
        {"role": role, "content": PROMPT_TEMPLATE.format(context=context, question=question)}
    ],
    temperature=0
    )

    return completion

def retrieve_answer(query):
    retrieval_start = time.time()

    similarity_start = time.time()

    # search for similar documents
    xq = openai.Embedding.create(input=query, engine="text-embedding-ada-002")['data'][0]['embedding']
    res = pinecone_index.query([xq], top_k=4, include_metadata=True, namespace='suncor')

    similarity_end = time.time()

    # building context string
    context = ''

    for match in res['matches']:
        context='\n'.join([context, match.get('metadata').get('text')])

    # request chat
    answer = request_chat(context=context, question=query)

    retrieval_end = time.time()

    if DEBUG:
        return answer["choices"][0]["message"]["content"], retrieval_end - retrieval_start, similarity_end - similarity_start

    return answer["choices"][0]["message"]["content"]


# UI
st.title('SuncorGPT')

with st.sidebar:
    st.title("About")
    st.write("This is a demonstration of Project SuncorGPT, a ChatGPT-like interface to ask questions about information related to Suncor.\
             \nCurrently supports the Annual, Climate, and Sustainability Reports.")
    st.header("Instructions")
    st.write("Enter a question in the text box and hit enter. For example, you can ask: ")
    st.markdown(">\"What is SAGD?\"")
    st.markdown(">\"How many people are on the Suncor and Syncrude workforce combined?\"")

query = st.text_input("Prompt Input", value="", type='default', key=None, help="Enter a question here", label_visibility="hidden", on_change=None, args=None, kwargs=None)

if query:
    # wall-clock performance evaluation
    if DEBUG:
        full_start = time.time()
        with st.spinner("Loading..."):
            answer, retrieval_time, similarity_time, text_gen_time = retrieve_answer(query)
        full_end = time.time()

        # if query not none
        st.write(answer['output_text'])

        # perf metrics
        st.write("Start to finish time: " + str(full_end - full_start))
        st.write("similarity time: " + str(similarity_time))
        st.write("text gen time: " + str(text_gen_time))
        st.write("retrieval_time time: " + str(retrieval_time))

        st.write(f"initialization invoked {st.session_state['num_times_inited']} times\n")

    else:
        with st.spinner("Loading..."):
            answer = retrieve_answer(query)
        st.success(answer)