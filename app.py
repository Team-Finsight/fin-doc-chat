import streamlit as st
from streamlit_chat import message
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import CTransformers
from langchain.llms import Replicate
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import TextLoader
from langchain.document_loaders import Docx2txtLoader
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import os
from dotenv import load_dotenv
import tempfile
import logging
import sys
from langchain.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import transformers



def initialize_session_state():
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Hello! Ask me anything about the document, i am here to serve🫡"]

    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hello our Best client🌟😀"]

def conversation_chat(query, chain, history):
    result = chain({"question": query, "chat_history": history})
    history.append((query, result["answer"]))
    return result["answer"]

def display_chat_history(chain):
    reply_container = st.container()
    container = st.container()

    with container:
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_input("Question:", placeholder="Ask about your Documents", key='input')
            submit_button = st.form_submit_button(label='Send')

        if submit_button and user_input:
            with st.spinner('Generating response...'):
                output = conversation_chat(user_input, chain, st.session_state['history'])

            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)

    if st.session_state['generated']:
        with reply_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="thumbs")
                message(st.session_state["generated"][i], key=str(i), avatar_style="fun-emoji")

def create_conversational_chain():
    from llama_index.prompts.prompts import SimpleInputPrompt


    system_prompt = "You are a Q&A assistant for income tax. Your goal is to answer questions only related to the document as accurately as possible based on the instructions and context provided."



    # This will wrap the default prompts that are internal to llama-index
    query_wrapper_prompt = SimpleInputPrompt("<|USER|>{query_str}<|ASSISTANT|>")    
    from huggingface_hub import login
    login(token="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")
    model_id = 'meta-llama/Llama-2-13b-chat-hf'

    # begin initializing HF items, need auth token for these
    hf_auth = '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'
    model_config = transformers.AutoConfig.from_pretrained(
        model_id,
        use_auth_token=hf_auth
    )
    # initialize the model
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        config=model_config,
        device_map='auto',
        use_auth_token=hf_auth
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_id,
        use_auth_token=hf_auth
    )
    generate_text = transformers.pipeline(
        model=model, tokenizer=tokenizer,
        return_full_text=True,  # langchain expects the full text
        task='text-generation',
        # we pass model parameters here too
        temperature=0.1,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
        max_new_tokens=512,  # mex number of tokens to generate in the output
        repetition_penalty=1.1  # without this output begins repeating
    )

    from langchain.llms import HuggingFacePipeline

    llm = HuggingFacePipeline(pipeline=generate_text)
    from langchain.memory import ConversationBufferWindowMemory
    from langchain.agents import load_tools

    import re
    import json
    from typing import Any  # Import the Any type

    FORMAT_INSTRUCTIONS = """Assistant is a large language model trained by OpenAI.

    Assistant is designed to support a wide range of tasks, from answering simple questions to providing detailed explanations and discussions on a wide range of topics. As a language model, Assistant can generate human-like text based on input received, and can provide natural-sounding conversation or consistent, on-topic responses.

    Assistant is constantly learning and improving, and its capabilities are always evolving. It can process vast amounts of text to understand and provide accurate and helpful answers to a variety of questions. Additionally, Assistant can generate its own text based on received input, allowing it to participate in discussions on a variety of topics, or provide explanations and commentary.

    Overall, Assistant is a powerful tool that can support a variety of tasks and provide valuable insights and information on a wide range of topics. Whether you need help with a specific question or want to have a conversation about a specific topic, Assistant is here to help.

    TOOLS:
    ------
    Assistant has access to the following tools."""
    suffix = """Answer the questions you know to the best of your knowledge.

    Begin!

    User Input: {input}
    {agent_scratchpad}"""  # You should define FORMAT_INSTRUCTIONS

    import logging
    import sys
    import llama_index
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

    from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
    from llama_index.llms import HuggingFaceLLM
    from langchain.embeddings.huggingface import HuggingFaceEmbeddings
    from llama_index import LangchainEmbedding, ServiceContext

    embed_model = LangchainEmbedding(
    HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    )
    service_context = ServiceContext.from_defaults(
        chunk_size=1024,
        llm=llm,
        embed_model=embed_model
    )

    documents = SimpleDirectoryReader("/workspace/Data").load_data()
    index = VectorStoreIndex.from_documents(documents, service_context=service_context)
    from llama_index.langchain_helpers.memory_wrapper import GPTIndexChatMemory



    # options here aren't well documented - you'll need to dig through the code
    # but I'll share some links below
    memory = GPTIndexChatMemory(
        # internally converts to query engine via `as_query_engine`
        index=index,
        # or whatever is appropriate for you here - defaults to "history"
        memory_key="chat_history",
        # return_source returns source nodes instead of querying index
        return_source=True,
        # return_messages returns context (message history) as an array
        # of messages as opposed to a concatenated string of all message content
        return_messages=True,
    )
    from langchain.agents import initialize_agent
    tools = load_tools(['ddg-search','llm-math'], llm)
    agent_chain = initialize_agent(
        tools, llm, agent="conversational-react-description", memory=memory,handle_parsing_errors=True
    )
    return agent_chain
def main():
    # Initialize session state
    initialize_session_state()
    st.title("FIN-DOC-AI :🏬📉:")
    # Initialize Streamlit
    st.sidebar.title("Document Processing")
    uploaded_files = st.sidebar.file_uploader("Upload files", accept_multiple_files=True)

    chain = create_conversational_chain()

    
    display_chat_history(chain)

if __name__ == "__main__":
    main()
