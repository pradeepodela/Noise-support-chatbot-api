from flask import Flask, request, jsonify, session
from flask_session import Session
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
import os
import whisper
from langchain_google_genai import ChatGoogleGenerativeAI
import requests
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from twilio.twiml.messaging_response import MessagingResponse
app = Flask(__name__)
app.config['SECRET_KEY'] = 'supersecretkey'
app.config['SESSION_TYPE'] = 'filesystem'
Session(app)
os.environ["GOOGLE_API_KEY"] = 'AIzaSyAddeZi7UI3D99cV1VXzKoJepXYcl6BUf8'



def get_vectorstore_from_url(url):
    if os.path.exists("store/un_sdg_chroma_cosine"):
        vector_store = Chroma(persist_directory="store/un_sdg_chroma_cosine", embedding_function=SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2"))
        return vector_store
    else: 
        loader = WebBaseLoader(url)
        document = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter()
        document_chunks = text_splitter.split_documents(document)
        
        vector_store = Chroma.from_documents(document_chunks, SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2"), collection_metadata={"hnsw:space": "cosine"}, persist_directory="store/un_sdg_chroma_cosine")

        return vector_store

def transcribe_audio_from_url(url):
    print(f"Transcribing audio from URL: {url}")
    try:
        response = requests.get(url)
        response.raise_for_status()
        with open("audio.mp3", 'wb') as f:
            f.write(response.content)
        print("Audio file downloaded successfully.")

        model = whisper.load_model("base")
        result = model.transcribe("audio.mp3")
        os.remove("audio.mp3")
        print("Audio file transcribed successfully.")
        return result["text"]
    except Exception as e:
        print(f"Error transcribing audio: {e}")
        return f"An error occurred: {e}"

def get_context_retriever_chain(vector_store):
    # llm = ChatGroq(groq_api_key='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', model_name="llama3-70b-8192")
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest")

    retriever = vector_store.as_retriever()
    
    prompt = ChatPromptTemplate.from_messages([
      MessagesPlaceholder(variable_name="chat_history"),
      ("user", "{input}"),
      ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])
    
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    
    return retriever_chain

def get_conversational_rag_chain(retriever_chain): 
    # llm = ChatGroq(groq_api_key='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', model_name="llama3-70b-8192")
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest")

    prompt = ChatPromptTemplate.from_messages([
      ("system", "act as a senior customer care executive and help users sort out their queries related to the Noise company. Be polite and friendly. Answer the user's questions based on the below context:\n\n{context} make sure to provide all the details. If the answer is not in the provided context just say, 'answer is not available in the context', don't provide the wrong answer. Make sure if the person asks for any external recommendation only provide information related to the Noise company. If the user asks you anything other than Noise company just say 'sorry I can't help you with that'"),
      MessagesPlaceholder(variable_name="chat_history"),
      ("user", "{input}"),
    ])
    
    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)
    
    return create_retrieval_chain(retriever_chain, stuff_documents_chain)

def get_response(user_input, vector_store):
    if "chat_history" not in session:
        session["chat_history"] = [
            {"role": "assistant", "content": "Hello, I am a bot. How can I help you?"}
        ]
    
    retriever_chain = get_context_retriever_chain(vector_store)
    conversation_rag_chain = get_conversational_rag_chain(retriever_chain)
    
    response = conversation_rag_chain.invoke({
        "chat_history": [AIMessage(**msg) if msg["role"] == "assistant" else HumanMessage(**msg) for msg in session["chat_history"]],
        "input": user_input
    })
    
    return response['answer']

@app.route('/chat', methods=['POST'])
def chat():
    data = request.form.to_dict()
    print("Request Data: ", data)  # Print the entire request data
    user_query = data.get('Body', '')  # Use .get to avoid KeyError and provide a default value
    print("User Query: ", user_query)

    audio_text = ''  # Initialize audio_text to ensure it's defined
    if 'MediaUrl0' in data:
        print("Media URL found in request.")
        audio_text = transcribe_audio_from_url(data['MediaUrl0'])
        print("Transcribed Audio Text: ", audio_text)
    else:
        print("No Media URL found in request.")

    # if not user_query:
    #     print("No user query provided.")  # Add a print statement for debugging
    #     return jsonify({"error": "No query provided"}), 400

    if not os.path.exists("store/un_sdg_chroma_cosine"):
        vector_store = get_vectorstore_from_url('https://support.gonoise.com/support/home')
    else:
        vector_store = Chroma(persist_directory="store/un_sdg_chroma_cosine", embedding_function=SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2"))


    if 'MediaUrl0' in data:
        q = audio_text
    else:
        q = user_query
    response = get_response(q, vector_store)
    if not session.get("chat_history"):  # Initialize chat_history in session if it doesn't exist
        session["chat_history"] = []
    session["chat_history"].append({"role": "user", "content": q})
    session["chat_history"].append({"role": "assistant", "content": response})
    print("BOT Response: ", response)
    # nm = audio_text
    # send_message(data['From'], response)
    ot_resp = MessagingResponse()
    msg = ot_resp.message(response)
    # msg.body(response)
    return str(ot_resp)



