import os
import logging
from flask import Flask, request, jsonify, session
from flask_session import Session
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
import requests
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from twilio.twiml.messaging_response import MessagingResponse
from twilio.rest import Client
from langdetect import detect, detect_langs
from googletrans import Translator
import whisper

# langlist = ['hi','te','ta','ml','kn','mr','bn','gu','pa','or','as','sd','ur']

# Configuration
app = Flask(__name__)
app.config['SECRET_KEY'] = 'supersecretkey' #os.getenv('FLASK_SECRET_KEY', 'supersecretkey')
app.config['SESSION_TYPE'] = 'filesystem'
app.config["CACHE_TYPE"] = "null"
Session(app)

# Set up logging
logging.basicConfig(level=logging.INFO)

# Environment variables
os.environ["GOOGLE_API_KEY"]  = ''
ACCOUNT_SID =''
AUTH_TOKEN = ''
TWILIO_PHONE_NUMBER = os.getenv('TWILIO_PHONE_NUMBER', 'whatsapp:+14155238886')

client = Client(ACCOUNT_SID, AUTH_TOKEN)

# Helper functions
def translate_to_english(text):
    translator = Translator()
    try:
        translated = translator.translate(text, dest='en')
        return translated.text
    except Exception as e:
        logging.error(f"Error translating text to English: {e}")
        return str(e)

def translate_to_language(text, language):
    translator = Translator()
    try:
        translated = translator.translate(text, dest=language)
        return translated.text
    except Exception as e:
        logging.error(f"Error translating text to {language}: {e}")
        return str(e)

def detect_language(text):
    try:
        language = detect(text)
        probabilities = detect_langs(text)
        return language, probabilities
    except Exception as e:
        logging.error(f"Error detecting language: {e}")
        return None, str(e)

def send_message(to, message):
    try:
        client.messages.create(
            from_=TWILIO_PHONE_NUMBER,
            body=message,
            to=to
        )
    except Exception as e:
        logging.error(f"Error sending message: {e}")

def get_vectorstore_from_url(url):
    if os.path.exists("store/un_sdg_chroma_cosine"):
        vector_store = Chroma(persist_directory="store/un_sdg_chroma_cosine", embedding_function=SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2"))
    else:
        loader = WebBaseLoader(url)
        document = loader.load()
        text_splitter = RecursiveCharacterTextSplitter()
        document_chunks = text_splitter.split_documents(document)
        vector_store = Chroma.from_documents(document_chunks, SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2"), collection_metadata={"hnsw:space": "cosine"}, persist_directory="store/un_sdg_chroma_cosine")
    return vector_store

def transcribe_audio_from_url(url):
    logging.info(f"Transcribing audio from URL: {url}")
    try:
        response = requests.get(url)
        response.raise_for_status()
        with open("audio.mp3", 'wb') as f:
            f.write(response.content)
        logging.info("Audio file downloaded successfully.")
        model = whisper.load_model("base")
        result = model.transcribe("audio.mp3")
        os.remove("audio.mp3")
        logging.info("Audio file transcribed successfully.")
        return result["text"]
    except Exception as e:
        logging.error(f"Error transcribing audio: {e}")
        return f"An error occurred: {e}"

def get_context_retriever_chain(vector_store):
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest")
    retriever = vector_store.as_retriever()
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])
    return create_history_aware_retriever(llm, retriever, prompt)

def get_conversational_rag_chain(retriever_chain):
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest")
    prompt = ChatPromptTemplate.from_messages([
        ("system", "act as a senior customer care executive and help users sort out their queries related to the Noise company. Be polite and friendly. Answer the user's questions based on the below context:\n\n{context} make sure to provide all the details. If the answer is not in the provided context just say, 'answer is not available in the context', don't provide the wrong answer. Make sure if the person asks for any external recommendation only provide information related to the Noise company. If the user asks you anything other than Noise company just say 'sorry I can't help you with that'"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}")
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

# Routes
@app.route('/')
def index():
    return "App is running!"

@app.route('/chat', methods=['POST'])
def chat():
    data = request.form.to_dict()
    logging.info(f"Request Data: {data}")
    user_query = data.get('Body', '')
    audio_text = ''
    if 'MediaUrl0' in data:
        logging.info("Media URL found in request.")
        audio_text = transcribe_audio_from_url(data['MediaUrl0'])
        logging.info(f"Transcribed Audio Text: {audio_text}")

    vector_store = get_vectorstore_from_url('https://www.zohoschools.com/')
    q = audio_text if 'MediaUrl0' in data else user_query
    language, _ = detect_language(q)
    # if language is 'so' or language is 'sw' or language is 'yo' or language is 'fr' or language is 'fi':
    #     language = 'en'
    # logging.info(f"Detected Language: {language}")

    if language != 'en':
        logging.info("Translating to English...")
        q = translate_to_english(q)
        logging.info(f"Translated Query: {q}")

    response = get_response(q, vector_store)
    print('Language: after response ', language)
    logging.info(f"Response: {response}")
    if language != 'en':
        print('Language: inside if ', language)
        response = translate_to_language(response, language)
        logging.info(f"Translated Response: {response}")
   


    session.setdefault("chat_history", []).append({"role": "user", "content": q})
    session["chat_history"].append({"role": "assistant", "content": response})

    send_message(data['From'], response)
    twilio_resp = MessagingResponse()
    twilio_resp.message('')
    return str(twilio_resp)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
