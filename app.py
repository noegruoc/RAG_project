import pysqlite3
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import sqlite3

import streamlit as st
from streamlit_chat import message
from langchain_community.document_loaders import PyPDFLoader  # Use PDFLoader instead of TextLoader
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_mistralai.embeddings import MistralAIEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import Chroma  # Using Chroma for vector storage
import warnings
import base64

warnings.filterwarnings("ignore")
api_key = st.secrets["MISTRAL_API_KEY"]

#mots sensibles
sensitive = ["poem", "verse", "rhyme", "haiku", "song", "lyric", "stanza", "artistic", "prose", "poème", "quote", "citation",
    # Political Sensitivities
    "corruption", "scandal", "controversy", "protest", "criticism", 
    "allegation", "accountability", "misconduct", "censorship", "opposition",

    # Financial Concerns
    "fraud", "mismanagement", "bankruptcy", "misuse", 
    "embezzlement", "tax evasion", "financial instability", "audit issues",

    # Reputation and Public Image
    "failure", "negative reviews", "complaints", "boycott", 
    "unethical", "reputation damage", "scorn", "backlash",

    # Competitors
    "competitors", "rivalry", "market share", "industry competition",
    "brand war", "market dominance",

    # Legal Issues
    "lawsuit", "penalty", "fine", "legal dispute", "non-compliance", 
    "illegal", "regulatory breach", "sanctions",

    # Cultural or Social Sensitivities
    "discrimination", "inequality", "controversial policies", 
    "public outrage", "racism", "sexism", "harassment",

    # Other Sensitive Topics
    "environmental harm", "climate irresponsibility", "human rights violation", 
    "child labor", "exploitation", "labor issues", "social injustice",

    #Promp injections 
    "ignore", "forget", "bypass", "override", "disregard", "disobey", "ignore previous instructions", 
     "cancel previous commands", "ignore system instructions", "ignore the rules", "don't follow the instructions", 
     "change the behavior", "do not follow", "oublie", "contourne", "outrepasse", "ne tiens pas compte", 
     "ne suis pas les instructions précédentes", "annule les instructions précédentes", "ignore les règles", 
     "ne suis pas les commandes", "change le comportement", "ne suis pas les consignes"
            ]

#directory of db
persist_directory = "./chroma_db"
#persist_directory = r"C:\Users\courg\chroma_db"

# Initialize embeddings and vector store
embeddings = MistralAIEmbeddings(model="mistral-embed", mistral_api_key=api_key) #mistral model for embeddings
vector = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
retriever = vector.as_retriever()

# Create the retrieval chain

#Répond en français à la question suivante basé sur le contexte donné de BNP Paribas.
model = ChatMistralAI(mistral_api_key=api_key, temperature=0) #Mistral model 7b
prompt = ChatPromptTemplate.from_template("""    

    Vous êtes un assistant français spécialisé dans les résultats financiers de la banque BNP Paribas et dans les transcriptions de leurs conférences de presse. 
    Ne répondez qu'aux questions portant sur ces sujets uniquement en langue française. Si la question est hors sujet ou concerne un autre domaine, répondez que vous ne pouvez pas répondre en français.
    Si la question est posée dans une autre langue, répondez en français.

    <context>
    {context}
    </context>

    Question: {input}
    """)
document_chain = create_stuff_documents_chain(model, prompt)
retrieval_chain = create_retrieval_chain(retriever, document_chain)

#########################################################################################################
def load_chain(user_input):

    sensitive_set = set(sensitive)  # Convert sensitive keywords to a set

    def sanitize_input(input_text):
        if any(keyword in input_text.lower() for keyword in sensitive_set):
            print(f"Detected sensitive content. Sanitizing input.")
            return True
        return input_text

    def process_user_input(user_input):
            if sanitize_input(user_input) is True:
                return f"Je ne suis pas capable de répondre à cette question."
            response = retrieval_chain.invoke({"input": user_input}, )
            return response["answer"]

    return process_user_input(user_input)

#########################################################################################################
# Streamlit UI
#########################################################################################################
st.set_page_config(page_title="Votre Assistant RAG sur les résultats financiers de BNP Paribas en 2024", page_icon=":robot_face:")

st.markdown("""
    # Votre Assistant RAG sur les résultats financiers de BNP Paribas <img src="data:image/png;base64,{}" width="50" style="vertical-align: -3px;">
""".format(base64.b64encode(open("bnp_logo.png", "rb").read()).decode()), unsafe_allow_html=True)

# Initialize chat states
if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []

with st.form(key="form"):
    user_input = st.text_input(
        label="You", 
        placeholder="Hello! Que voulez-vous savoir sur les résultats ou la dernière conférence de presse ?", 
        key="input"
    )
    submit_button_pressed = st.form_submit_button("Submit to RAG")

if submit_button_pressed:
    result = load_chain(user_input)
    st.session_state.past.append(user_input)
    st.session_state.generated.append(result)

if st.session_state["generated"]:
    for i in range(len(st.session_state["generated"]) - 1, -1, -1):
        message(st.session_state["generated"][i], key=str(i), avatar_style="bottts", seed=129)
        message(st.session_state["past"][i], is_user=True, key=f"{i}_user", avatar_style="fun-emoji", seed=2281)
