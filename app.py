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
    "child labor", "exploitation", "labor issues", "social injustice"]

#directory of db
persist_directory = "."
#persist_directory = r"C:\Users\courg\chroma_db"

# Initialize embeddings and vector store
embeddings = MistralAIEmbeddings(model="mistral-embed", mistral_api_key=api_key) #mistral model for embeddings
vector = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
retriever = vector.as_retriever()

# Create the retrieval chain
model = ChatMistralAI(mistral_api_key=api_key, temperature=0) #Mistral model 7b
prompt = ChatPromptTemplate.from_template("""    
    Répond en français à la question suivante basé sur le contexte donné de BNP Paribas.

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
    user_input = st.text_input("You", "Hello ! Que voulez vous savoir sur les résultats ou la dernière conférence de presse ?", key="input")
    submit_button_pressed = st.form_submit_button("Submit to RAG")

if submit_button_pressed:
    result = load_chain(user_input)
    st.session_state.past.append(user_input)
    st.session_state.generated.append(result)

if st.session_state["generated"]:
    for i in range(len(st.session_state["generated"]) - 1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state["past"][i], is_user=True, key=f"{i}_user")
