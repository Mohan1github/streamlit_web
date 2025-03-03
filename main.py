# import streamlit as st
# from langchain_core.messages import AIMessage, HumanMessage
# from langchain_community.document_loaders import WebBaseLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import Chroma
# from langchain_openai import OpenAIEmbeddings, ChatOpenAI
# from dotenv import load_dotenv   
# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langchain.chains import create_history_aware_retriever, create_retrieval_chain
# from langchain.chains.combine_documents import create_stuff_documents_chain

# # Load environment variables
# load_dotenv()

# # Streamlit page config
# st.set_page_config(page_title="RAG Website Summarizer", page_icon="🔍")
# st.title("Chat with Websites: RAG-based Summarizer")

# # Session state initialization
# if "chat_history" not in st.session_state:
#     st.session_state.chat_history = [
#         AIMessage(content="Hello, I am a bot. How can I help you?")
#     ]

# if "vector_db" not in st.session_state:
#     st.session_state.vector_db = None
#     st.session_state.last_url = None  # Track last entered URL

# # Function to get response from RAG system
# def get_response(user_query):
#     retriever_chain = get_context_retriever_chain(st.session_state.vector_db)
#     conversation_rag_chain = get_conversational_rag_chain(retriever_chain)
    
#     response = conversation_rag_chain.invoke({
#         "chat_history": st.session_state.chat_history,
#         "input": user_query
#     })
    
#     st.session_state.chat_history.append(HumanMessage(content=user_query))
#     st.session_state.chat_history.append(AIMessage(content=response["answer"]))

#     return response["answer"]

# # Function to vectorize website content
# def get_vectorizer_web_url(url):
#     loader = WebBaseLoader(url)
#     doc = loader.load()
#     text_splitter = RecursiveCharacterTextSplitter()
#     docs_chunk = text_splitter.split_documents(doc)
#     vector_db = Chroma.from_documents(docs_chunk, OpenAIEmbeddings())
#     return vector_db

# # Function to create the retriever chain
# def get_context_retriever_chain(vector_db):
#     llm = ChatOpenAI()
#     retriever = vector_db.as_retriever()

#     prompt = ChatPromptTemplate.from_messages([
#         MessagesPlaceholder(variable_name="chat_history"),
#         ("user", "{input}"),
#         ("system", "Based on the conversation history, generate a search query to find relevant information."),
#     ])

#     retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
#     return retriever_chain

# # Function to create the conversational RAG chain
# def get_conversational_rag_chain(retriever_chain):
#     llm = ChatOpenAI()

#     prompt = ChatPromptTemplate.from_messages([
#         ("system", "Answer the user's questions using the context below:\n\n{context}"),
#         MessagesPlaceholder(variable_name="chat_history"),
#         ("user", "{input}"),
#     ])

#     stuff_documents_chain = create_stuff_documents_chain(llm, prompt)
#     retrieval_chain = create_retrieval_chain(retriever_chain, stuff_documents_chain)

#     return retrieval_chain

# # Sidebar input for website URL
# with st.sidebar:
#     st.header("Settings")
#     web_url = st.text_input("Enter the website URL")

# # Process the input URL
# if web_url:
#     # Update vector store only if a new URL is entered
#     if web_url != st.session_state.last_url:
#         st.session_state.vector_db = get_vectorizer_web_url(web_url)
#         st.session_state.last_url = web_url  # Save the last entered URL

#     user_query = st.chat_input("Type your question...")

#     if user_query:
#         response = get_response(user_query)
#         st.write(response)

#     # Display chat history
#     for message in st.session_state.chat_history:
#         if isinstance(message, AIMessage):
#             with st.chat_message("AI"):
#                 st.write(message.content)
#         elif isinstance(message, HumanMessage):
#             with st.chat_message("Human"):
#                 st.write(message.content)









# import streamlit as st
# from langchain_community.document_loaders import WebBaseLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import Chroma
# from langchain_openai import OpenAIEmbeddings, ChatOpenAI
# from dotenv import load_dotenv
# from langchain.chains import ConversationalRetrievalChain
# from langchain.memory import ConversationBufferMemory
# from bs4 import BeautifulSoup
# import requests
# import os
# # Load environment variables (for OpenAI API key)
# load_dotenv()
# openai_api_key = os.getenv("OPENAI_API_KEY")


# if not openai_api_key:
#     st.error("⚠️ OpenAI API key not found. Please set it in the .env file.")


# embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
# llm = ChatOpenAI(model="gpt-4", temperature=0.5, openai_api_key=openai_api_key)
# # Streamlit app setup
# st.set_page_config(page_title="RAG Website Analyzer", page_icon="🌍")
# st.title("🔎 RAG-based Website Analyzer")

# # Initialize session state
# if "vector_db" not in st.session_state:
#     st.session_state.vector_db = None

# if "chat_history" not in st.session_state:
#     st.session_state.chat_history = []

# if "website_details" not in st.session_state:
#     st.session_state.website_details = {}

# # Function to scrape and extract website details
# def extract_website_details(url):
#     st.info("🔄 Extracting website details...")
    
#     response = requests.get(url, timeout=10)
#     if response.status_code != 200:
#         st.error("⚠️ Failed to fetch the website. Please check the URL.")
#         return None

#     soup = BeautifulSoup(response.text, "html.parser")

#     # Extract key details
#     title = soup.title.string if soup.title else "N/A"
#     meta_desc = soup.find("meta", attrs={"name": "description"})
#     meta_keywords = soup.find("meta", attrs={"name": "keywords"})

#     description = meta_desc["content"] if meta_desc else "N/A"
#     keywords = meta_keywords["content"] if meta_keywords else "N/A"

#     # Extract headings
#     headings = {
#         "H1": [h.text.strip() for h in soup.find_all("h1")],
#         "H2": [h.text.strip() for h in soup.find_all("h2")],
#         "H3": [h.text.strip() for h in soup.find_all("h3")],
#     }

#     # Extract main content (first 500 characters as a preview)
#     paragraphs = soup.find_all("p")
#     main_content = " ".join([p.text.strip() for p in paragraphs[:5]])[:500] + "..."

#     details = {
#         "Title": title,
#         "Description": description,
#         "Keywords": keywords,
#         "Headings": headings,
#         "Content Summary": main_content,
#     }

#     st.success("✅ Website details extracted successfully!")
#     return details

# # Function to scrape and vectorize website content
# def process_website(url):
#     st.info("🔄 Scraping and Processing Website...")

#     # Load website content
#     loader = WebBaseLoader(url)
#     documents = loader.load()

#     # Split text into chunks
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
#     docs_chunk = text_splitter.split_documents(documents)

#     # Convert to vectors and store in ChromaDB
#     # embeddings = OpenAIEmbeddings()
#     vector_db = Chroma.from_documents(docs_chunk, embeddings)

#     st.success("✅ Website processed successfully!")
#     return vector_db

# # Function to create the RAG retrieval chain
# def get_rag_chain(vector_db):
#     retriever = vector_db.as_retriever(search_kwargs={"k": 3})  # Retrieve top 3 relevant chunks
#     # llm = ChatOpenAI(model="gpt-4", temperature=0.5)

#     memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
#     rag_chain = ConversationalRetrievalChain.from_llm(
#         llm=llm, retriever=retriever, memory=memory
#     )
#     return rag_chain

# # Sidebar for website input
# with st.sidebar:
#     st.header("🔗 Website Input")
#     website_url = st.text_input("Enter Website URL")

#     if st.button("Process Website"):
#         if website_url:
#             # Extract and display website details
#             st.session_state.website_details = extract_website_details(website_url)

#             # Process website for RAG retrieval
#             st.session_state.vector_db = process_website(website_url)
#         else:
#             st.warning("⚠️ Please enter a website URL.")

# # Display extracted website details
# if st.session_state.website_details:
#     st.subheader("📄 Extracted Website Details")
#     st.write(f"**🔹 Title:** {st.session_state.website_details['Title']}")
#     st.write(f"**🔹 Description:** {st.session_state.website_details['Description']}")
#     st.write(f"**🔹 Keywords:** {st.session_state.website_details['Keywords']}")
    
#     # Display headings
#     st.write("**🔹 Headings:**")
#     for level, items in st.session_state.website_details["Headings"].items():
#         if items:
#             st.write(f"  **{level}**: {', '.join(items)}")

#     st.write("**🔹 Content Summary:**")
#     st.info(st.session_state.website_details["Content Summary"])

# # Chat interface
# if st.session_state.vector_db:
#     rag_chain = get_rag_chain(st.session_state.vector_db)
#     user_query = st.chat_input("💬 Ask about the website...")

#     if user_query:
#         response = rag_chain.run({"question": user_query, "chat_history": st.session_state.chat_history})
        
#         # Store chat history
#         st.session_state.chat_history.append(("User", user_query))
#         st.session_state.chat_history.append(("AI", response))

#         st.write("🧠 --AI Response:--")
#         st.write(response)

#     # Display chat history
#     for role, msg in st.session_state.chat_history:
#         with st.chat_message(role):
#             st.write(msg)


import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from bs4 import BeautifulSoup
import requests
import os

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    st.error("⚠️ OpenAI API key not found. Please set it in the .env file.")

embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
llm = ChatOpenAI(model="gpt-4", temperature=0.5, openai_api_key=openai_api_key)

# Streamlit app setup
st.set_page_config(page_title="RAG Website Analyzer", page_icon="🌍", layout="wide")
st.title("🔎 RAG-based Website Analyzer")

# Initialize session state
if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "website_details" not in st.session_state:
    st.session_state.website_details = {}

def extract_website_details(url):
    st.info("🔄 Extracting website details...")
    
    response = requests.get(url, timeout=10)
    if response.status_code != 200:
        st.error("⚠️ Failed to fetch the website. Please check the URL.")
        return None

    soup = BeautifulSoup(response.text, "html.parser")

    title = soup.title.string if soup.title else "N/A"
    meta_desc = soup.find("meta", attrs={"name": "description"})
    meta_keywords = soup.find("meta", attrs={"name": "keywords"})
    
    description = meta_desc["content"] if meta_desc else "N/A"
    keywords = meta_keywords["content"] if meta_keywords else "N/A"
    
    headings = {
        "H1": [h.text.strip() for h in soup.find_all("h1")],
        "H2": [h.text.strip() for h in soup.find_all("h2")],
        "H3": [h.text.strip() for h in soup.find_all("h3")],
    }

    paragraphs = soup.find_all("p")
    main_content = " ".join([p.text.strip() for p in paragraphs[:5]])[:500] + "..."
    
    # Extract images
    images = [img["src"] for img in soup.find_all("img") if "src" in img.attrs]
    
    # Extract links
    links = [a["href"] for a in soup.find_all("a", href=True)]
    
    # Count words
    word_count = len(soup.get_text().split())
    
    details = {
        "Title": title,
        "Description": description,
        "Keywords": keywords,
        "Headings": headings,
        "Content Summary": main_content,
        "Images": images,
        "Links": links,
        "Word Count": word_count,
    }

    st.success("✅ Website details extracted successfully!")
    return details

def process_website(url):
    st.info("🔄 Scraping and Processing Website...")
    loader = WebBaseLoader(url)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs_chunk = text_splitter.split_documents(documents)
    # vector_db = Chroma.from_documents(docs_chunk, embeddings)
    st.success("✅ Website processed successfully!")
    # return vector_db
    return docs_chunk

# def get_rag_chain(vector_db):
#     retriever = vector_db.as_retriever(search_kwargs={"k": 3})
#     memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
#     rag_chain = ConversationalRetrievalChain.from_llm(
#         llm=llm, retriever=retriever, memory=memory
#     )
#     return rag_chain

with st.sidebar:
    st.header("🔗 Website Input")
    website_url = st.text_input("Enter Website URL")
    
    if st.button("Process Website"):
        if website_url:
            st.session_state.website_details = extract_website_details(website_url)
            # st.session_state.vector_db = process_website(website_url)
        else:
            st.warning("⚠️ Please enter a website URL.")

if st.session_state.website_details:
    st.subheader("📄 Extracted Website Details")
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**🔹 Title:** {st.session_state.website_details['Title']}")
        st.write(f"**🔹 Description:** {st.session_state.website_details['Description']}")
        st.write(f"**🔹 Keywords:** {st.session_state.website_details['Keywords']}")
        st.write(f"**🔹 Word Count:** {st.session_state.website_details['Word Count']}")
    with col2:
        st.write("**🔹 Headings:**")
        for level, items in st.session_state.website_details["Headings"].items():
            if items:
                st.write(f"  **{level}**: {', '.join(items)}")
    
    with st.expander("📸 Extracted Images"):
        for img in st.session_state.website_details["Images"][:5]:
            st.image(img, width=300)
    
    with st.expander("🔗 Extracted Links"):
        for link in st.session_state.website_details["Links"][:10]:
            st.write(link)
    
    # Display tags for keywords
    st.write("### 🏷️ Extracted Tags")
    for tag in st.session_state.website_details["Keywords"].split(","):
        st.markdown(f"<span style='height:3rem;width:10rem;align-text:center;background-color:black; padding:5px 10px; border-radius:5px; margin-right:5px; display:inline-block; display:grid;'>{tag.strip()}</span>", unsafe_allow_html=True)

# if st.session_state.vector_db:
#     rag_chain = get_rag_chain(st.session_state.vector_db)
#     user_query = st.chat_input("💬 Ask about the website...")
    
#     if user_query:
#         response = rag_chain.run({"question": user_query, "chat_history": st.session_state.chat_history})
#         st.session_state.chat_history.append(("User", user_query)) if st.session():
#         st.session_state.chat_history.append(("AI", response))
#         st.write("🧠 **AI Response:**")
#         st.write(response)
    
#     for role, msg in st.session_state.chat_history:
#         with st.chat_message(role):
#             st.write(msg)
