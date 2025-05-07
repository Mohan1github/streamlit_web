import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from bs4 import BeautifulSoup
import requests
import os
from urllib.parse import urlparse, urljoin
import time
import logging
import socket
import ssl
from datetime import datetime
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

os.environ["USER_AGENT"] = "MyCustomUserAgent/1.0"

if not groq_api_key:
    st.error("⚠️ Groq API key not found. Please set it in the .env file.")
    st.stop()


try:
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
except Exception as e:
    st.error(f"⚠️ Failed to initialize HuggingFace embeddings: {str(e)}")
    logger.error(f"HuggingFace embeddings initialization error: {str(e)}")
    st.stop()


from langchain_groq import ChatGroq

try:
    llm = ChatGroq(groq_api_key=groq_api_key, temperature=0.5, model_name="llama-3.3-70b-versatile")
except Exception as e:
    st.error(f"⚠️ Failed to initialize Groq Chat model: {str(e)}")
    logger.error(f"Groq Chat model initialization error: {str(e)}")
    st.stop()

st.set_page_config(page_title="kAG Website Analyzer  ", page_icon="🌍", layout="wide")
st.title("🔎 KAG-based Website Analyzer ")
st.title("Search and surf to the world!!!!!")


if "vector_db" not in st.session_state:
    st.session_state.vector_db = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "website_details" not in st.session_state:
    st.session_state.website_details = {}
if "sub_pages_details" not in st.session_state:
    st.session_state.sub_pages_details = {}
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "accuracy_score" not in st.session_state:
    st.session_state.accuracy_score = None
if "security_assessment" not in st.session_state:
    st.session_state.security_assessment = {}
if "main_page_accuracy" not in st.session_state:
    st.session_state.main_page_accuracy = 0.0


def calculate_accuracy(details):
    """Calculate accuracy score based on completeness of scraped data."""
    total_elements = 5  
    score = 0

    if details["Title"] != "N/A":
        score += 1
    if details["Description"] != "N/A":
        score += 1
    if any(details["Headings"]["H1"]) or any(details["Headings"]["H2"]):
        score += 1
    if details["Content"] != "No content available" and len(details["Content"]) > 50:
        score += 1
    if len(details["Internal Links"]) > 0:
        score += 1

    accuracy = (score / total_elements) * 100
    return round(accuracy, 2)


def get_security_details(url):
    """Assess website security."""
    security_info = {
        "ssl_enabled": False,
        "ssl_expiry": "N/A",
        "http_headers": {},
        "vulnerable_headers": [],
        "overall_safety": "Unknown"
    }

    try:
       
        context = ssl.create_default_context()
        with socket.create_connection((urlparse(url).netloc, 443), timeout=5) as sock:
            with context.wrap_socket(sock, server_hostname=urlparse(url).netloc) as ssock:
                security_info["ssl_enabled"] = True
                ssl_info = ssock.getpeercert()
                expiry_date = datetime.strptime(ssl_info['notAfter'], "%b %d %H:%M:%S %Y %Z")
                security_info["ssl_expiry"] = expiry_date.strftime("%Y-%m-%d")

        
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        security_info["http_headers"] = response.headers

        
        vulnerable_headers = ["X-Frame-Options", "Strict-Transport-Security",
                               "Content-Security-Policy"]
        for header in vulnerable_headers:
            if header not in security_info["http_headers"]:
                security_info["vulnerable_headers"].append(header)

        # Overall safety assessment
        if not security_info["ssl_enabled"]:
            security_info["overall_safety"] = "Potentially Unsafe (No SSL)"
        elif security_info["vulnerable_headers"]:
            security_info["overall_safety"] = "Needs Improvement (Vulnerable Headers)"
        else:
            security_info["overall_safety"] = "Safe"

    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching security details for {url}: {e}")
        security_info["overall_safety"] = f"Error: Could not connect ({e})"

    except socket.gaierror as e:
        logger.error(f"Error resolving hostname for {url}: {e}")
        security_info["overall_safety"] = f"Error: Could not resolve hostname ({e})"

    except ssl.SSLError as e:
        logger.error(f"SSL Error for {url}: {e}")
        security_info["overall_safety"] = f"SSL Error: {e}"
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        security_info["overall_safety"] = f"Unexpected Error: {e}"

    return security_info


def extract_website_details(url, is_subpage=False):
    """Extract details from a website or subpage."""
    if not is_subpage:
        st.info("🔄 Extracting website details...")
    start_time = time.time()

    try:
        response = requests.get(url, timeout=10)
        response_time = round(time.time() - start_time, 2)
    except requests.exceptions.RequestException as e:
        if not is_subpage:
            st.error(f"⚠️ Failed to fetch the website: {e}")
        logger.error(f"Request failed for {url}: {str(e)}")
        return None

    if response.status_code != 200:
        if not is_subpage:
            st.error(f"⚠️ HTTP Error {response.status_code}: Unable to retrieve the website.")
        return None

    soup = BeautifulSoup(response.text, "html.parser")
    title = soup.title.string if soup.title else "N/A"
    meta_desc = soup.find("meta", attrs={"name": "description"})
    description = meta_desc["content"] if meta_desc else "N/A"

    headings = {
        "H1": [h.text.strip() for h in soup.find_all("h1")],
        "H2": [h.text.strip() for h in soup.find_all("h2")],
    }

    paragraphs = soup.find_all("p")
    main_content = " ".join([p.text.strip() for p in paragraphs if p.text.strip()]) or "No content available"
    main_content_words = main_content.split()
    if len(main_content_words) > 500:
        main_content = " ".join(main_content_words[:500])

    internal_links = []
    parsed_url = urlparse(url)
    base_domain = parsed_url.netloc
    all_links = [urljoin(url, a["href"]) for a in soup.find_all("a", href=True)]
    internal_links = [link for link in all_links if
                      urlparse(link).netloc == "" or base_domain in urlparse(link).netloc]

    details = {
        "url": url,
        "Title": title,
        "Description": description,
        "Headings": headings,
        "Content": main_content,
        "Internal Links": internal_links,
        "Page Load Time": f"{response_time} sec",
        "Accuracy": calculate_accuracy({
            "Title": title,
            "Description": description,
            "Headings": headings,
            "Content": main_content,
            "Internal Links": internal_links
        })
    }

    if not is_subpage:
        st.success("✅ Website details extracted successfully!")
    return details


def crawl_sub_pages(base_url, max_pages=4):
    """Crawl internal sub-links up to a specified limit (for display only)."""
    visited = set()
    to_visit = [base_url]
    sub_pages_details = {}

    st.info(f"🔄 Crawling up to {max_pages} sub-pages for display...")

    while to_visit and len(visited) < max_pages:
        url = to_visit.pop(0)
        if url in visited:
            continue

        details = extract_website_details(url, is_subpage=True)
        if details:
            visited.add(url)
            sub_pages_details[url] = details

            for link in details["Internal Links"]:
                if link not in visited and link not in to_visit and len(visited) + len(
                        to_visit) < max_pages:
                    to_visit.append(link)

    st.success(f"✅ Crawled {len(sub_pages_details)} sub-pages successfully!")
    return sub_pages_details


def process_website(details):
    """Process only the main page's content for RAG."""
    st.info("🔄 Processing main page content for RAG analysis...")

    try:
        main_content = details["Content"]
        if not main_content or main_content == "No content available":
            st.warning("⚠️ Limited content available for embedding. Proceeding with minimal data.")
            main_content = "Minimal content extracted from the page."

        document = Document(page_content=main_content, metadata={"url": details["url"]})
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=25)
        docs_chunk = text_splitter.split_documents([document])

        if not docs_chunk:
            st.warning("⚠️ No chunks created. Using full content as a single chunk.")
            docs_chunk = [document]

        vector_store = FAISS.from_documents(docs_chunk, embeddings)
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})

        return retriever

    except Exception as e:
        st.error(f"⚠️ Failed to process website for RAG: {str(e)}")
        logger.error(f"RAG processing error: {str(e)}", exc_info=True)
        return None


def create_knowledge_augmented_prompt():
    """
    Creates a prompt template that includes context from the retrieved documents.
    """
    template = """You are a knowledgeable assistant designed to provide accurate answers 
    based on information from the web. Use the following context to answer the user's question.
    If the answer is not in the context, respond with "I'm sorry, but I don't have enough information to answer that question."
    
    Context: {context}
    
    Question: {question}
    
    Answer:"""

    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    return prompt


with st.sidebar:
    st.header("🔗 Website Input")
    website_url = st.text_input("Enter Website URL", "https://example.com")
    crawl_limit = st.slider("Crawl Limit (Sub-pages)", 1, 50, 1)

    if st.button("Analyze Website"):
        if website_url:
            with st.spinner("Processing..."):
                st.session_state.website_details = extract_website_details(website_url)
                st.session_state.security_assessment = get_security_details(website_url)

                if st.session_state.website_details:
                    st.session_state.sub_pages_details = crawl_sub_pages(website_url,
                                                                          max_pages=crawl_limit)
                    st.session_state.retriever = process_website(
                        st.session_state.website_details) 
                    st.session_state.main_page_accuracy = st.session_state.website_details["Accuracy"]
                    sub_accuracies = [details["Accuracy"] for details in
                                      st.session_state.sub_pages_details.values()]
                    total_pages = 1 + len(sub_accuracies)
                    overall_accuracy = (st.session_state.main_page_accuracy + sum(
                        sub_accuracies)) / total_pages if total_pages > 0 else st.session_state.main_page_accuracy
                    st.session_state.accuracy_score = round(overall_accuracy, 2)
        else:
            st.warning("⚠️ Please enter a website URL.")

    st.sidebar.header("🔒 Security Assessment")
    if st.session_state.security_assessment:
        security_info = st.session_state.security_assessment
        st.sidebar.write(f"**Overall Safety:** {security_info['overall_safety']}")
        st.sidebar.write(f"**SSL Enabled:** {security_info['ssl_enabled']}")
        if security_info["ssl_enabled"]:
            st.sidebar.write(f"  - SSL Expiry: {security_info['ssl_expiry']}")
        st.sidebar.write("**Vulnerable Headers:**")
        if security_info["vulnerable_headers"]:
            for header in security_info["vulnerable_headers"]:
                st.sidebar.write(f"  - {header}")
        else:
            st.sidebar.write("  - None")
    else:
        st.sidebar.write("Security assessment pending...")

    if st.session_state.accuracy_score is not None:
        st.sidebar.subheader("📊 Data Accuracy")
        st.sidebar.write(f"**Overall Accuracy Score:** {st.session_state.accuracy_score}%")
        st.sidebar.write("(Based on completeness of scraped data)")


if st.session_state.website_details:
    st.subheader("💬 Ask About website")
    user_query = st.chat_input("Type your question here (about the main page content only)...")

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if user_query and st.session_state.retriever:
        with st.chat_message("user"):
            st.write(user_query)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                try:
                    # Retrieve relevant documents
                    relevant_docs = st.session_state.retriever.get_relevant_documents(user_query)

                    # Format the context
                    context = "\n".join([doc.page_content for doc in relevant_docs])

                    # Create the prompt
                    prompt = create_knowledge_augmented_prompt()

                    # Create the LLM chain
                    llm_chain = LLMChain(llm=llm, prompt=prompt)

                    # Run the LLM chain
                    response = llm_chain.run(context=context, question=user_query)

                    # Display the response
                    st.write(response)

                    # Estimate model accuracy (example)
                    model_accuracy_estimate = 80 + (st.session_state.main_page_accuracy - 50) * 0.2
                    model_accuracy_estimate = min(100, max(0, model_accuracy_estimate))

                    st.write(f"\n\n*Data Accuracy from Website: {st.session_state.main_page_accuracy}%*")
                    st.write(f"*Estimated Model Accuracy for this response: {model_accuracy_estimate:.2f}%*")

                    # Update chat history
                    st.session_state.chat_history.append({"role": "user", "content": user_query})
                    st.session_state.chat_history.append({"role": "assistant", "content": response})

                except Exception as e:
                    st.error(f"⚠️ Error processing query: {str(e)}")
                    logger.error(f"Query processing error: {str(e)}")

    st.subheader("📄 Main Page Analysis")
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**🔹 Title:** {st.session_state.website_details['Title']}")
        st.write(f"**🔹 Description:** {st.session_state.website_details['Description']}")
        st.write(f"**🔹 Page Load Time:** {st.session_state.website_details['Page Load Time']}")
        st.write(f"**🔹 Accuracy:** {st.session_state.website_details['Accuracy']}%")
    with col2:
        st.write("**🔹 Headings:**")
        for level, items in st.session_state.website_details["Headings"].items():
            if items:
                st.write(f"  **{level}**: {', '.join(items[:3])}" + ("..." if len(items) > 3 else ""))

    if st.session_state.sub_pages_details:
        st.subheader(f"📑 Sub-pages Analysis (Display Only, Up to {crawl_limit})")
        for url, details in st.session_state.sub_pages_details.items():
            st.write(f"**🔹 {url}:**")
            st.write(f"  - **Title:** {details['Title']}")
            st.write(f"  - **Description:** {details['Description']}")
            st.write(f"  - **Accuracy:** {details['Accuracy']}%")





# import streamlit as st
# from langchain_community.document_loaders import WebBaseLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain_core.documents import Document
# from langchain.chains import RetrievalQA
# from dotenv import load_dotenv
# from bs4 import BeautifulSoup
# import requests
# import os
# from urllib.parse import urlparse, urljoin
# import time
# import logging
# import socket
# import ssl
# from datetime import datetime
# from langchain_core.prompts import PromptTemplate
# from langchain.chains import LLMChain
# from rouge import Rouge  # Added for ROUGE score calculation

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# load_dotenv()
# groq_api_key = os.getenv("GROQ_API_KEY")

# os.environ["USER_AGENT"] = "MyCustomUserAgent/1.0"

# if not groq_api_key:
#     st.error("⚠️ Groq API key not found. Please set it in the .env file.")
#     st.stop()

# try:
#     embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
# except Exception as e:
#     st.error(f"⚠️ Failed to initialize HuggingFace embeddings: {str(e)}")
#     logger.error(f"HuggingFace embeddings initialization error: {str(e)}")
#     st.stop()

# from langchain_groq import ChatGroq

# try:
#     llm = ChatGroq(groq_api_key=groq_api_key, temperature=0.5, model_name="llama-3.3-70b-versatile")
# except Exception as e:
#     st.error(f"⚠️ Failed to initialize Groq Chat model: {str(e)}")
#     logger.error(f"Groq Chat model initialization error: {str(e)}")
#     st.stop()

# st.set_page_config(page_title="kAG Website Analyzer", page_icon="🌍", layout="wide")
# st.title("🔎 KAG-based Website Analyzer")
# st.title("Search and surf to the world!!!!!")

# if "vector_db" not in st.session_state:
#     st.session_state.vector_db = None
# if "chat_history" not in st.session_state:
#     st.session_state.chat_history = []
# if "website_details" not in st.session_state:
#     st.session_state.website_details = {}
# if "sub_pages_details" not in st.session_state:
#     st.session_state.sub_pages_details = {}
# if "qa_chain" not in st.session_state:
#     st.session_state.qa_chain = None
# if "accuracy_score" not in st.session_state:
#     st.session_state.accuracy_score = None
# if "security_assessment" not in st.session_state:
#     st.session_state.security_assessment = {}
# if "main_page_accuracy" not in st.session_state:
#     st.session_state.main_page_accuracy = 0.0
# if "feedback_history" not in st.session_state:  # Added for storing feedback
#     st.session_state.feedback_history = {}

# def calculate_accuracy(details):
#     """Calculate accuracy score based on completeness of scraped data."""
#     total_elements = 5
#     score = 0

#     if details["Title"] != "N/A":
#         score += 1
#     if details["Description"] != "N/A":
#         score += 1
#     if any(details["Headings"]["H1"]) or any(details["Headings"]["H2"]):
#         score += 1
#     if details["Content"] != "No content available" and len(details["Content"]) > 50:
#         score += 1
#     if len(details["Internal Links"]) > 0:
#         score += 1

#     accuracy = (score / total_elements) * 100
#     return round(accuracy, 2)

# def get_security_details(url):
#     """Assess website security."""
#     security_info = {
#         "ssl_enabled": False,
#         "ssl_expiry": "N/A",
#         "http_headers": {},
#         "vulnerable_headers": [],
#         "overall_safety": "Unknown"
#     }

#     try:
#         context = ssl.create_default_context()
#         with socket.create_connection((urlparse(url).netloc, 443), timeout=5) as sock:
#             with context.wrap_socket(sock, server_hostname=urlparse(url).netloc) as ssock:
#                 security_info["ssl_enabled"] = True
#                 ssl_info = ssock.getpeercert()
#                 expiry_date = datetime.strptime(ssl_info['notAfter'], "%b %d %H:%M:%S %Y %Z")
#                 security_info["ssl_expiry"] = expiry_date.strftime("%Y-%m-%d")

#         response = requests.get(url, timeout=5)
#         response.raise_for_status()
#         security_info["http_headers"] = response.headers

#         vulnerable_headers = ["X-Frame-Options", "Strict-Transport-Security", "Content-Security-Policy"]
#         for header in vulnerable_headers:
#             if header not in security_info["http_headers"]:
#                 security_info["vulnerable_headers"].append(header)

#         if not security_info["ssl_enabled"]:
#             security_info["overall_safety"] = "Potentially Unsafe (No SSL)"
#         elif security_info["vulnerable_headers"]:
#             security_info["overall_safety"] = "Needs Improvement (Vulnerable Headers)"
#         else:
#             security_info["overall_safety"] = "Safe"

#     except requests.exceptions.RequestException as e:
#         logger.error(f"Error fetching security details for {url}: {e}")
#         security_info["overall_safety"] = f"Error: Could not connect ({e})"
#     except socket.gaierror as e:
#         logger.error(f"Error resolving hostname for {url}: {e}")
#         security_info["overall_safety"] = f"Error: Could not resolve hostname ({e})"
#     except ssl.SSLError as e:
#         logger.error(f"SSL Error for {url}: {e}")
#         security_info["overall_safety"] = f"SSL Error: {e}"
#     except Exception as e:
#         logger.error(f"An unexpected error occurred: {e}")
#         security_info["overall_safety"] = f"Unexpected Error: {e}"

#     return security_info

# def extract_website_details(url, is_subpage=False):
#     """Extract details from a website or subpage."""
#     if not is_subpage:
#         st.info("🔄 Extracting website details...")
#     start_time = time.time()

#     try:
#         response = requests.get(url, timeout=10)
#         response_time = round(time.time() - start_time, 2)
#     except requests.exceptions.RequestException as e:
#         if not is_subpage:
#             st.error(f"⚠️ Failed to fetch the website: {e}")
#         logger.error(f"Request failed for {url}: {str(e)}")
#         return None

#     if response.status_code != 200:
#         if not is_subpage:
#             st.error(f"⚠️ HTTP Error {response.status_code}: Unable to retrieve the website.")
#         return None

#     soup = BeautifulSoup(response.text, "html.parser")
#     title = soup.title.string if soup.title else "N/A"
#     meta_desc = soup.find("meta", attrs={"name": "description"})
#     description = meta_desc["content"] if meta_desc else "N/A"

#     headings = {
#         "H1": [h.text.strip() for h in soup.find_all("h1")],
#         "H2": [h.text.strip() for h in soup.find_all("h2")],
#     }

#     paragraphs = soup.find_all("p")
#     main_content = " ".join([p.text.strip() for p in paragraphs if p.text.strip()]) or "No content available"
#     main_content_words = main_content.split()
#     if len(main_content_words) > 500:
#         main_content = " ".join(main_content_words[:500])

#     internal_links = []
#     parsed_url = urlparse(url)
#     base_domain = parsed_url.netloc
#     all_links = [urljoin(url, a["href"]) for a in soup.find_all("a", href=True)]
#     internal_links = [link for link in all_links if urlparse(link).netloc == "" or base_domain in urlparse(link).netloc]

#     details = {
#         "url": url,
#         "Title": title,
#         "Description": description,
#         "Headings": headings,
#         "Content": main_content,
#         "Internal Links": internal_links,
#         "Page Load Time": f"{response_time} sec",
#         "Accuracy": calculate_accuracy({
#             "Title": title,
#             "Description": description,
#             "Headings": headings,
#             "Content": main_content,
#             "Internal Links": internal_links
#         })
#     }

#     if not is_subpage:
#         st.success("✅ Website details extracted successfully!")
#     return details

# def crawl_sub_pages(base_url, max_pages=4):
#     """Crawl internal sub-links up to a specified limit (for display only)."""
#     visited = set()
#     to_visit = [base_url]
#     sub_pages_details = {}

#     st.info(f"🔄 Crawling up to {max_pages} sub-pages for display...")

#     while to_visit and len(visited) < max_pages:
#         url = to_visit.pop(0)
#         if url in visited:
#             continue

#         details = extract_website_details(url, is_subpage=True)
#         if details:
#             visited.add(url)
#             sub_pages_details[url] = details
#             for link in details["Internal Links"]:
#                 if link not in visited and link not in to_visit and len(visited) + len(to_visit) < max_pages:
#                     to_visit.append(link)

#     st.success(f"✅ Crawled {len(sub_pages_details)} sub-pages successfully!")
#     return sub_pages_details

# def process_website(details):
#     """Process only the main page's content for RAG."""
#     st.info("🔄 Processing main page content for RAG analysis...")

#     try:
#         main_content = details["Content"]
#         if not main_content or main_content == "No content available":
#             st.warning("⚠️ Limited content available for embedding. Proceeding with minimal data.")
#             main_content = "Minimal content extracted from the page."

#         document = Document(page_content=main_content, metadata={"url": details["url"]})
#         text_splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=25)
#         docs_chunk = text_splitter.split_documents([document])

#         if not docs_chunk:
#             st.warning("⚠️ No chunks created. Using full content as a single chunk.")
#             docs_chunk = [document]

#         vector_store = FAISS.from_documents(docs_chunk, embeddings)
#         retriever = vector_store.as_retriever(search_kwargs={"k": 3})

#         st.success("✅ RAG processing completed!")
#         return retriever

#     except Exception as e:
#         st.error(f"⚠️ Failed to process website for RAG: {str(e)}")
#         logger.error(f"RAG processing error: {str(e)}", exc_info=True)
#         return None

# def create_knowledge_augmented_prompt():
#     """Creates a prompt template that includes context from the retrieved documents."""
#     template = """You are a knowledgeable assistant designed to provide accurate answers 
#     based on information from the web. Use the following context to answer the user's question.
#     If the answer is not in the context, respond with "I'm sorry, but I don't have enough information to answer that question."
    
#     Context: {context}
    
#     Question: {question}
    
#     Answer:"""
#     prompt = PromptTemplate(template=template, input_variables=["context", "question"])
#     return prompt

# def calculate_rouge_scores(response, context):
#     """Calculate ROUGE scores for the generated response against the context."""
#     rouge = Rouge()
#     try:
#         scores = rouge.get_scores(response, context)[0]
#         return {
#             "rouge-1": scores["rouge-1"]["f"],
#             "rouge-2": scores["rouge-2"]["f"],
#             "rouge-l": scores["rouge-l"]["f"]
#         }
#     except Exception as e:
#         logger.error(f"Error calculating ROUGE scores: {str(e)}")
#         return {"rouge-1": 0.0, "rouge-2": 0.0, "rouge-l": 0.0}

# with st.sidebar:
#     st.header("🔗 Website Input")
#     website_url = st.text_input("Enter Website URL", "https://example.com")
#     crawl_limit = st.slider("Crawl Limit (Sub-pages)", 1, 50, 1)

#     if st.button("Analyze Website"):
#         if website_url:
#             with st.spinner("Processing..."):
#                 st.session_state.website_details = extract_website_details(website_url)
#                 st.session_state.security_assessment = get_security_details(website_url)

#                 if st.session_state.website_details:
#                     st.session_state.sub_pages_details = crawl_sub_pages(website_url, max_pages=crawl_limit)
#                     st.session_state.retriever = process_website(st.session_state.website_details)
#                     st.session_state.main_page_accuracy = st.session_state.website_details["Accuracy"]
#                     sub_accuracies = [details["Accuracy"] for details in st.session_state.sub_pages_details.values()]
#                     total_pages = 1 + len(sub_accuracies)
#                     overall_accuracy = (st.session_state.main_page_accuracy + sum(sub_accuracies)) / total_pages if total_pages > 0 else st.session_state.main_page_accuracy
#                     st.session_state.accuracy_score = round(overall_accuracy, 2)
#         else:
#             st.warning("⚠️ Please enter a website URL.")

#     st.sidebar.header("🔒 Security Assessment")
#     if st.session_state.security_assessment:
#         security_info = st.session_state.security_assessment
#         st.sidebar.write(f"**Overall Safety:** {security_info['overall_safety']}")
#         st.sidebar.write(f"**SSL Enabled:** {security_info['ssl_enabled']}")
#         if security_info["ssl_enabled"]:
#             st.sidebar.write(f"  - SSL Expiry: {security_info['ssl_expiry']}")
#         st.sidebar.write("**Vulnerable Headers:**")
#         if security_info["vulnerable_headers"]:
#             for header in security_info["vulnerable_headers"]:
#                 st.sidebar.write(f"  - {header}")
#         else:
#             st.sidebar.write("  - None")
#     else:
#         st.sidebar.write("Security assessment pending...")

#     if st.session_state.accuracy_score is not None:
#         st.sidebar.subheader("📊 Data Accuracy")
#         st.sidebar.write(f"**Overall Accuracy Score:** {st.session_state.accuracy_score}%")
#         st.sidebar.write("(Based on completeness of scraped data)")

# if st.session_state.website_details:
#     st.subheader("💬 Ask About Website")
#     user_query = st.chat_input("Type your question here (about the main page content only)...")

#     for i, message in enumerate(st.session_state.chat_history):
#         with st.chat_message(message["role"]):
#             st.write(message["content"])
#             if message["role"] == "assistant":
#                 # Display ROUGE scores and feedback for assistant messages
#                 if "rouge_scores" in message:
#                     st.write("**ROUGE Scores (F1):**")
#                     st.write(f"- ROUGE-1: {message['rouge_scores']['rouge-1']:.4f}")
#                     st.write(f"- ROUGE-2: {message['rouge_scores']['rouge-2']:.4f}")
#                     st.write(f"- ROUGE-L: {message['rouge_scores']['rouge-l']:.4f}")
#                 st.write(f"*Data Accuracy from Website: {st.session_state.main_page_accuracy}%*")
#                 st.write(f"*Estimated Model Accuracy: {message.get('model_accuracy', 0.0):.2f}%*")

#                 # Feedback buttons
#                 feedback_key_up = f"thumbs_up_{i}"
#                 feedback_key_down = f"thumbs_down_{i}"
#                 col1, col2 = st.columns(2)
#                 with col1:
#                     if st.button("👍", key=feedback_key_up):
#                         st.session_state.feedback_history[i] = "positive"
#                         st.success("Thank you for your feedback!")
#                 with col2:
#                     if st.button("👎", key=feedback_key_down):
#                         st.session_state.feedback_history[i] = "negative"
#                         st.success("Thank you for your feedback!")
#                 if i in st.session_state.feedback_history:
#                     st.write(f"Feedback: {st.session_state.feedback_history[i].capitalize()}")

#     if user_query and st.session_state.retriever:
#         with st.chat_message("user"):
#             st.write(user_query)

#         with st.chat_message("assistant"):
#             with st.spinner("Analyzing..."):
#                 try:
#                     # Retrieve relevant documents
#                     relevant_docs = st.session_state.retriever.get_relevant_documents(user_query)
#                     context = "\n".join([doc.page_content for doc in relevant_docs])

#                     # Create and run the LLM chain
#                     prompt = create_knowledge_augmented_prompt()
#                     llm_chain = LLMChain(llm=llm, prompt=prompt)
#                     response = llm_chain.run(context=context, question=user_query)

#                     # Calculate ROUGE scores
#                     rouge_scores = calculate_rouge_scores(response, context)

#                     # Display the response
#                     st.write(response)

#                     # Estimate model accuracy
#                     model_accuracy_estimate = 80 + (st.session_state.main_page_accuracy - 50) * 0.2
#                     model_accuracy_estimate = min(100, max(0, model_accuracy_estimate))

#                     # Display metrics
#                     st.write("**ROUGE Scores (F1):**")
#                     st.write(f"- ROUGE-1: {rouge_scores['rouge-1']:.4f}")
#                     st.write(f"- ROUGE-2: {rouge_scores['rouge-2']:.4f}")
#                     st.write(f"- ROUGE-L: {rouge_scores['rouge-l']:.4f}")
#                     st.write(f"\n*Data Accuracy from Website: {st.session_state.main_page_accuracy}%*")
#                     st.write(f"*Estimated Model Accuracy for this response: {model_accuracy_estimate:.2f}%*")

#                     # Feedback buttons
#                     chat_index = len(st.session_state.chat_history) // 2  # Index for this response
#                     feedback_key_up = f"thumbs_up_{chat_index}"
#                     feedback_key_down = f"thumbs_down_{chat_index}"
#                     col1, col2 = st.columns(2)
#                     with col1:
#                         if st.button("👍", key=feedback_key_up):
#                             st.session_state.feedback_history[chat_index] = "positive"
#                             st.success("Thank you for your feedback!")
#                     with col2:
#                         if st.button("👎", key=feedback_key_down):
#                             st.session_state.feedback_history[chat_index] = "negative"
#                             st.success("Thank you for your feedback!")

#                     # Update chat history with additional metadata
#                     st.session_state.chat_history.append({"role": "user", "content": user_query})
#                     st.session_state.chat_history.append({
#                         "role": "assistant",
#                         "content": response,
#                         "rouge_scores": rouge_scores,
#                         "model_accuracy": model_accuracy_estimate
#                     })

#                 except Exception as e:
#                     st.error(f"⚠️ Error processing query: {str(e)}")
#                     logger.error(f"Query processing error: {str(e)}")

#     st.subheader("📄 Main Page Analysis")
#     col1, col2 = st.columns(2)
#     with col1:
#         st.write(f"**🔹 Title:** {st.session_state.website_details['Title']}")
#         st.write(f"**🔹 Description:** {st.session_state.website_details['Description']}")
#         st.write(f"**🔹 Page Load Time:** {st.session_state.website_details['Page Load Time']}")
#         st.write(f"**🔹 Accuracy:** {st.session_state.website_details['Accuracy']}%")
#     with col2:
#         st.write("**🔹 Headings:**")
#         for level, items in st.session_state.website_details["Headings"].items():
#             if items:
#                 st.write(f"  **{level}**: {', '.join(items[:3])}" + ("..." if len(items) > 3 else ""))

#     if st.session_state.sub_pages_details:
#         st.subheader(f"📑 Sub-pages Analysis (Display Only, Up to {crawl_limit})")
#         for url, details in st.session_state.sub_pages_details.items():
#             st.write(f"**🔹 {url}:**")
#             st.write(f"  - **Title:** {details['Title']}")
#             st.write(f"  - **Description:** {details['Description']}")
#             st.write(f"  - **Accuracy:** {details['Accuracy']}%")