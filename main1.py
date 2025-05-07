import os
import requests
import re
from urllib.parse import urlparse, urljoin
from bs4 import BeautifulSoup
import time
import numpy as np
import pandas as pd
import streamlit as st
import threading
import validators
import pickle
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import nltk
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from gensim.models import Word2Vec


try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class LSTMEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=1):
        super(LSTMEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, hidden_dim)  # bidirectional output
        self.hidden_dim = hidden_dim
    
    def forward(self, x, lengths):
        
        embedded = self.embedding(x) 
        
        
        packed = nn.utils.rnn.pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)
        
        
        output, (hidden, _) = self.lstm(packed)
        
        
        hidden_cat = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        
        
        embedding = self.fc(hidden_cat)
        
        return embedding

class TextDataset(Dataset):
    def __init__(self, texts, word_to_idx, max_length=100):
        self.texts = texts
        self.word_to_idx = word_to_idx
        self.max_length = max_length
        self.unk_idx = word_to_idx.get('<UNK>', 1)
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        tokens = word_tokenize(self.texts[idx].lower())
        tokens = tokens[:self.max_length]  
        
        
        indices = [self.word_to_idx.get(token, self.unk_idx) for token in tokens]
        
        return torch.tensor(indices, dtype=torch.long), len(indices)

def collate_fn(batch):
    texts, lengths = zip(*batch)
    texts_padded = pad_sequence(texts, batch_first=True, padding_value=0)
    return texts_padded, torch.tensor(lengths, dtype=torch.long)

class LSTMWebsiteSummarizer:
    def __init__(self, base_url, max_depth=1, max_pages=20):
        self.base_url = base_url
        self.base_domain = urlparse(base_url).netloc
        self.max_depth = max_depth
        self.max_pages = max_pages
        self.visited = set()
        self.to_visit = [(base_url, 0)]  
        self.progress_callback = None
        self.is_scraping = False
        
        
        self.embedding_dim = 100
        self.hidden_dim = 128
        self.batch_size = 32
        self.num_epochs = 5
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        
        self.word_to_idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx_to_word = {0: '<PAD>', 1: '<UNK>'}
        self.vocab_size = len(self.word_to_idx)
        self.model = None
        self.w2v_model = None
        
        
        self.embeddings = {}
        self.chunks = []
        
    def set_progress_callback(self, callback):
        """Set a callback function to report progress during scraping."""
        self.progress_callback = callback

    def is_valid_url(self, url):
        """Check if the URL is valid and belongs to the same domain."""
        if not validators.url(url):
            return False
        
        parsed_url = urlparse(url)
        if parsed_url.netloc != self.base_domain:
            return False
            
       
        excluded_extensions = ['.pdf', '.jpg', '.jpeg', '.png', '.gif', '.css', '.js', '.xml']
        if any(url.lower().endswith(ext) for ext in excluded_extensions):
            return False
            
        return True

    def extract_text_from_html(self, html_content):
        """Extract readable text content from HTML."""
        soup = BeautifulSoup(html_content, 'html.parser')
        
       
        for script in soup(["script", "style", "header", "footer", "nav"]):
            script.extract()
        
        
        text = soup.get_text(separator=' ', strip=True)
        
        
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
       
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    def extract_links(self, url, html_content):
        """Extract all links from the HTML content."""
        soup = BeautifulSoup(html_content, 'html.parser')
        links = []
        
        for link in soup.find_all('a', href=True):
            href = link['href']
            full_url = urljoin(url, href)
            
            if self.is_valid_url(full_url) and full_url not in self.visited:
                links.append(full_url)
                
        return links

    def scrape_website(self):
        """Scrape the website and collect content."""
        self.is_scraping = True
        pages_scraped = 0
        
        while self.to_visit and pages_scraped < self.max_pages and self.is_scraping:
            url, depth = self.to_visit.pop(0)
            
            if url in self.visited:
                continue
                
            self.visited.add(url)
            
            if self.progress_callback:
                self.progress_callback(f"Scraping {url} (depth: {depth})")
            
            try:
                response = requests.get(url, timeout=10)
                response.raise_for_status()  
                
                html_content = response.text
                text_content = self.extract_text_from_html(html_content)
                
                
                self.pages_content.append({
                    "url": url,
                    "content": text_content,
                    "depth": depth
                })
                
                pages_scraped += 1
                
                if self.progress_callback:
                    self.progress_callback(f"Scraped {pages_scraped}/{self.max_pages} pages")
                
                
                if depth < self.max_depth:
                    links = self.extract_links(url, html_content)
                    for link in links:
                        if link not in self.visited:
                            self.to_visit.append((link, depth + 1))
                
                
                time.sleep(1)
                
            except Exception as e:
                if self.progress_callback:
                    self.progress_callback(f"Error scraping {url}: {e}")
                
        if self.progress_callback:
            self.progress_callback(f"Completed scraping {pages_scraped} pages.")
        
        self.is_scraping = False
        return pages_scraped

    def stop_scraping(self):
        """Stop the scraping process."""
        self.is_scraping = False

    def build_vocabulary(self):
        """Build vocabulary from scraped text."""
        if self.progress_callback:
            self.progress_callback("Building vocabulary...")
        
        all_tokens = []
        for page in self.pages_content:
            tokens = word_tokenize(page["content"].lower())
            all_tokens.append(tokens)
        
        
        self.w2v_model = Word2Vec(all_tokens, vector_size=self.embedding_dim, window=5, min_count=1, workers=4)
        
        
        for tokens in all_tokens:
            for token in tokens:
                if token not in self.word_to_idx:
                    idx = len(self.word_to_idx)
                    self.word_to_idx[token] = idx
                    self.idx_to_word[idx] = token
        
        self.vocab_size = len(self.word_to_idx)
        
        if self.progress_callback:
            self.progress_callback(f"Vocabulary built with {self.vocab_size} words")
        
        return self.vocab_size

    def chunk_text(self, text, chunk_size=200, overlap=50):
        """Split text into overlapping chunks."""
        tokens = word_tokenize(text.lower())
        
        chunks = []
        for i in range(0, len(tokens), chunk_size - overlap):
            chunk = tokens[i:i + chunk_size]
            if len(chunk) > chunk_size // 2:  
                chunks.append(' '.join(chunk))
                
        return chunks

    def train_lstm_model(self):
        """Train LSTM model for text embedding."""
        if self.progress_callback:
            self.progress_callback("Preparing data for LSTM training...")
        
        
        all_chunks = []
        for page in self.pages_content:
            chunks = self.chunk_text(page["content"])
            self.chunks.extend([{
                "text": chunk,
                "url": page["url"],
                "depth": page["depth"]
            } for chunk in chunks])
            all_chunks.extend(chunks)
            
       
        dataset = TextDataset(all_chunks, self.word_to_idx)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, collate_fn=collate_fn)
        
       
        self.model = LSTMEmbedding(self.vocab_size, self.embedding_dim, self.hidden_dim)
        self.model.to(self.device)
        
       
        if self.w2v_model:
            pretrained_embeddings = torch.zeros(self.vocab_size, self.embedding_dim)
            
            for word, idx in self.word_to_idx.items():
                if word in self.w2v_model.wv:
                    pretrained_embeddings[idx] = torch.tensor(self.w2v_model.wv[word])
                    
            self.model.embedding.weight.data.copy_(pretrained_embeddings)
        
      
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        
        if self.progress_callback:
            self.progress_callback("Training LSTM model...")
        
        
        self.model.train()
        for epoch in range(self.num_epochs):
            total_loss = 0
            
            for batch_texts, batch_lengths in dataloader:
                batch_texts = batch_texts.to(self.device)
                batch_lengths = batch_lengths.to(self.device)
                
                optimizer.zero_grad()
                
                
                embeddings = self.model(batch_texts, batch_lengths)
                
                # Calculate loss - simplified to use identity loss for demonstration
                # In a real implementation, you might use a triplet loss or other contrastive objective
                # Here we're just training the model to produce consistent embeddings
                target = embeddings.detach().clone()
                loss = criterion(embeddings, target)
                
                # Backward and optimize
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)
            if self.progress_callback:
                self.progress_callback(f"Epoch {epoch+1}/{self.num_epochs}, Loss: {avg_loss:.4f}")
        
        if self.progress_callback:
            self.progress_callback("LSTM model training complete")
            
       
        self.generate_embeddings()
        
        return True
            
    def generate_embeddings(self):
        """Generate embeddings for all text chunks."""
        if self.progress_callback:
            self.progress_callback("Generating embeddings for all chunks...")
            
        dataset = TextDataset([chunk["text"] for chunk in self.chunks], self.word_to_idx)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, collate_fn=collate_fn)
        
        self.model.eval()
        all_embeddings = []
        
        with torch.no_grad():
            for batch_texts, batch_lengths in dataloader:
                batch_texts = batch_texts.to(self.device)
                batch_lengths = batch_lengths.to(self.device)
                
                embeddings = self.model(batch_texts, batch_lengths)
                all_embeddings.append(embeddings.cpu().numpy())
        
        all_embeddings = np.vstack(all_embeddings)
        
        
        for i, chunk in enumerate(self.chunks):
            self.embeddings[i] = all_embeddings[i]
        
        if self.progress_callback:
            self.progress_callback(f"Generated {len(self.embeddings)} embeddings")
            
        return len(self.embeddings)
        
    def save_model(self, directory="website_lstm_data"):
        """Save the LSTM model, vocabulary, and embeddings."""
        os.makedirs(directory, exist_ok=True)
        
        
        torch.save(self.model.state_dict(), f"{directory}/lstm_model.pt")
        
        
        with open(f"{directory}/vocabulary.pkl", "wb") as f:
            pickle.dump({
                "word_to_idx": self.word_to_idx,
                "idx_to_word": self.idx_to_word
            }, f)
        
        
        with open(f"{directory}/embeddings.pkl", "wb") as f:
            pickle.dump({
                "embeddings": self.embeddings,
                "chunks": self.chunks
            }, f)
        
        if self.progress_callback:
            self.progress_callback(f"Model and data saved to {directory}")

    def load_model(self, directory="website_lstm_data"):
        """Load the LSTM model, vocabulary, and embeddings."""
        if not os.path.exists(directory):
            return False
            
        try:
            
            with open(f"{directory}/vocabulary.pkl", "rb") as f:
                vocab_data = pickle.load(f)
                self.word_to_idx = vocab_data["word_to_idx"]
                self.idx_to_word = vocab_data["idx_to_word"]
                self.vocab_size = len(self.word_to_idx)
            
            
            self.model = LSTMEmbedding(self.vocab_size, self.embedding_dim, self.hidden_dim)
            
           
            self.model.load_state_dict(torch.load(f"{directory}/lstm_model.pt", map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
            
           
            with open(f"{directory}/embeddings.pkl", "rb") as f:
                data = pickle.load(f)
                self.embeddings = data["embeddings"]
                self.chunks = data["chunks"]
            
            if self.progress_callback:
                self.progress_callback(f"Model and data loaded from {directory}")
                
            return True
        except Exception as e:
            if self.progress_callback:
                self.progress_callback(f"Error loading model: {str(e)}")
            return False

    def find_similar_chunks(self, query, k=5):
        """Find chunks most similar to query."""
       
        tokens = word_tokenize(query.lower())
        
        
        indices = [self.word_to_idx.get(token, self.word_to_idx.get('<UNK>')) for token in tokens]
        tensor = torch.tensor([indices], dtype=torch.long).to(self.device)
        lengths = torch.tensor([len(indices)], dtype=torch.long).to(self.device)
        
        
        self.model.eval()
        with torch.no_grad():
            query_embedding = self.model(tensor, lengths).cpu().numpy()[0]
        
        
        similarities = {}
        for idx, emb in self.embeddings.items():
            similarity = np.dot(query_embedding, emb) / (np.linalg.norm(query_embedding) * np.linalg.norm(emb))
            similarities[idx] = similarity
        
        
        top_indices = sorted(similarities.keys(), key=lambda x: similarities[x], reverse=True)[:k]
        
        return [self.chunks[idx] for idx in top_indices]

    def answer_query(self, query, k=5):
        """Answer a query about the website using LSTM embeddings."""
        if not self.model or not self.embeddings:
            return {"answer": "Error: Model not trained. Please scrape a website first."}
            
        try:
            # Find similar chunks
            similar_chunks = self.find_similar_chunks(query, k)
            
            # Combine chunks into context
            context = "\n\n".join([chunk["text"] for chunk in similar_chunks])
            
            # Simple answer generation using the context
            answer = f"Based on the information from the website, here's what I found about '{query}':\n\n"
            answer += context
            
            return {
                "answer": answer,
                "source_documents": [chunk["url"] for chunk in similar_chunks]
            }
        except Exception as e:
            return {"answer": f"Error processing query: {str(e)}"}


# Streamlit UI
def main():
    st.set_page_config(
        page_title="LSTM Website Summarizer",
        page_icon="🔍",
        layout="wide"
    )

    st.title("🔍 LSTM Website Summarizer")
    st.subheader("Explore and query websites using LSTM embeddings")
    
    # Initialize session state
    if 'summarizer' not in st.session_state:
        st.session_state.summarizer = None
    if 'status' not in st.session_state:
        st.session_state.status = ""
    if 'scrape_thread' not in st.session_state:
        st.session_state.scrape_thread = None
    if 'is_ready' not in st.session_state:
        st.session_state.is_ready = False
    if 'query_history' not in st.session_state:
        st.session_state.query_history = []

    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        
        url = st.text_input("Website URL", placeholder="https://example.com", help="Enter the URL of the website to analyze")
        
        col1, col2 = st.columns(2)
        with col1:
            max_depth = st.number_input("Max Crawl Depth", min_value=0, max_value=3, value=1, 
                                    help="How many levels of links to follow")
        with col2:
            max_pages = st.number_input("Max Pages", min_value=1, max_value=100, value=20,
                                    help="Maximum number of pages to scrape")
        
        if st.button("Start Scraping & Training", use_container_width=True, type="primary", 
                    disabled=not url or st.session_state.scrape_thread is not None and st.session_state.scrape_thread.is_alive()):
            
            # Validate URL
            if not validators.url(url):
                st.error("Please enter a valid URL including http:// or https://")
            else:
                # Create a model directory name based on the domain
                domain = urlparse(url).netloc
                model_dir = f"website_lstm_{domain.replace('.', '_')}"
                
                # Initialize the summarizer
                st.session_state.summarizer = LSTMWebsiteSummarizer(
                    base_url=url,
                    max_depth=max_depth,
                    max_pages=max_pages
                )
                
                # Set the progress callback
                def update_status(message):
                    st.session_state.status = message
                
                st.session_state.summarizer.set_progress_callback(update_status)
                
                # Check if we have a saved model
                if not st.session_state.summarizer.load_model(model_dir):
                    # Start the scraping and training in a separate thread
                    def scrape_and_train():
                        try:
                            st.session_state.summarizer.scrape_website()
                            st.session_state.summarizer.build_vocabulary()
                            st.session_state.summarizer.train_lstm_model()
                            st.session_state.summarizer.save_model(model_dir)
                            st.session_state.is_ready = True
                        except Exception as e:
                            st.session_state.status = f"Error: {str(e)}"
                    
                    st.session_state.scrape_thread = threading.Thread(target=scrape_and_train)
                    st.session_state.scrape_thread.start()
                else:
                    st.session_state.status = "LSTM model loaded successfully!"
                    st.session_state.is_ready = True
        
        if st.session_state.scrape_thread is not None and st.session_state.scrape_thread.is_alive():
            if st.button("Stop Process", use_container_width=True):
                if st.session_state.summarizer:
                    st.session_state.summarizer.stop_scraping()
                    st.session_state.status = "Process stopped by user."
        
        st.divider()
        st.markdown("### About")
        st.markdown("""
        This app uses LSTM neural networks to analyze websites.
        
        1. Enter a website URL
        2. Configure the scraping parameters
        3. Start scraping and wait for the model to train
        4. Ask questions about the website content
        
        The LSTM model will learn to create embeddings for the website content,
        allowing semantic search and retrieval.
        """)

    # Main content area
    col1, col2 = st.columns([2, 3])
    
    with col1:
        # Status area
        st.subheader("Status")
        status_container = st.empty()
        status_container.info(st.session_state.status or "Ready to start")
        
        # Display stats
        if st.session_state.summarizer and st.session_state.summarizer.pages_content:
            st.metric("Pages Scraped", len(st.session_state.summarizer.pages_content))
            
            if st.session_state.summarizer.chunks:
                st.metric("Text Chunks", len(st.session_state.summarizer.chunks))
            
            if st.session_state.summarizer.word_to_idx:
                st.metric("Vocabulary Size", len(st.session_state.summarizer.word_to_idx))
            
            # Display the list of scraped URLs
            with st.expander("Scraped URLs", expanded=False):
                for page in st.session_state.summarizer.pages_content:
                    st.markdown(f"- [{page['url']}]({page['url']})")
    
    with col2:
        # Query interface
        st.subheader("Ask about the website")
        
        query = st.text_input("Your question", 
                             placeholder="What is this website about?",
                             disabled=not st.session_state.is_ready)
        
        if query and st.session_state.summarizer:
            with st.spinner("Finding relevant information..."):
                result = st.session_state.summarizer.answer_query(query)
                
                # Add to history
                st.session_state.query_history.append({
                    "query": query,
                    "result": result
                })
        
        # Display query history
        if st.session_state.query_history:
            for i, item in enumerate(reversed(st.session_state.query_history)):
                with st.container(border=True):
                    st.markdown(f"**Q: {item['query']}**")
                    st.markdown(item['result']['answer'])
                    
                    if 'source_documents' in item['result']:
                        with st.expander("Sources"):
                            sources = set(item['result']['source_documents'])
                            for source in sources:
                                st.markdown(f"- [{source}]({source})")

if __name__ == "__main__":
    main()