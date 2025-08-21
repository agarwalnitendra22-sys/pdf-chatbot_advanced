import streamlit as st
import re
import numpy as np
from typing import List, Tuple, Dict
import string

# Try importing with better error handling
missing_packages = []

try:
    import PyPDF2
except ImportError:
    missing_packages.append("PyPDF2")
    PyPDF2 = None

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    sklearn_available = True
except ImportError:
    missing_packages.append("scikit-learn")
    sklearn_available = False

# tiktoken removed - not needed for core functionality

try:
    import nltk
    from nltk.tokenize import sent_tokenize
    nltk_available = True
    # Download required NLTK data for cloud deployment
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        try:
            nltk.download('punkt', quiet=True)
        except Exception as e:
            # For cloud deployment, try downloading to a writable directory
            import os
            try:
                os.makedirs('/tmp/nltk_data', exist_ok=True)
                nltk.data.path.append('/tmp/nltk_data')
                nltk.download('punkt', download_dir='/tmp/nltk_data', quiet=True)
            except:
                nltk_available = False
except ImportError:
    missing_packages.append("nltk")
    nltk_available = False

# Set page config for Streamlit Cloud
st.set_page_config(
    page_title="PDF Q&A Tool - Enhanced",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Initialize NLTK data for cloud deployment
@st.cache_resource
def setup_nltk():
    """Setup NLTK data for cloud deployment."""
    if nltk_available:
        try:
            import os
            # Create NLTK data directory in tmp for cloud deployment
            nltk_data_dir = '/tmp/nltk_data'
            os.makedirs(nltk_data_dir, exist_ok=True)
            
            # Add to NLTK data path
            if nltk_data_dir not in nltk.data.path:
                nltk.data.path.append(nltk_data_dir)
            
            # Download punkt tokenizer
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                nltk.download('punkt', download_dir=nltk_data_dir, quiet=True)
                
            return True
        except Exception as e:
            st.warning(f"NLTK setup warning: {e}")
            return False
    return False

# Setup NLTK for cloud
if nltk_available:
    setup_nltk()

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #2e7d32, #4caf50);
        color: white;
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .answer-box {
        background: #e8f5e8;
        border: 1px solid #4caf50;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 4px solid #2e7d32;
    }
    .confidence-high {
        background: #4caf50;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-size: 0.8rem;
        font-weight: bold;
        display: inline-block;
        margin-bottom: 0.5rem;
    }
    .confidence-medium {
        background: #ff9800;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-size: 0.8rem;
        font-weight: bold;
        display: inline-block;
        margin-bottom: 0.5rem;
    }
    .confidence-low {
        background: #f44336;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-size: 0.8rem;
        font-weight: bold;
        display: inline-block;
        margin-bottom: 0.5rem;
    }
    .context-box {
        background: #f5f5f5;
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        font-style: italic;
    }
    .search-result {
        background: #f0f7ff;
        border: 1px solid #1f77b4;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        border-left: 4px solid #1f77b4;
    }
    .status-success {
        background: #e8f5e8;
        color: #2e7d32;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #4caf50;
        margin: 1rem 0;
    }
    .warning-box {
        background: #fff3cd;
        color: #856404;
        border: 1px solid #ffeaa7;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        border-left: 4px solid #f39c12;
    }
    .info-box {
        background: #e3f2fd;
        border: 1px solid #1f77b4;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .stButton > button {
        background: #2e7d32;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 2rem;
    }
    /* Hide default streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def extract_text_from_pdf(pdf_file) -> str:
    """Extract text from PDF file with caching for cloud performance."""
    if PyPDF2 is None:
        return ""
    
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return ""

def clean_text(text: str) -> str:
    """Clean and normalize text."""
    # Remove extra whitespace and normalize
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters but keep punctuation
    text = re.sub(r'[^\w\s.,!?;:()\-\'""]', ' ', text)
    return text.strip()

@st.cache_data
def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences using NLTK or simple regex with caching."""
    if nltk_available:
        try:
            sentences = sent_tokenize(text)
        except:
            # Fallback to simple regex
            sentences = re.split(r'(?<=[.!?])\s+', text)
    else:
        # Simple sentence splitting
        sentences = re.split(r'(?<=[.!?])\s+', text)
    
    # Clean and filter sentences
    cleaned_sentences = []
    for sentence in sentences:
        sentence = clean_text(sentence)
        if len(sentence) > 20 and len(sentence) < 500:  # Filter very short/long sentences
            cleaned_sentences.append(sentence)
    
    return cleaned_sentences

def chunk_text_advanced(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """Advanced text chunking with overlap."""
    if not text.strip():
        return []
    
    sentences = split_into_sentences(text)
    
    chunks = []
    current_chunk = ""
    current_sentences = []
    
    for sentence in sentences:
        # If adding this sentence would make chunk too long
        if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            
            # Create overlap by keeping last few sentences
            overlap_text = ""
            overlap_chars = 0
            for s in reversed(current_sentences):
                if overlap_chars + len(s) <= overlap:
                    overlap_text = s + " " + overlap_text
                    overlap_chars += len(s)
                else:
                    break
            
            current_chunk = overlap_text + sentence
            current_sentences = [sentence]
        else:
            current_chunk += " " + sentence
            current_sentences.append(sentence)
    
    # Add final chunk
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return [chunk for chunk in chunks if len(chunk) > 100]

def preprocess_for_matching(text: str) -> str:
    """Preprocess text for matching."""
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

class EnhancedQASystem:
    """Enhanced Q&A system with extractive answering."""
    
    def __init__(self, text: str):
        self.original_text = text
        self.sentences = split_into_sentences(text)
        self.chunks = chunk_text_advanced(text)
        self.use_tfidf = sklearn_available
        
        if self.use_tfidf and self.sentences:
            try:
                # Prepare sentence-level vectors for precise matching
                processed_sentences = [preprocess_for_matching(s) for s in self.sentences]
                self.sentence_vectorizer = TfidfVectorizer(
                    max_features=5000,
                    stop_words='english',
                    ngram_range=(1, 3),
                    min_df=1,
                    max_df=0.95
                )
                self.sentence_vectors = self.sentence_vectorizer.fit_transform(processed_sentences)
                
                # Prepare chunk-level vectors for context
                processed_chunks = [preprocess_for_matching(c) for c in self.chunks]
                self.chunk_vectorizer = TfidfVectorizer(
                    max_features=3000,
                    stop_words='english',
                    ngram_range=(1, 2),
                    min_df=1,
                    max_df=0.95
                )
                self.chunk_vectors = self.chunk_vectorizer.fit_transform(processed_chunks)
                
            except Exception as e:
                self.use_tfidf = False
    
    def answer_question(self, question: str) -> Dict:
        """Answer a question using extractive QA."""
        if self.use_tfidf:
            return self._extractive_answer_tfidf(question)
        else:
            return self._extractive_answer_simple(question)
    
    def _extractive_answer_tfidf(self, question: str) -> Dict:
        """TF-IDF based extractive answering with improved answer construction."""
        try:
            processed_question = preprocess_for_matching(question)
            
            # Find best matching sentences
            question_vector = self.sentence_vectorizer.transform([processed_question])
            sentence_similarities = cosine_similarity(question_vector, self.sentence_vectors).flatten()
            
            # Get top sentences with better scoring
            top_sentence_indices = np.argsort(sentence_similarities)[::-1][:10]
            
            # Find best matching chunks for comprehensive context
            chunk_question_vector = self.chunk_vectorizer.transform([processed_question])
            chunk_similarities = cosine_similarity(chunk_question_vector, self.chunk_vectors).flatten()
            best_chunk_idx = np.argmax(chunk_similarities)
            
            # Construct comprehensive answer
            best_sentence_idx = top_sentence_indices[0]
            primary_sentence = self.sentences[best_sentence_idx]
            confidence_score = sentence_similarities[best_sentence_idx]
            
            # Build comprehensive answer by combining related sentences
            answer_parts = [primary_sentence]
            
            # Add closely related sentences if they're highly relevant
            for idx in top_sentence_indices[1:4]:  # Check next 3 best matches
                if sentence_similarities[idx] > max(0.3, confidence_score * 0.6):
                    candidate_sentence = self.sentences[idx]
                    # Avoid duplicate or very similar content
                    if not any(self._sentences_too_similar(candidate_sentence, existing) for existing in answer_parts):
                        answer_parts.append(candidate_sentence)
            
            # Create comprehensive answer
            if len(answer_parts) > 1:
                comprehensive_answer = " ".join(answer_parts)
            else:
                comprehensive_answer = primary_sentence
            
            # Get extended context window
            context_start = max(0, best_sentence_idx - 3)
            context_end = min(len(self.sentences), best_sentence_idx + 4)
            extended_context = " ".join(self.sentences[context_start:context_end])
            
            return {
                'answer': comprehensive_answer,
                'confidence': confidence_score,
                'context': extended_context,
                'source_chunk': self.chunks[best_chunk_idx] if best_chunk_idx < len(self.chunks) else "",
                'method': 'TF-IDF Extractive Enhanced',
                'answer_parts': len(answer_parts)
            }
            
        except Exception as e:
            return self._extractive_answer_simple(question)
    
    def _extractive_answer_simple(self, question: str) -> Dict:
        """Enhanced keyword-based extractive answering with answer construction."""
        question_words = set(preprocess_for_matching(question).split())
        question_lower = question.lower()
        
        # Identify question type for better answer selection
        question_type = self._identify_question_type(question_lower)
        
        sentence_scores = []
        
        # Score each sentence with enhanced criteria
        for i, sentence in enumerate(self.sentences):
            sentence_words = set(preprocess_for_matching(sentence).split())
            sentence_lower = sentence.lower()
            
            if question_words and sentence_words:
                # Basic overlap score
                overlap = len(question_words.intersection(sentence_words))
                base_score = overlap / len(question_words)
                
                # Bonus scoring based on question type
                bonus_score = 0
                
                # Question type bonuses
                if question_type == 'what' and any(word in sentence_lower for word in ['is', 'are', 'means', 'refers', 'definition', 'purpose']):
                    bonus_score += 0.2
                elif question_type == 'how' and any(word in sentence_lower for word in ['method', 'process', 'way', 'approach', 'procedure']):
                    bonus_score += 0.2
                elif question_type == 'why' and any(word in sentence_lower for word in ['because', 'reason', 'due to', 'purpose', 'cause']):
                    bonus_score += 0.2
                elif question_type == 'when' and any(word in sentence_lower for word in ['date', 'time', 'year', 'month', 'during', 'period']):
                    bonus_score += 0.2
                elif question_type == 'where' and any(word in sentence_lower for word in ['location', 'place', 'region', 'area', 'country', 'city']):
                    bonus_score += 0.2
                
                # Length penalty for very short or very long sentences
                length_penalty = 0
                word_count = len(sentence.split())
                if word_count < 8:
                    length_penalty = 0.1
                elif word_count > 50:
                    length_penalty = 0.05
                
                final_score = base_score + bonus_score - length_penalty
                sentence_scores.append((i, sentence, final_score))
        
        # Sort by score
        sentence_scores.sort(key=lambda x: x[2], reverse=True)
        
        # Build comprehensive answer
        answer_parts = []
        used_sentences = set()
        
        # Add best sentences that complement each other
        for idx, sentence, score in sentence_scores[:5]:
            if score > 0.1 and idx not in used_sentences:  # Minimum relevance threshold
                # Check if this sentence adds new information
                if not answer_parts or not self._sentences_too_similar(sentence, " ".join(answer_parts)):
                    answer_parts.append(sentence)
                    used_sentences.add(idx)
                    
                    # Stop if we have enough content or confidence is low
                    if len(answer_parts) >= 3 or (len(answer_parts) >= 2 and score < 0.3):
                        break
        
        if not answer_parts:
            return {
                'answer': "I couldn't find a specific answer to your question in the document.",
                'confidence': 0.0,
                'context': "",
                'source_chunk': "",
                'method': 'No Match Found'
            }
        
        comprehensive_answer = " ".join(answer_parts)
        best_score = sentence_scores[0][2] if sentence_scores else 0
        
        # Get context around the best matching sentence
        best_idx = sentence_scores[0][0] if sentence_scores else 0
        context_start = max(0, best_idx - 3)
        context_end = min(len(self.sentences), best_idx + 4)
        context = " ".join(self.sentences[context_start:context_end])
        
        # Find best chunk for additional context
        chunk_scores = []
        for chunk in self.chunks:
            chunk_words = set(preprocess_for_matching(chunk).split())
            if question_words and chunk_words:
                overlap = len(question_words.intersection(chunk_words))
                score = overlap / len(question_words)
                chunk_scores.append((chunk, score))
        
        best_chunk = ""
        if chunk_scores:
            chunk_scores.sort(key=lambda x: x[1], reverse=True)
            best_chunk = chunk_scores[0][0]
        
        return {
            'answer': comprehensive_answer,
            'confidence': best_score,
            'context': context,
            'source_chunk': best_chunk,
            'method': 'Enhanced Keyword Extractive',
            'answer_parts': len(answer_parts)
        }
    
    def _identify_question_type(self, question_lower: str) -> str:
        """Identify the type of question being asked."""
        if question_lower.startswith('what'):
            return 'what'
        elif question_lower.startswith('how'):
            return 'how'
        elif question_lower.startswith('why'):
            return 'why'
        elif question_lower.startswith('when'):
            return 'when'
        elif question_lower.startswith('where'):
            return 'where'
        elif question_lower.startswith('who'):
            return 'who'
        elif question_lower.startswith('which'):
            return 'which'
        else:
            return 'general'
    
    def _sentences_too_similar(self, sent1: str, sent2: str, threshold: float = 0.7) -> bool:
        """Check if two sentences are too similar to avoid redundancy."""
        words1 = set(preprocess_for_matching(sent1).split())
        words2 = set(preprocess_for_matching(sent2).split())
        
        if not words1 or not words2:
            return False
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        jaccard_similarity = intersection / union if union > 0 else 0
        return jaccard_similarity > threshold
    
    def search_chunks(self, query: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """Search for relevant chunks (for fallback)."""
        if self.use_tfidf:
            try:
                processed_query = preprocess_for_matching(query)
                query_vector = self.chunk_vectorizer.transform([processed_query])
                similarities = cosine_similarity(query_vector, self.chunk_vectors).flatten()
                
                top_indices = np.argsort(similarities)[::-1][:top_k]
                return [(self.chunks[idx], similarities[idx]) for idx in top_indices]
            except:
                pass
        
        # Fallback to keyword search
        query_words = set(preprocess_for_matching(query).split())
        chunk_scores = []
        
        for chunk in self.chunks:
            chunk_words = set(preprocess_for_matching(chunk).split())
            if query_words and chunk_words:
                overlap = len(query_words.intersection(chunk_words))
                score = overlap / len(query_words)
                chunk_scores.append((chunk, score))
        
        chunk_scores.sort(key=lambda x: x[1], reverse=True)
        return chunk_scores[:top_k]

def get_confidence_level(score: float) -> Tuple[str, str]:
    """Get confidence level and CSS class with adjusted thresholds."""
    if score >= 0.4:
        return "High Confidence", "confidence-high"
    elif score >= 0.2:
        return "Medium Confidence", "confidence-medium"
    else:
        return "Low Confidence", "confidence-low"

def extract_keywords(text: str, top_n: int = 15) -> List[str]:
    """Extract key terms from text."""
    words = preprocess_for_matching(text).split()
    word_freq = {}
    
    # Count word frequencies, excluding very short words and common terms
    stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'this', 'that', 'these', 'those', 'a', 'an'}
    
    for word in words:
        if len(word) > 3 and word not in stop_words:
            word_freq[word] = word_freq.get(word, 0) + 1
    
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    return [word for word, freq in sorted_words[:top_n]]

def main():
    # Check for missing packages
    if missing_packages:
        st.markdown("""
        <div class="main-header">
            <h1>🤖 PDF Q&A Tool - Enhanced</h1>
            <p>Missing Required Dependencies</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="warning-box">
            ⚠️ <strong>Missing packages:</strong> {', '.join(missing_packages)}<br><br>
            Please install the required packages from requirements.txt
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Initialize session state
    if "qa_results" not in st.session_state:
        st.session_state.qa_results = []
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🤖 Free PDF Q&A Tool - Enhanced</h1>
        <p>Get precise answers from your PDF documents - No API key required!</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        🆓 <strong>Completely Free Enhanced Q&A!</strong> This tool provides direct answers extracted from your PDF content without hallucinations.
        Ask specific questions and get precise, contextual answers.
    </div>
    """, unsafe_allow_html=True)
    
    # File upload
    st.subheader("📁 Upload PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    if uploaded_file is not None:
        # Process PDF only if it's a new file
        if "current_pdf_qa" not in st.session_state or st.session_state.current_pdf_qa != uploaded_file.name:
            with st.spinner("Processing PDF for Q&A..."):
                # Use file hash for better caching in cloud
                file_hash = hash(uploaded_file.read())
                uploaded_file.seek(0)  # Reset file pointer
                
                pdf_text = extract_text_from_pdf(uploaded_file)
                
                if not pdf_text.strip():
                    st.error("Could not extract text from PDF. Please ensure the PDF contains readable text.")
                    return
                
                # Show progress for large documents
                if len(pdf_text) > 50000:
                    progress_bar = st.progress(0)
                    progress_bar.progress(0.3, "Creating Q&A system...")
                
                qa_system = EnhancedQASystem(pdf_text)
                
                if len(pdf_text) > 50000:
                    progress_bar.progress(0.7, "Extracting keywords...")
                
                if not qa_system.sentences:
                    st.error("Could not process text into sentences.")
                    return
                
                keywords = extract_keywords(pdf_text, 20)
                
                if len(pdf_text) > 50000:
                    progress_bar.progress(1.0, "Complete!")
                    progress_bar.empty()
            
            # Store in session state with file hash for cloud optimization
            st.session_state.qa_system = qa_system
            st.session_state.current_pdf_qa = uploaded_file.name
            st.session_state.file_hash = file_hash
            st.session_state.keywords_qa = keywords
            st.session_state.qa_results = []
            
            method = "TF-IDF Extractive QA" if sklearn_available else "Keyword Extractive QA"
            st.markdown(f"""
            <div class="status-success">
                ✅ PDF processed for Q&A!<br>
                📊 Processed {len(qa_system.sentences)} sentences and {len(qa_system.chunks)} chunks<br>
                🤖 Using {method}<br>
                📝 Document length: {len(pdf_text):,} characters
            </div>
            """, unsafe_allow_html=True)
            
            # Show key topics
            if keywords:
                st.markdown("**🏷️ Key topics in your document:**")
                keyword_display = " • ".join(keywords[:12])
                st.markdown(f"*{keyword_display}*")
        
        st.markdown("---")
        
        # Q&A Interface
        st.subheader("🤖 Ask Questions About Your Document")
        
        # Question form
        with st.form(key="qa_form", clear_on_submit=True):
            col1, col2 = st.columns([4, 1])
            
            with col1:
                question = st.text_input(
                    "Ask a specific question:",
                    placeholder="e.g., What is the main conclusion? How does the methodology work? What are the key findings?",
                    key="question_input"
                )
            
            with col2:
                ask_button = st.form_submit_button("🤖 Ask")
                clear_qa_button = st.form_submit_button("🗑️ Clear")
        
        # Process question
        if ask_button and question.strip():
            with st.spinner("🤖 Analyzing document for answer..."):
                result = st.session_state.qa_system.answer_question(question)
                st.session_state.qa_results.append((question, result))
        
        # Clear results
        if clear_qa_button:
            st.session_state.qa_results = []
            st.success("🗑️ Q&A history cleared!")
        
        # Display Q&A results
        if st.session_state.qa_results:
            st.subheader("💬 Q&A Results")
            
            for i, (q, result) in enumerate(reversed(st.session_state.qa_results)):
                confidence_text, confidence_class = get_confidence_level(result['confidence'])
                
                # Enhanced answer display with better formatting
                answer_display = result['answer']
                if len(answer_display) < 50 and result.get('source_chunk'):
                    # If answer is very short, try to provide more context from source chunk
                    source_sentences = split_into_sentences(result['source_chunk'])
                    for sent in source_sentences[:3]:
                        if any(word in sent.lower() for word in q.lower().split() if len(word) > 3):
                            answer_display = sent
                            break
                
                st.markdown(f"""
                <div class="answer-box">
                    <div class="{confidence_class}">{confidence_text} ({result['confidence']:.1%})</div>
                    <strong>❓ Question:</strong> {q}<br><br>
                    <strong>💡 Answer:</strong><br>
                    {answer_display}
                    {f"<br><br><small><em>Combined from {result.get('answer_parts', 1)} relevant sections</em></small>" if result.get('answer_parts', 1) > 1 else ""}
                </div>
                """, unsafe_allow_html=True)
                
                # Show context if different from answer
                if result['context'] and result['context'] != result['answer']:
                    with st.expander(f"📖 Context for Question {len(st.session_state.qa_results)-i}"):
                        st.markdown(f"""
                        <div class="context-box">
                            <strong>Surrounding context:</strong><br>
                            {result['context']}
                        </div>
                        """, unsafe_allow_html=True)
                
                # Show source chunk for very low confidence
                if result['confidence'] < 0.2 and result['source_chunk']:
                    with st.expander(f"🔍 Source Material for Question {len(st.session_state.qa_results)-i}"):
                        st.write(result['source_chunk'][:800] + "..." if len(result['source_chunk']) > 800 else result['source_chunk'])
        
        # Search fallback
        st.markdown("---")
        st.subheader("🔍 Or Search Document Sections")
        
        with st.form(key="search_form", clear_on_submit=True):
            col1, col2 = st.columns([4, 1])
            
            with col1:
                search_query = st.text_input(
                    "Search for topics or keywords:",
                    placeholder="e.g., methodology, results, conclusion...",
                    key="search_input"
                )
            
            with col2:
                search_button = st.form_submit_button("🔍 Search")
        
        if search_button and search_query.strip():
            with st.spinner("🔍 Searching document..."):
                search_results = st.session_state.qa_system.search_chunks(search_query, 3)
                
                st.subheader(f"📋 Search Results for: '{search_query}'")
                
                for i, (chunk, score) in enumerate(search_results):
                    if score > 0:
                        st.markdown(f"""
                        <div class="search-result">
                            <strong>Section {i+1} (Relevance: {score:.1%}):</strong><br>
                            {chunk[:400]}{'...' if len(chunk) > 400 else ''}
                        </div>
                        """, unsafe_allow_html=True)
                        
                        if len(chunk) > 400:
                            with st.expander(f"📖 Read full Section {i+1}"):
                                st.write(chunk)
        
        # Usage tips
        with st.expander("💡 Q&A Tips"):
            st.markdown("""
            **How to ask better questions:**
            - ✅ **Specific questions**: "What is the main conclusion?" instead of "Tell me about this"
            - ✅ **Direct questions**: "How does X work?" "What are the benefits of Y?"
            - ✅ **Factual questions**: "What methodology was used?" "What were the results?"
            - ✅ **Complete questions**: Include context words that might appear in the answer
            
            **Question types that work well:**
            - **What**: What is the purpose? What are the findings? What does X mean?
            - **How**: How does it work? How was it measured? How do you do X?
            - **When**: When was this conducted? When should this be applied?
            - **Why**: Why is this important? Why was this approach chosen?
            - **Where**: Where was this study done? Where is this applicable?
            - **Who**: Who conducted this? Who is responsible for X?
            
            **Tips for better answers:**
            - 🎯 Use keywords that likely appear in the answer
            - 📝 Ask one clear question at a time
            - 🔍 If confidence is low, try rephrasing or check search results
            - 📖 Review context section for additional details
            
            **Confidence levels explained:**
            - 🟢 **High (40%+)**: Very relevant answer found
            - 🟡 **Medium (20-40%)**: Somewhat relevant answer found  
            - 🔴 **Low (<20%)**: Limited relevance, try rephrasing or search instead
            """)
        
        # Document statistics
        if hasattr(st.session_state, 'qa_system'):
            qa_sys = st.session_state.qa_system
            st.markdown(f"""
            <div class="info-box">
                📊 <strong>Document Analysis:</strong> 
                {len(qa_sys.sentences)} sentences • 
                {len(qa_sys.chunks)} chunks • 
                {len(st.session_state.keywords_qa)} key topics •
                Method: {qa_sys._extractive_answer_tfidf.__doc__.split('.')[0] if qa_sys.use_tfidf else 'Keyword Extractive'}
            </div>
            """, unsafe_allow_html=True)
    
    else:
        # Upload prompt
        st.markdown("""
        ### 📤 Upload Your PDF Document
        
        **Enhanced Q&A Features:**
        - 🤖 **Direct Answers**: Get specific answers to your questions, not just search results
        - 🎯 **No Hallucinations**: All answers are extracted directly from your document
        - 📊 **Confidence Scores**: Know how reliable each answer is
        - 📖 **Context Provided**: See surrounding text for better understanding
        - 🔍 **Fallback Search**: Traditional search when Q&A confidence is low
        - ☁️ **Cloud Optimized**: Fast processing with smart caching
        
        **Perfect for:**
        - Research papers: "What is the main hypothesis?"
        - Reports: "What are the key recommendations?"
        - Manuals: "How do I configure X?"
        - Legal documents: "What are the terms for Y?"
        - Academic papers: "What methodology was used?"
        
        **Completely FREE with no API keys required!**
        
        **📋 Supported file types:** PDF files with readable text
        **⚡ Processing:** Optimized for Streamlit Cloud deployment
        """)

if __name__ == "__main__":
    main()
