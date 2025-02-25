"""
vLLM Document RAG Integration Tool for OpenWebUI (Advanced)
=================================================================

This tool enables document retrieval using vLLM embeddings and Pinecone
within the OpenWebui environment, providing a seamless interface for RAG operations
with all types of document content (PowerPoint, PDF, Word, etc.).

License: Apache 2.0

Requirements:
    - Python 3.8+
    - OpenWebui
    - vLLM
    - Pinecone
    - Requests
"""

# Tool version
__version__ = "2.0.0"
__description__ = "vLLM Document RAG integration for OpenWebUI"

import os
import requests
import json
import traceback
import re
import math
import heapq
from collections import Counter, defaultdict
from io import StringIO
from pydantic import BaseModel, Field
from typing import Callable, Any, Literal, Union, List, Dict, Optional, Tuple, Set
import pinecone  # Import for Pinecone vector database

# -------------------------------
# Event Emitter Classes
# -------------------------------

class StatusEventEmitter:
    def __init__(self, event_emitter: Callable[[dict], Any] = None):
        self.event_emitter = event_emitter

    async def emit(self, description="Unknown State", status="in_progress", done=False):
        if self.event_emitter:
            await self.event_emitter(
                {
                    "type": "status",
                    "data": {
                        "status": status,
                        "description": description,
                        "done": done,
                    },
                }
            )


class MessageEventEmitter:
    def __init__(self, event_emitter: Callable[[dict], Any] = None):
        self.event_emitter = event_emitter

    async def emit(self, content="Some message"):
        if self.event_emitter:
            await self.event_emitter(
                {
                    "type": "message",
                    "data": {
                        "content": content,
                    },
                }
            )

# -------------------------------
# Main Tool Class
# -------------------------------

class Tools:
    """
    OpenWebUI Tool for Document RAG with vLLM and Pinecone
    
    This tool provides retrieval-augmented generation capabilities for all document types
    using vLLM embeddings and Pinecone vector database. It allows the chat model to
    search through documents and provide answers based on their content.
    
    Features:
    - Semantic search using vLLM embeddings
    - Keyword-based filtering with BM25
    - Context-aware retrieval with document chunking
    - Rich metadata extraction
    - Works with PowerPoint, PDF, Word, and other document types
    """
    class Valves(BaseModel):
        VLLM_EMBEDDINGS_URL: str = Field(
            default="http://vllm-container2:8001/v1/embeddings",
            description="The URL for the vLLM embedding API endpoint",
        )
        PINECONE_API_KEY: str = Field(
            default="pcsk_2yYsZE_KG5KrrY9CWrP694FTCnsLMatAVXrynUShvqn8hsRQKSMpzgj6TXE7JVHY5g287r",
            description="Pinecone API key for authentication",
        )
        PINECONE_ENVIRONMENT: str = Field(
            default="us-east-1", #adjust to your region
            description="Pinecone environment to use",
        )
        PINECONE_INDEX: str = Field(
            default="openwebui-experimental", #adjust to your index name
            description="Name of the Pinecone index to query",
        )
        PINECONE_NAMESPACE: str = Field(
            default="ns1", #adjust to your namespace
            description="Pinecone namespace where slide embeddings are stored",
        )
        TOP_K: int = Field(
            default=3,
            description="Number of documents to retrieve from vector database",
        )
        TOP_K_SEMANTIC: int = Field(
            default=5,
            description="Number of documents to retrieve using semantic search",
        )
        TOP_K_KEYWORD: int = Field(
            default=5,
            description="Number of documents to retrieve using keyword search",
        )
        SIMILARITY_THRESHOLD: float = Field(
            default=0.7,
            description="Minimum similarity score to include a document (0.0 to 1.0)",
        )
        EMBEDDING_MODEL: str = Field(
            default="Alibaba-NLP/gte-Qwen2-7B-instruct",
            description="The embedding model to use for encoding queries",
        )
        KEYWORD_SEARCH_WEIGHT: float = Field(
            default=0.3,
            description="Weight of keyword match in hybrid search (0.0 to 1.0)",
        )
        INCLUDE_CONTEXT_SLIDES: bool = Field(
            default=True,
            description="Include adjacent slides in context when available",
        )
        INCLUDE_CONTEXT_CHUNKS: bool = Field(
            default=True,
            description="Include adjacent chunks in context when available",
        )
        CONTEXT_PREVIEW_LENGTH: int = Field(
            default=300,
            description="Length of content preview to include for context slides",
        )
        DEBUG_MODE: bool = Field(
            default=False,
            description="If True, debugging information will be emitted",
        )
        INCLUDE_METADATA: bool = Field(
            default=True,
            description="If True, include metadata in the retrieved documents",
        )
        FORMAT_TEMPLATE: str = Field(
            default="### Source {index}: {doc_type} - {file_name} (Section {section_id})\n\n{content_preview}\n\n**Keywords**: {keywords}\n\n**Source Link**: {source_link}\n\n",
            description="Template for formatting each retrieved document chunk",
        )
        USE_BM25: bool = Field(
            default=True,
            description="If True, use BM25 for keyword search instead of embeddings",
        )
        BM25_K1: float = Field(
            default=1.2,
            description="BM25 term frequency saturation parameter",
        )
        BM25_B: float = Field(
            default=0.75,
            description="BM25 length normalization factor",
        )

    def __init__(self):
        self.valves = self.Valves()
        self.headers = {
            "Content-Type": "application/json",
            "User-Agent": "vLLM-RAG-Tool/1.0",
        }
        self.pinecone_client = None

    # -------------------------------
    # Pinecone Connection Methods
    # -------------------------------

    async def setup_pinecone(self):
        """Initialize the Pinecone client if not already initialized."""
        try:
            if self.pinecone_client is None:
                pinecone.init(
                    api_key=self.valves.PINECONE_API_KEY,
                    environment=self.valves.PINECONE_ENVIRONMENT
                )
                self.pinecone_client = pinecone.Index(self.valves.PINECONE_INDEX)
                return True
            return True
        except Exception as e:
            return False, f"Failed to initialize Pinecone: {str(e)}"

    # -------------------------------
    # Embedding & Vector Methods
    # -------------------------------

    async def get_embeddings(self, text: str) -> List[float]:
        """
        Generate embeddings for the input text using the vLLM API.
        
        Args:
            text: The text to embed
            
        Returns:
            List of float values representing the embedding vector
        """
        payload = {
            "input": text,
            "model": self.valves.EMBEDDING_MODEL,
        }
        
        try:
            response = requests.post(
                self.valves.VLLM_EMBEDDINGS_URL,
                json=payload,
                headers=self.headers,
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            
            # Extract the embedding from the response
            if "data" in data and len(data["data"]) > 0:
                return data["data"][0]["embedding"]
            else:
                raise ValueError("No embedding found in response")
        except Exception as e:
            raise Exception(f"Error generating embeddings: {str(e)}")
            
    async def extract_keywords_from_query(self, query: str) -> List[str]:
        """
        Extract potential keywords from a user query.
        
        Args:
            query: The user query
            
        Returns:
            List of extracted keywords
        """
        # Simple keyword extraction using stopword filtering
        stopwords = {'a', 'an', 'the', 'and', 'or', 'but', 'if', 'because', 'as', 'what', 
                    'which', 'this', 'that', 'these', 'those', 'then', 'just', 'so', 'than', 'such',
                    'when', 'who', 'how', 'where', 'why', 'is', 'are', 'was', 'were', 'be', 'been',
                    'have', 'has', 'had', 'do', 'does', 'did', 'but', 'at', 'by', 'with', 'from',
                    'here', 'there', 'to', 'of', 'for', 'in', 'on', 'about', 'into', 'over', 'after'}
        
        # Normalize and tokenize
        query = query.lower()
        import re
        query = re.sub(r'[^\w\s]', ' ', query)
        words = query.split()
        
        # Filter and keep meaningful words
        keywords = [word for word in words if word not in stopwords and len(word) > 3]
        
        # Extract potential bigrams (two consecutive words)
        bigrams = []
        for i in range(len(words) - 1):
            if (words[i] not in stopwords and words[i+1] not in stopwords and 
                len(words[i]) > 2 and len(words[i+1]) > 2):
                bigrams.append(f"{words[i]} {words[i+1]}")
        
        # Combine and limit
        all_keywords = keywords + bigrams
        return all_keywords[:10]  # Limit to 10 keywords

    # -------------------------------
    # BM25 Implementation
    # -------------------------------
    
    class BM25:
        """BM25 implementation for keyword-based ranking."""
        
        def __init__(self, k1=1.2, b=0.75):
            """Initialize BM25 with parameters.
            
            Args:
                k1: Term frequency saturation parameter
                b: Length normalization factor
            """
            self.k1 = k1
            self.b = b
            self.corpus_size = 0
            self.avg_doc_len = 0
            self.doc_freqs = []
            self.idf = {}
            self.doc_len = []
            self.tokenized_docs = []
        
        def tokenize(self, text):
            """Simple tokenization for BM25."""
            # Convert to lowercase and split on non-alphanumeric
            text = text.lower()
            tokens = re.findall(r'\w+', text)
            return [t for t in tokens if len(t) > 2]  # Filter out very short tokens
        
        def fit(self, corpus):
            """Fit BM25 parameters on a corpus of documents."""
            self.tokenized_docs = [self.tokenize(doc) for doc in corpus]
            self.corpus_size = len(self.tokenized_docs)
            
            # Calculate document frequencies
            df = Counter()
            self.doc_len = []
            
            for doc in self.tokenized_docs:
                self.doc_len.append(len(doc))
                df.update(set(doc))  # Count each term once per document
            
            # Calculate average document length
            self.avg_doc_len = sum(self.doc_len) / self.corpus_size if self.corpus_size > 0 else 0
            
            # Calculate IDF values
            self.idf = {}
            for term, freq in df.items():
                self.idf[term] = math.log((self.corpus_size - freq + 0.5) / (freq + 0.5) + 1.0)
        
        def get_scores(self, query):
            """Calculate BM25 scores for a query against all documents."""
            query_tokens = self.tokenize(query)
            scores = [0.0] * self.corpus_size
            
            for term in query_tokens:
                if term not in self.idf:
                    continue
                
                for i, doc in enumerate(self.tokenized_docs):
                    if self.doc_len[i] == 0:
                        continue
                        
                    # Count term frequency in document
                    term_freq = doc.count(term)
                    if term_freq == 0:
                        continue
                    
                    # BM25 score calculation
                    doc_len_norm = self.doc_len[i] / self.avg_doc_len
                    term_score = self.idf[term] * (term_freq * (self.k1 + 1)) / (term_freq + self.k1 * (1 - self.b + self.b * doc_len_norm))
                    scores[i] += term_score
            
            return scores
        
        def get_top_n(self, query, documents, n=5):
            """Get the top N documents for a query."""
            if len(documents) == 0:
                return []
                
            # Make sure we have a corpus
            if self.corpus_size == 0:
                self.fit(documents)
            
            # Get scores
            scores = self.get_scores(query)
            
            # Create (score, index) pairs and get top n
            score_index_pairs = [(score, i) for i, score in enumerate(scores)]
            top_n = heapq.nlargest(n, score_index_pairs, key=lambda x: x[0])
            
            # Return list of (document, score) tuples
            return [(documents[i], score) for score, i in top_n]
    
    # -------------------------------
    # Pinecone Query Methods
    # -------------------------------

    async def query_pinecone(self, vector: List[float], namespace: Optional[str] = None, keywords: Optional[List[str]] = None) -> List[Dict]:
        """
        Query Pinecone vector database with the embedding vector and optional keywords.
        
        Args:
            vector: The embedding vector to query with
            namespace: Optional namespace to query within
            keywords: Optional list of keywords to filter results
            
        Returns:
            List of matching slides with their content and metadata
        """
        namespace = namespace or self.valves.PINECONE_NAMESPACE
        
        query_params = {
            "vector": vector,
            "top_k": self.valves.TOP_K * (2 if keywords else 1),  # Get more results if keyword filtering
            "include_metadata": self.valves.INCLUDE_METADATA
        }
        
        if namespace:
            query_params["namespace"] = namespace
            
        query_response = self.pinecone_client.query(**query_params)
        
        results = []
        for match in query_response.matches:
            # Skip if below threshold
            if match.score < self.valves.SIMILARITY_THRESHOLD:
                continue
                
            # Calculate keyword score if keywords are provided
            keyword_match_score = 0
            if keywords and "keywords" in match.metadata:
                for keyword in keywords:
                    if keyword.lower() in match.metadata["keywords"].lower():
                        keyword_match_score += 1
                    if "content_preview" in match.metadata and keyword.lower() in match.metadata["content_preview"].lower():
                        keyword_match_score += 0.5
                        
                # Calculate hybrid score
                hybrid_score = (1 - self.valves.KEYWORD_SEARCH_WEIGHT) * match.score + \
                               self.valves.KEYWORD_SEARCH_WEIGHT * (keyword_match_score / len(keywords))
            else:
                hybrid_score = match.score
            
            # Parse the ID to extract file and slide info
            id_parts = match.id.split('_')
            file_name = '_'.join(id_parts[:-1]) if len(id_parts) > 1 else match.id
            
            doc = {
                "id": match.id,
                "score": match.score,
                "hybrid_score": hybrid_score,
                "metadata": match.metadata
            }
            
            if keywords:
                doc["keyword_score"] = keyword_match_score / len(keywords) if keywords else 0
                
            results.append(doc)
        
        # If we have keywords, sort by hybrid score
        if keywords:
            results.sort(key=lambda x: x["hybrid_score"], reverse=True)
            results = results[:self.valves.TOP_K]  # Limit to top_k after sorting
                
        return results

    # -------------------------------
    # Output Formatting Methods
    # -------------------------------
        
    async def format_documents(self, documents: List[Dict]) -> str:
        """
        Format the retrieved PowerPoint slides into a single context string.
        
        Args:
            documents: List of document dictionaries
            
        Returns:
            Formatted context string
        """
        context = []
        for i, doc in enumerate(documents):
            metadata = doc.get("metadata", {})
            
            # Extract main slide information
            file_name = metadata.get("file_name", "Unknown")
            slide_number = metadata.get("slide_number", "?")
            content_preview = metadata.get("content_preview", "No content available")
            keywords = metadata.get("keywords", "")
            one_drive_link = metadata.get("one_drive_link", "#")
            
            # Parse the content to extract the main content
            # Different document types might have different section markers
            main_content = ""
            if "content_preview" in metadata:
                content_lines = metadata["content_preview"].split("\n")
                
                # Look for section markers that might exist
                main_section_marker = None
                for possible_marker in ["--- MAIN SLIDE", "--- MAIN CHUNK", "--- MAIN SECTION", "--- MAIN CONTENT"]:
                    for line in content_lines:
                        if line.startswith(possible_marker):
                            main_section_marker = possible_marker
                            break
                    if main_section_marker:
                        break
                
                # Extract the main content based on markers if found
                if main_section_marker:
                    in_main_section = False
                    for line in content_lines:
                        if line.startswith(main_section_marker):
                            in_main_section = True
                            continue
                        elif line.startswith("---"):  # Any other section marker
                            in_main_section = False
                            continue
                        if in_main_section:
                            main_content += line + "\n"
                else:
                    # If no markers, use the whole content
                    main_content = metadata["content_preview"]
            
            # Format the main content
            formatted_doc = f"### Source {i+1}: {metadata.get('doc_type', 'Document')} - {file_name}"
            
            # Handle different document types appropriately
            if "slide_number" in metadata:
                # PowerPoint format
                formatted_doc += f" (Slide {metadata.get('slide_number', '?')})\n\n"
            elif "page_number" in metadata:
                # PDF format
                formatted_doc += f" (Page {metadata.get('page_number', '?')})\n\n"
            elif "section_id" in metadata:
                # Generic section format
                formatted_doc += f" (Section {metadata.get('section_id', '?')})\n\n"
            else:
                formatted_doc += "\n\n"
            
            # Add content
            if main_content.strip():
                formatted_doc += f"{main_content.strip()}\n\n"
            else:
                formatted_doc += f"{content_preview}\n\n"
            
            # Add keywords if available
            if keywords:
                formatted_doc += f"**Keywords**: {keywords}\n\n"
            
            # Add context information if included in metadata
            context_key = None
            if "context_slides" in metadata:
                context_key = "context_slides"
                context_label = "Related Slides"
            elif "context_chunks" in metadata:
                context_key = "context_chunks"
                context_label = "Related Sections"
            elif "context_pages" in metadata:
                context_key = "context_pages"
                context_label = "Related Pages"
            
            if self.valves.INCLUDE_CONTEXT_CHUNKS and context_key and context_key in metadata:
                context_items = metadata[context_key]
                if context_items:
                    formatted_doc += f"**{context_label}**: "
                    if isinstance(context_items, list):
                        item_labels = []
                        for item in context_items:
                            if isinstance(item, (int, str)):
                                item_labels.append(f"{item}")
                            elif isinstance(item, dict) and "id" in item:
                                item_labels.append(f"{item['id']}")
                        formatted_doc += ", ".join(item_labels)
                    elif isinstance(context_items, str):
                        # Try to parse JSON string
                        try:
                            import json
                            items = json.loads(context_items)
                            if isinstance(items, list):
                                formatted_doc += ", ".join([str(item) for item in items])
                        except:
                            # Just use as is
                            formatted_doc += context_items
                    formatted_doc += "\n\n"
            
            # Extract source link and make it easily accessible for citation
            source_link = metadata.get('source_link', metadata.get('one_drive_link', "#"))
            
            # Add citation-friendly identifier (document title, slide/page number)
            citation_identifier = ""
            if "slide_number" in metadata:
                citation_identifier = f"Slide {metadata.get('slide_number', '?')}"
            elif "page_number" in metadata:
                citation_identifier = f"Page {metadata.get('page_number', '?')}"
            elif "section_id" in metadata:
                citation_identifier = f"Section {metadata.get('section_id', '?')}"
                
            # Add source link with proper formatting for citation
            formatted_doc += f"**Source Link**: {source_link}\n"
            formatted_doc += f"**Citation**: [Source {i+1}]({source_link}) ({citation_identifier})\n\n"
            
            context.append(formatted_doc)
            
        return "\n".join(context)
        
    async def format_rag_prompt(self, query: str, context: str) -> str:
        """
        Format a prompt for the vLLM chat model that includes the retrieved context.
        
        Args:
            query: The original user query
            context: The retrieved context from relevant documents
            
        Returns:
            Formatted prompt for the vLLM chat model
        """
        prompt = f"""You are an assistant that answers questions based on document content.
        
Below is information retrieved from relevant documents:

{context}

Answer the following question using ONLY the information provided above. If you cannot answer the question based on the provided information, say so clearly but suggest what might be relevant based on the available documents.

CITATION INSTRUCTIONS:
1. When referencing information, cite the specific source by including the source number in [brackets].
2. For important references, include the direct link to the source when first mentioning it.
3. Use this format for citations with links: "According to [Source 2](link from Source 2), the project deadline is March 15th."
4. For subsequent mentions of the same source, you can use just the bracket notation: "As mentioned in [Source 2], the team consists of 5 people."
5. When referring to specific slides or pages, mention them: "On slide 4 of [Source 1], it states that..."

Be precise in your answer and ensure all key information is properly cited with the appropriate source.

Question: {query}

Answer:"""
        return prompt

    # -------------------------------
    # Main API Methods
    # -------------------------------
    
    async def advanced_rag_query(
        self,
        query: str,
        namespace: Optional[str] = None,
        __event_emitter__: Callable[[dict], Any] = None,
    ) -> List[Dict]:
        """
        Advanced RAG implementation that combines semantic search and keyword search (BM25)
        in a retrieve-then-rerank approach.
        
        Args:
            query: The user query
            namespace: Optional Pinecone namespace to query within
            __event_emitter__: Event emitter for status updates
            
        Returns:
            List of ranked document dictionaries
        """
        self.status_emitter = StatusEventEmitter(__event_emitter__)
        self.message_emitter = MessageEventEmitter(__event_emitter__)
        
        try:
            # Setup and initialize
            await self.status_emitter.emit("Initializing advanced RAG query...")
            
            namespace = namespace or self.valves.PINECONE_NAMESPACE
            
            # Debug mode information
            if self.valves.DEBUG_MODE:
                await self.message_emitter.emit(
                    f"#### Advanced RAG Query\n{query}\n"
                )
            
            # Initialize Pinecone
            await self.status_emitter.emit("Connecting to Pinecone...")
            pinecone_setup = await self.setup_pinecone()
            if not isinstance(pinecone_setup, bool):
                raise Exception(f"Pinecone initialization failed: {pinecone_setup[1]}")
            
            # 1. Semantic Search: Get documents based on vector similarity
            await self.status_emitter.emit("Performing semantic search...")
            embedding = await self.get_embeddings(query)
            
            semantic_query_params = {
                "vector": embedding,
                "top_k": self.valves.TOP_K_SEMANTIC,
                "include_metadata": True,
                "namespace": namespace
            }
            
            semantic_results = self.pinecone_client.query(**semantic_query_params)
            
            # Process semantic results
            semantic_docs = []
            for match in semantic_results.matches:
                if match.score >= self.valves.SIMILARITY_THRESHOLD * 0.9:  # Slightly lower threshold for combined search
                    semantic_docs.append({
                        "id": match.id,
                        "score": match.score,
                        "semantic_score": match.score,
                        "metadata": match.metadata,
                        "source": "semantic"
                    })
            
            # 2. Keyword Search: Get documents based on keyword matching
            await self.status_emitter.emit("Performing keyword search...")
            
            # Extract keywords from query
            keywords = await self.extract_keywords_from_query(query)
            keyword_string = " ".join(keywords) if keywords else query
            
            if self.valves.DEBUG_MODE and keywords:
                await self.message_emitter.emit(
                    f"#### Extracted Keywords\n{', '.join(keywords)}\n"
                )
            
            keyword_docs = []
            
            if self.valves.USE_BM25:
                # BM25 approach - requires fetching candidate documents first
                await self.status_emitter.emit("Retrieving candidate documents for BM25...")
                
                # Get a broader set of candidates to run BM25 on
                # We'll use a metadata filter to get documents that might match
                fetch_limit = min(100, self.valves.TOP_K_KEYWORD * 3)  # Reasonable limit to avoid too many API calls
                
                # Fetch some document IDs to start with
                stats = self.pinecone_client.describe_index_stats()
                namespace_stats = stats.get('namespaces', {}).get(namespace, {})
                vector_count = namespace_stats.get('vector_count', 0)
                
                # If there aren't too many vectors, we can fetch all metadata
                # Otherwise, we'll use a hybrid approach with a secondary embedding search
                candidate_ids = []
                candidate_docs = []
                candidate_contents = []
                
                if vector_count <= fetch_limit:
                    # Small enough index to fetch all
                    fetch_response = self.pinecone_client.fetch(
                        ids=[], 
                        namespace=namespace, 
                        limit=fetch_limit
                    )
                    
                    for id, vector in fetch_response.get('vectors', {}).items():
                        metadata = vector.get('metadata', {})
                        content = metadata.get('content_preview', '')
                        if content:
                            candidate_ids.append(id)
                            candidate_docs.append({
                                "id": id,
                                "metadata": metadata
                            })
                            candidate_contents.append(content)
                else:
                    # Use a broader semantic search to get candidates
                    broader_params = {
                        "vector": embedding,
                        "top_k": fetch_limit,
                        "include_metadata": True,
                        "namespace": namespace
                    }
                    
                    broader_results = self.pinecone_client.query(**broader_params)
                    
                    for match in broader_results.matches:
                        metadata = match.metadata
                        content = metadata.get('content_preview', '')
                        if content:
                            candidate_ids.append(match.id)
                            candidate_docs.append({
                                "id": match.id,
                                "metadata": metadata
                            })
                            candidate_contents.append(content)
                
                # Now we can run BM25 on these candidates
                if candidate_contents:
                    await self.status_emitter.emit(f"Running BM25 on {len(candidate_contents)} candidate documents...")
                    
                    # Create BM25 instance
                    bm25 = self.BM25(k1=self.valves.BM25_K1, b=self.valves.BM25_B)
                    bm25.fit(candidate_contents)
                    
                    # Get BM25 scores
                    top_bm25_results = bm25.get_top_n(
                        query=keyword_string,
                        documents=candidate_contents,
                        n=self.valves.TOP_K_KEYWORD
                    )
                    
                    # Map results back to documents
                    for content, score in top_bm25_results:
                        content_idx = candidate_contents.index(content)
                        doc = candidate_docs[content_idx]
                        
                        keyword_docs.append({
                            "id": doc["id"],
                            "score": score,  # BM25 score
                            "keyword_score": score,
                            "metadata": doc["metadata"],
                            "source": "keyword_bm25"
                        })
            else:
                # Use embedding-based keyword search as fallback
                keyword_embedding = await self.get_embeddings(keyword_string)
                
                keyword_query_params = {
                    "vector": keyword_embedding,
                    "top_k": self.valves.TOP_K_KEYWORD,
                    "include_metadata": True,
                    "namespace": namespace
                }
                
                keyword_results = self.pinecone_client.query(**keyword_query_params)
                
                # Calculate keyword match scores manually
                for match in keyword_results.matches:
                    if match.score >= self.valves.SIMILARITY_THRESHOLD * 0.8:  # Lower threshold for keywords
                        metadata = match.metadata
                        keyword_match_score = 0
                        
                        # Check keyword matches in metadata fields
                        for keyword in keywords:
                            if "keywords" in metadata and keyword.lower() in metadata["keywords"].lower():
                                keyword_match_score += 1
                            if "content_preview" in metadata and keyword.lower() in metadata["content_preview"].lower():
                                keyword_match_score += 0.5
                        
                        # Normalize score
                        keyword_match_score = keyword_match_score / max(len(keywords), 1)
                        
                        keyword_docs.append({
                            "id": match.id,
                            "score": match.score,
                            "keyword_score": keyword_match_score,
                            "metadata": metadata,
                            "source": "keyword_embedding"
                        })
            
            # 3. Combine and deduplicate results
            await self.status_emitter.emit("Combining search results...")
            
            # Combine both result sets into a single list
            all_docs = semantic_docs + keyword_docs
            
            # Deduplicate by ID, keeping track of sources
            unique_docs = {}
            for doc in all_docs:
                doc_id = doc["id"]
                if doc_id in unique_docs:
                    # Update existing entry with information from both sources
                    existing = unique_docs[doc_id]
                    existing["source"] = f"{existing['source']}+{doc['source']}"
                    
                    # Keep track of both scores
                    if "semantic_score" in doc:
                        existing["semantic_score"] = doc["semantic_score"]
                    if "keyword_score" in doc:
                        existing["keyword_score"] = doc["keyword_score"]
                else:
                    # Add as new entry
                    unique_docs[doc_id] = doc
            
            # 4. Rerank combined results
            await self.status_emitter.emit("Reranking combined results...")
            reranked_docs = []
            
            for doc_id, doc in unique_docs.items():
                # Calculate hybrid score
                semantic_score = doc.get("semantic_score", 0)
                keyword_score = doc.get("keyword_score", 0)
                
                # Give precedence to docs found by both methods
                source_boost = 1.1 if "+" in doc["source"] else 1.0
                
                # Calculate hybrid score - weight between semantic and keyword matches
                hybrid_score = source_boost * (
                    (1 - self.valves.KEYWORD_SEARCH_WEIGHT) * semantic_score + 
                    self.valves.KEYWORD_SEARCH_WEIGHT * keyword_score
                )
                
                # Update doc with hybrid score
                doc["hybrid_score"] = hybrid_score
                reranked_docs.append(doc)
            
            # Sort by hybrid score
            reranked_docs.sort(key=lambda x: x.get("hybrid_score", 0), reverse=True)
            
            # Limit to top-k
            top_docs = reranked_docs[:self.valves.TOP_K]
            
            if self.valves.DEBUG_MODE:
                await self.message_emitter.emit(
                    f"#### Search Results\nSemantic: {len(semantic_docs)}, " +
                    f"Keyword: {len(keyword_docs)}, " +
                    f"Combined (deduplicated): {len(unique_docs)}, " +
                    f"Final: {len(top_docs)}\n"
                )
                
                for i, doc in enumerate(top_docs):
                    await self.message_emitter.emit(
                        f"Document {i+1}: ID={doc['id']}, " +
                        f"Hybrid={doc.get('hybrid_score', 0):.3f}, " +
                        f"Semantic={doc.get('semantic_score', 0):.3f}, " +
                        f"Keyword={doc.get('keyword_score', 0):.3f}, " +
                        f"Source={doc['source']}"
                    )
            
            return top_docs
            
        except Exception as e:
            error_traceback = traceback.format_exc()
            await self.status_emitter.emit(
                status="error",
                description=f"Error during advanced RAG query: {str(e)}",
                done=True
            )
            
            if self.valves.DEBUG_MODE:
                await self.message_emitter.emit(
                    f"#### Error Details\n```\n{error_traceback}\n```\n"
                )
            
            raise e
    
    async def rag_query(
        self,
        query: str,
        namespace: Optional[str] = None,
        include_prompt_template: bool = True,
        use_keywords: bool = True,
        __event_emitter__: Callable[[dict], Any] = None,
    ) -> str:
        """
        Process a RAG query by embedding the query, retrieving documents from Pinecone,
        and formatting the results for the LLM.
        
        Args:
            query: The user query to process
            namespace: Optional Pinecone namespace to query within
            include_prompt_template: Whether to include a prompt template in the response
            use_keywords: Whether to use keyword extraction for hybrid search
            __event_emitter__: Event emitter for status updates
            
        Returns:
            Formatted context from relevant documents or a complete prompt
        """
        self.status_emitter = StatusEventEmitter(__event_emitter__)
        self.message_emitter = MessageEventEmitter(__event_emitter__)
        
        try:
            # Setup and initialize
            await self.status_emitter.emit("Initializing vLLM RAG query...")
            
            # Debug mode information
            if self.valves.DEBUG_MODE:
                await self.message_emitter.emit(
                    "### Debug Mode Active\n\nDebugging information will be displayed.\n"
                )
                await self.message_emitter.emit(
                    f"#### Query\n```\n{query}\n```\n"
                )
            
            # Initialize Pinecone
            await self.status_emitter.emit("Connecting to Pinecone...")
            pinecone_setup = await self.setup_pinecone()
            if not isinstance(pinecone_setup, bool):
                await self.status_emitter.emit(
                    status="error",
                    description=f"Pinecone initialization failed: {pinecone_setup[1]}",
                    done=True
                )
                return json.dumps({"error": pinecone_setup[1]})
            
            # Generate embeddings
            await self.status_emitter.emit("Generating embeddings for query...")
            embedding = await self.get_embeddings(query)
            
            if self.valves.DEBUG_MODE:
                await self.message_emitter.emit(
                    f"#### Embedding Generated\nDimensions: {len(embedding)}\n"
                )
            
            # Use the advanced RAG pipeline for document retrieval
            await self.status_emitter.emit("Running advanced RAG pipeline...")
            documents = []
            try:
                documents = await self.advanced_rag_query(query, namespace, __event_emitter__)
                
                # Format results
                await self.status_emitter.emit("Formatting retrieved documents...")
                if not documents:
                    formatted_context = "No relevant documents found for the query."
                else:
                    formatted_context = await self.format_documents(documents)
                    
                    if self.valves.DEBUG_MODE and len(documents) > 0:
                        await self.message_emitter.emit(f"#### Retrieved {len(documents)} documents\n")
                        
            except Exception as advanced_error:
                # Fallback to simpler approach if advanced RAG fails
                await self.status_emitter.emit("Advanced RAG failed, falling back to simple search...")
                if self.valves.DEBUG_MODE:
                    await self.message_emitter.emit(f"Advanced RAG error: {str(advanced_error)}")
                    
                # Generate embeddings
                embedding = await self.get_embeddings(query)
                
                # Extract keywords (for simple hybrid search)
                keywords = None
                if use_keywords:
                    await self.status_emitter.emit("Extracting keywords from query...")
                    keywords = await self.extract_keywords_from_query(query)
                
                # Simple query
                await self.status_emitter.emit("Retrieving relevant documents from Pinecone...")
                documents = await self.query_pinecone(embedding, namespace, keywords)
                
                # Format results
                if not documents:
                    formatted_context = "No relevant documents found for the query."
                else:
                    formatted_context = await self.format_documents(documents)
            
            # Complete the query
            await self.status_emitter.emit(
                status="complete",
                description="RAG query completed successfully",
                done=True
            )
            
            if include_prompt_template:
                # Return a complete prompt template that includes both the context and instructions
                return await self.format_rag_prompt(query, formatted_context)
            else:
                # Return just the context
                return formatted_context
            
        except Exception as e:
            error_traceback = traceback.format_exc()
            await self.status_emitter.emit(
                status="error",
                description=f"Error during RAG query: {str(e)}",
                done=True
            )
            
            if self.valves.DEBUG_MODE:
                await self.message_emitter.emit(
                    f"#### Error Details\n```\n{error_traceback}\n```\n"
                )
                
            return json.dumps({"error": str(e)})
    
    async def search_by_keywords(
        self,
        keywords: List[str],
        namespace: Optional[str] = None,
        __event_emitter__: Callable[[dict], Any] = None,
    ) -> str:
        """
        Search for PowerPoint slides by keywords.
        
        Args:
            keywords: List of keywords to search for
            namespace: Optional namespace to query within
            __event_emitter__: Event emitter for status updates
            
        Returns:
            Formatted context from relevant slides
        """
        self.status_emitter = StatusEventEmitter(__event_emitter__)
        self.message_emitter = MessageEventEmitter(__event_emitter__)
        
        try:
            # Initialize Pinecone
            await self.status_emitter.emit("Connecting to Pinecone...")
            pinecone_setup = await self.setup_pinecone()
            if not isinstance(pinecone_setup, bool):
                await self.status_emitter.emit(
                    status="error",
                    description=f"Pinecone initialization failed: {pinecone_setup[1]}",
                    done=True
                )
                return json.dumps({"error": pinecone_setup[1]})
            
            # Prepare for search
            namespace = namespace or self.valves.PINECONE_NAMESPACE
            keyword_string = ", ".join(keywords)
            
            if self.valves.DEBUG_MODE:
                await self.message_emitter.emit(
                    f"#### Searching for keywords\n{keyword_string}\n"
                )
            
            # Fetch metadata for all vectors in the namespace
            # This is a simplified approach - in production you might want to use a more efficient method
            await self.status_emitter.emit("Searching for relevant slides...")
            
            # Generate embedding for the keyword string to support hybrid search
            embedding = await self.get_embeddings(keyword_string)
            
            # Query Pinecone with both the embedding and metadata filter
            query_params = {
                "vector": embedding,
                "top_k": self.valves.TOP_K * 2,  # Get more results for keyword filtering
                "include_metadata": True,
                "namespace": namespace,
            }
            
            query_response = self.pinecone_client.query(**query_params)
            
            # Post-process results to prioritize keyword matches
            results = []
            for match in query_response.matches:
                # Skip if below similarity threshold
                if match.score < self.valves.SIMILARITY_THRESHOLD * 0.8:  # Slightly lower threshold for keyword search
                    continue
                
                metadata = match.metadata
                # Check for keyword matches in either keywords or content
                keyword_match_score = 0
                if "keywords" in metadata:
                    for keyword in keywords:
                        if keyword.lower() in metadata["keywords"].lower():
                            keyword_match_score += 1
                
                if "content_preview" in metadata:
                    for keyword in keywords:
                        if keyword.lower() in metadata["content_preview"].lower():
                            keyword_match_score += 0.5
                
                # Combine vector similarity with keyword match score
                hybrid_score = (1 - self.valves.KEYWORD_SEARCH_WEIGHT) * match.score + \
                               self.valves.KEYWORD_SEARCH_WEIGHT * (keyword_match_score / len(keywords))
                
                doc = {
                    "id": match.id,
                    "score": match.score,
                    "keyword_score": keyword_match_score / len(keywords),
                    "hybrid_score": hybrid_score,
                    "metadata": metadata
                }
                results.append(doc)
            
            # Sort by hybrid score
            results.sort(key=lambda x: x["hybrid_score"], reverse=True)
            
            # Limit to the top_k most relevant results
            results = results[:self.valves.TOP_K]
            
            if self.valves.DEBUG_MODE:
                await self.message_emitter.emit(
                    f"#### Retrieved {len(results)} slides\n"
                )
                for i, doc in enumerate(results):
                    await self.message_emitter.emit(
                        f"Slide {i+1}: Vector Score {doc['score']:.3f}, " +
                        f"Keyword Score {doc['keyword_score']:.3f}, " +
                        f"Hybrid Score {doc['hybrid_score']:.3f}\n"
                    )
            
            # Format results
            if not results:
                formatted_context = f"No relevant slides found for keywords: {keyword_string}"
            else:
                formatted_context = await self.format_documents(results)
            
            await self.status_emitter.emit(
                status="complete",
                description="Keyword search completed successfully",
                done=True
            )
            
            return formatted_context
            
        except Exception as e:
            error_traceback = traceback.format_exc()
            await self.status_emitter.emit(
                status="error",
                description=f"Error during keyword search: {str(e)}",
                done=True
            )
            
            if self.valves.DEBUG_MODE:
                await self.message_emitter.emit(
                    f"#### Error Details\n```\n{error_traceback}\n```\n"
                )
                
            return json.dumps({"error": str(e)})
            
    async def list_available_documents(
        self,
        doc_type: Optional[str] = None,
        namespace: Optional[str] = None,
        __event_emitter__: Callable[[dict], Any] = None,
    ) -> str:
        """
        List all available documents in the Pinecone index, optionally filtered by type.
        
        Args:
            doc_type: Optional document type to filter by (e.g., "powerpoint", "pdf", "word")
            namespace: Optional namespace to query within
            __event_emitter__: Event emitter for status updates
            
        Returns:
            Formatted list of documents
        """
        self.status_emitter = StatusEventEmitter(__event_emitter__)
        self.message_emitter = MessageEventEmitter(__event_emitter__)
        
        try:
            # Setup and initialize
            await self.status_emitter.emit("Initializing Pinecone connection...")
            
            # Initialize Pinecone
            pinecone_setup = await self.setup_pinecone()
            if not isinstance(pinecone_setup, bool):
                await self.status_emitter.emit(
                    status="error",
                    description=f"Pinecone initialization failed: {pinecone_setup[1]}",
                    done=True
                )
                return json.dumps({"error": pinecone_setup[1]})
            
            # Fetch and process stats
            await self.status_emitter.emit("Fetching document data...")
            namespace = namespace or self.valves.PINECONE_NAMESPACE
            
            # This is a simplified approach - in practice you might need pagination for large indexes
            stats = self.pinecone_client.describe_index_stats()
            
            # Get unique documents by parsing vector IDs
            document_info = defaultdict(lambda: {"type": "unknown", "sections": set()})
            
            # Fetch some vectors to analyze
            fetch_response = self.pinecone_client.fetch(
                ids=[], 
                namespace=namespace, 
                limit=100
            )
            
            for vector_id, vector_data in fetch_response.get('vectors', {}).items():
                metadata = vector_data.get('metadata', {})
                
                # Parse the ID to get the document name
                id_parts = vector_id.split('_')
                if len(id_parts) > 1:
                    # The last part is usually the section/page/slide number
                    doc_name = '_'.join(id_parts[:-1])
                    section_id = id_parts[-1]
                    
                    # Determine document type from metadata or file extension
                    doc_type_from_metadata = metadata.get('doc_type', '').lower()
                    
                    if doc_type_from_metadata:
                        document_info[doc_name]['type'] = doc_type_from_metadata
                    elif doc_name.endswith('.pptx') or 'slide_number' in metadata:
                        document_info[doc_name]['type'] = 'powerpoint'
                    elif doc_name.endswith('.pdf') or 'page_number' in metadata:
                        document_info[doc_name]['type'] = 'pdf'
                    elif doc_name.endswith('.docx'):
                        document_info[doc_name]['type'] = 'word'
                    
                    document_info[doc_name]['sections'].add(section_id)
            
            # Filter by document type if specified
            if doc_type:
                filtered_docs = {
                    name: info for name, info in document_info.items()
                    if doc_type.lower() in info['type'].lower()
                }
            else:
                filtered_docs = document_info
            
            # Format the results
            if filtered_docs:
                # Group by document type
                docs_by_type = defaultdict(list)
                for doc_name, info in filtered_docs.items():
                    docs_by_type[info['type']].append({
                        'name': doc_name,
                        'sections': len(info['sections'])
                    })
                
                # Build the formatted output
                result = f"### Available Documents\n\n"
                
                for doc_type, docs in sorted(docs_by_type.items()):
                    result += f"#### {doc_type.title()} Documents\n\n"
                    for doc in sorted(docs, key=lambda d: d['name']):
                        result += f"- {doc['name']} ({doc['sections']} sections)\n"
                    result += "\n"
            else:
                if doc_type:
                    result = f"No {doc_type} documents found in the vector database."
                else:
                    result = "No documents found in the vector database."
            
            await self.status_emitter.emit(
                status="complete",
                description="Retrieved document list successfully",
                done=True
            )
            
            return result
            
        except Exception as e:
            await self.status_emitter.emit(
                status="error",
                description=f"Error listing documents: {str(e)}",
                done=True
            )
            return json.dumps({"error": str(e)})