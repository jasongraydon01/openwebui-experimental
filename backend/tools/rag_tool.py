"""
Simple Document RAG Integration Tool for OpenWebUI
=================================================================

This tool enables simple document retrieval using embeddings and Pinecone
for semantic search with hybrid keyword-based ranking.

License: Apache 2.0

Requirements:
    - Python 3.8+
    - OpenWebui
    - Pinecone
    - Requests
"""

import os
import json
import re
from typing import List, Dict, Any, Optional
import requests
import pinecone

class Tools:
    """
    Simple RAG Tool for OpenWebUI using Pinecone vector database
    
    This tool provides basic retrieval-augmented generation capabilities using
    embeddings and Pinecone vector database with hybrid search functionality.
    """
    
    def __init__(self):
        """Initialize the RAG tool with default configuration."""
        # Configuration
        self.embedding_url = os.getenv("VLLM_EMBEDDINGS_URL", "http://localhost:8001/v1/embeddings")
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "Alibaba-NLP/gte-Qwen2-7B-instruct")
        self.pinecone_api_key = os.getenv("PINECONE_API_KEY", "")
        self.pinecone_environment = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")
        self.pinecone_index = os.getenv("PINECONE_INDEX", "openwebui-rag")
        self.pinecone_namespace = os.getenv("PINECONE_NAMESPACE", "default")
        self.top_k = int(os.getenv("TOP_K", "5"))
        self.similarity_threshold = float(os.getenv("SIMILARITY_THRESHOLD", "0.7"))
        self.keyword_weight = float(os.getenv("KEYWORD_WEIGHT", "0.3"))
        
        # HTTP headers
        self.headers = {
            "Content-Type": "application/json",
            "User-Agent": "OpenWebUI-RAG-Tool/1.0",
        }
        
        # Initialize Pinecone client
        self.pinecone_client = None
        
    def _setup_pinecone(self) -> bool:
        """Initialize the Pinecone client if not already initialized."""
        if not self.pinecone_api_key:
            print("WARNING: PINECONE_API_KEY environment variable not set")
            return False
            
        if self.pinecone_client is None:
            try:
                pinecone.init(
                    api_key=self.pinecone_api_key,
                    environment=self.pinecone_environment
                )
                self.pinecone_client = pinecone.Index(self.pinecone_index)
                return True
            except Exception as e:
                print(f"ERROR: Failed to initialize Pinecone: {str(e)}")
                return False
        return True
    
    def _get_embeddings(self, text: str) -> List[float]:
        """Generate embeddings for the input text using the embedding API."""
        if not text or not isinstance(text, str):
            raise ValueError("Empty or invalid text provided for embedding")
            
        payload = {
            "input": text,
            "model": self.embedding_model,
        }
        
        response = requests.post(
            self.embedding_url,
            json=payload,
            headers=self.headers,
            timeout=30
        )
        response.raise_for_status()
        data = response.json()
        
        if "data" in data and len(data["data"]) > 0:
            return data["data"][0]["embedding"]
        else:
            raise ValueError("No embedding found in response")
            
    def _extract_keywords(self, query: str) -> List[str]:
        """Extract potential keywords from a user query."""
        stopwords = {'a', 'an', 'the', 'and', 'or', 'but', 'if', 'because', 'as', 'what', 
                    'which', 'this', 'that', 'these', 'those', 'then', 'just', 'so'}
        
        # Normalize and tokenize
        query = query.lower()
        query = re.sub(r'[^\w\s]', ' ', query)
        words = query.split()
        
        # Filter and keep meaningful words
        keywords = [word for word in words if word not in stopwords and len(word) > 3]
        return keywords[:10]  # Limit to 10 keywords
    
    def _format_results(self, results: List[Dict]) -> str:
        """Format the retrieved documents into a user-friendly output with citation-friendly format."""
        if not results:
            return "No relevant documents found."
            
        formatted_output = "### Retrieved Documents\n\n"
        citation_info = []
        
        for i, doc in enumerate(results):
            source_num = i + 1
            metadata = doc.get("metadata", {})
            
            # Extract document information
            doc_type = metadata.get("doc_type", "Document")
            file_name = metadata.get("file_name", "Unknown")
            section_id = metadata.get("section_id", metadata.get("page_number", metadata.get("slide_number", "?")))
            content = metadata.get("content_preview", "No content available")
            keywords = metadata.get("keywords", "")
            source_link = metadata.get("source_link", metadata.get("one_drive_link", "#"))
            
            # Additional metadata fields
            last_modified = metadata.get("last_modified", "")
            research_type = metadata.get("research_type", "")
            project_type = metadata.get("project_type", "")
            client = metadata.get("client", "")
            product = metadata.get("product", "")
            
            # Format document entry with citation marker
            formatted_output += f"### [Source {source_num}] {doc_type} - {file_name} (Section {section_id})\n\n"
            formatted_output += f"{content}\n\n"
            
            # Add metadata information
            metadata_fields = []
            
            if keywords:
                metadata_fields.append(f"**Keywords**: {keywords}")
                
            if last_modified:
                # Convert timestamp to readable date if it's a number
                if isinstance(last_modified, (int, float)):
                    from datetime import datetime
                    date_str = datetime.fromtimestamp(last_modified).strftime("%Y-%m-%d %H:%M")
                    metadata_fields.append(f"**Last Modified**: {date_str}")
                else:
                    metadata_fields.append(f"**Last Modified**: {last_modified}")
                    
            if research_type:
                metadata_fields.append(f"**Research Type**: {research_type}")
                
            if project_type:
                metadata_fields.append(f"**Project Type**: {project_type}")
                
            if client:
                metadata_fields.append(f"**Client**: {client}")
                
            if product:
                metadata_fields.append(f"**Product**: {product}")
                
            formatted_output += "\n".join(metadata_fields) + "\n\n"
            
            # Ensure the source link is formatted as a clickable URL
            formatted_output += f"**Source Link**: [Link]({source_link})\n\n"
            
            # Add to citation info for easy reference
            citation = {
                "number": source_num,
                "title": file_name,
                "type": doc_type,
                "section": section_id,
                "link": source_link
            }
            citation_info.append(citation)
            
        # Add citation guide for LLMs
        formatted_output += "\n### Citation Guide for Models\n"
        formatted_output += "When using information from these sources, please cite them as follows:\n"
        formatted_output += "- For inline citations: [Source X]\n"
        formatted_output += "- For linked citations: [Source X](link)\n"
        formatted_output += "- Example: According to [Source 1](link), the project timeline is scheduled for Q3.\n\n"
        
        formatted_output += "### Citation References\n"
        for citation in citation_info:
            formatted_output += f"[Source {citation['number']}]: {citation['title']} ({citation['type']}, Section {citation['section']}) - [Link]({citation['link']})\n"
        
        return formatted_output
    
    def format_for_llm(self, query: str, search_results: str) -> str:
        """
        Format search results into a complete response template for language models.
        
        :param query: The original user query
        :param search_results: The formatted search results
        :return: A complete template for the LLM to fill in with its response
        """
        prompt = f"""
### Query
{query}

### Available Information
{search_results}

### Instructions for Answering
1. Answer the query based ONLY on the information provided above.
2. If the information doesn't contain the answer, acknowledge this limitation.
3. Cite your sources using [Source X] notation, where X is the source number.
4. For key information, include the source link on first mention: [Source X](link).
5. Be concise, accurate, and helpful.

### Answer
"""
        return prompt
    
    def rag_search(self, query: str, namespace: Optional[str] = None, format_for_llm: bool = False) -> str:
        """
        Perform a hybrid search using both semantic similarity and keyword matching.
        
        :param query: The user query to search for
        :param namespace: Optional Pinecone namespace to use (defaults to configured namespace)
        :param format_for_llm: Whether to format the results as a prompt for language models
        :return: Formatted results containing the retrieved documents
        """
        try:
            # Initialize Pinecone if needed
            if not self._setup_pinecone():
                return "RAG search failed: Could not connect to Pinecone database."
                
            # Use the specified namespace or default
            namespace = namespace or self.pinecone_namespace
            
            # 1. Get embedding for the query
            embedding = self._get_embeddings(query)
            
            # 2. Extract keywords for hybrid search
            keywords = self._extract_keywords(query)
            
            # 3. Query Pinecone for similar documents
            query_response = self.pinecone_client.query(
                vector=embedding,
                top_k=self.top_k * 2,  # Get more results for reranking
                namespace=namespace,
                include_metadata=True
            )
            
            # 4. Hybrid reranking with keywords
            results = []
            for match in query_response.matches:
                # Skip if below similarity threshold
                if match.score < self.similarity_threshold:
                    continue
                    
                metadata = match.metadata
                
                # Calculate keyword match score
                keyword_score = 0
                if keywords and "keywords" in metadata:
                    for keyword in keywords:
                        if keyword.lower() in metadata["keywords"].lower():
                            keyword_score += 1
                    if "content_preview" in metadata:
                        for keyword in keywords:
                            if keyword.lower() in metadata["content_preview"].lower():
                                keyword_score += 0.5
                
                # Normalize keyword score
                keyword_score = keyword_score / max(len(keywords), 1)
                
                # Calculate hybrid score
                hybrid_score = (1 - self.keyword_weight) * match.score + self.keyword_weight * keyword_score
                
                # Add to results
                results.append({
                    "id": match.id,
                    "score": match.score,
                    "keyword_score": keyword_score,
                    "hybrid_score": hybrid_score,
                    "metadata": metadata
                })
            
            # 5. Sort by hybrid score and limit to top_k
            results.sort(key=lambda x: x["hybrid_score"], reverse=True)
            results = results[:self.top_k]
            
            # 6. Format results
            formatted_results = self._format_results(results)
            
            # 7. Format for LLM if requested
            if format_for_llm:
                return self.format_for_llm(query, formatted_results)
            
            return formatted_results
            
        except Exception as e:
            return f"RAG search failed: {str(e)}"
    
    def search_by_keywords(self, keywords: List[str], namespace: Optional[str] = None) -> str:
        """
        Search for documents by specific keywords.
        
        :param keywords: List of keywords to search for
        :param namespace: Optional Pinecone namespace to use
        :return: Formatted results containing the retrieved documents
        """
        try:
            # Initialize Pinecone if needed
            if not self._setup_pinecone():
                return "Keyword search failed: Could not connect to Pinecone database."
                
            # Use the specified namespace or default
            namespace = namespace or self.pinecone_namespace
            
            # Convert keywords to embedding for semantic search
            keyword_string = " ".join(keywords)
            embedding = self._get_embeddings(keyword_string)
            
            # Query Pinecone
            query_response = self.pinecone_client.query(
                vector=embedding,
                top_k=self.top_k * 2,
                namespace=namespace,
                include_metadata=True
            )
            
            # Process and rerank results by keyword matches
            results = []
            for match in query_response.matches:
                metadata = match.metadata
                
                # Calculate keyword match score
                keyword_score = 0
                if "keywords" in metadata:
                    for keyword in keywords:
                        if keyword.lower() in metadata["keywords"].lower():
                            keyword_score += 1
                if "content_preview" in metadata:
                    for keyword in keywords:
                        if keyword.lower() in metadata["content_preview"].lower():
                            keyword_score += 0.5
                
                # Normalize score
                keyword_score = keyword_score / max(len(keywords), 1)
                
                # Calculate hybrid score with higher weight on keyword matches
                hybrid_score = (1 - self.keyword_weight * 2) * match.score + (self.keyword_weight * 2) * keyword_score
                
                results.append({
                    "id": match.id,
                    "score": match.score,
                    "keyword_score": keyword_score,
                    "hybrid_score": hybrid_score,
                    "metadata": metadata
                })
            
            # Sort by hybrid score and limit to top_k
            results.sort(key=lambda x: x["hybrid_score"], reverse=True)
            results = results[:self.top_k]
            
            # Format and return results
            return self._format_results(results)
            
        except Exception as e:
            return f"Keyword search failed: {str(e)}"
    
    def list_documents(self, doc_type: Optional[str] = None) -> str:
        """
        List available documents in the vector database, optionally filtered by type.
        
        :param doc_type: Optional document type to filter by (e.g., "pdf", "powerpoint")
        :return: A formatted list of documents
        """
        try:
            # Initialize Pinecone if needed
            if not self._setup_pinecone():
                return "Could not connect to Pinecone database."
                
            # Get index stats
            stats = self.pinecone_client.describe_index_stats()
            
            # Get total vector count
            namespace_stats = stats.get('namespaces', {}).get(self.pinecone_namespace, {})
            vector_count = namespace_stats.get('vector_count', 0)
            
            if vector_count == 0:
                return "No documents found in the vector database."
                
            # Fetch a sample of vectors to analyze
            fetch_limit = min(100, vector_count)
            fetch_response = self.pinecone_client.fetch(
                ids=[], 
                namespace=self.pinecone_namespace, 
                limit=fetch_limit
            )
            
            # Collect unique document information
            documents = {}
            for vector_id, vector_data in fetch_response.get('vectors', {}).items():
                metadata = vector_data.get('metadata', {})
                
                # Parse document name from ID
                id_parts = vector_id.split('_')
                if len(id_parts) > 1:
                    doc_name = '_'.join(id_parts[:-1])
                    doc_type_value = metadata.get('doc_type', 'unknown')
                    
                    if doc_type and doc_type.lower() not in doc_type_value.lower():
                        continue
                        
                    if doc_name not in documents:
                        documents[doc_name] = {
                            'type': doc_type_value,
                            'sections': 1
                        }
                    else:
                        documents[doc_name]['sections'] += 1
            
            # Format results
            if not documents:
                return f"No{' ' + doc_type if doc_type else ''} documents found in the vector database."
                
            result = "### Available Documents\n\n"
            
            # Group by document type
            by_type = {}
            for name, info in documents.items():
                doc_type = info['type']
                if doc_type not in by_type:
                    by_type[doc_type] = []
                by_type[doc_type].append({
                    'name': name,
                    'sections': info['sections']
                })
            
            # Format by type
            for doc_type, docs in sorted(by_type.items()):
                result += f"#### {doc_type.title()} Documents\n\n"
                for doc in sorted(docs, key=lambda x: x['name']):
                    result += f"- {doc['name']} ({doc['sections']} sections)\n"
                result += "\n"
                
            return result
            
        except Exception as e:
            return f"Error listing documents: {str(e)}"