# RAG and Retrieval Augmented Generation Concepts

## What is Retrieval Augmented Generation (RAG)?

Retrieval Augmented Generation (RAG) is a framework that combines a retrieval system with a generative language model. Instead of relying solely on the model's parametric memory (knowledge baked into weights during training), RAG retrieves relevant documents from an external corpus at inference time and uses them as context for generation. This grounds the model's output in real source documents, significantly reducing hallucination and enabling the model to answer questions about content it was never trained on.

The RAG pipeline consists of two main components: a retriever that fetches relevant documents given a query, and a generator (typically a large language model) that produces an answer conditioned on those documents. The original RAG paper by Lewis et al. (2020) demonstrated that this approach outperforms pure parametric models on knowledge-intensive NLP tasks.

## Vector Retrieval in RAG

Vector retrieval uses dense embeddings to find semantically similar documents from a corpus. Each document chunk is encoded into a high-dimensional vector using an embedding model. At query time, the query is also encoded into a vector and the system finds the nearest neighbors in the vector space using approximate nearest neighbor search. This approach captures semantic meaning — it can match documents that are conceptually related even if they use different words than the query.

Dense retrieval models such as DPR (Dense Passage Retrieval) learn to encode queries and passages into a shared embedding space where relevant pairs are close together. This contrasts with sparse retrieval methods like BM25 which rely on exact term matching and term frequency statistics.

## What are Vector Embeddings?

A vector embedding is a numerical representation of text as a fixed-size array of floating point numbers. Embedding models are neural networks trained to map text into a vector space where semantically similar texts are geometrically close. The quality of embeddings directly affects retrieval quality — better embeddings produce more semantically meaningful representations, which means the retriever finds more relevant chunks for a given query.

Popular embedding models include OpenAI's text-embedding-ada-002, sentence-transformers models such as all-MiniLM-L6-v2, and nomic-embed-text. These models are trained using contrastive learning objectives on large datasets of similar and dissimilar text pairs.

## Token Embeddings

Token embeddings convert individual words or subword tokens into fixed-size vectors. These are learned during model training and capture syntactic and semantic relationships between words. Token embeddings are the foundation of transformer models — the input text is tokenized and each token is mapped to an embedding vector before being processed by the attention layers. Unlike sentence embeddings which represent entire passages, token embeddings represent individual units and are combined by the model's layers.

## What is a Vector Database?

A vector database is a specialized data store designed to efficiently store and query high-dimensional embedding vectors. Unlike traditional databases that query by exact values, vector databases support approximate nearest neighbor (ANN) search — finding the k vectors most similar to a query vector according to a distance metric such as cosine similarity or Euclidean distance.

Popular vector databases include Chroma, Pinecone, Weaviate, Qdrant, and FAISS (Facebook AI Similarity Search). Chroma is an open-source embedding database that runs locally, making it well-suited for development and on-premise deployments. FAISS is a library for efficient similarity search developed by Facebook Research that supports billion-scale vector search.

## BM25 and Sparse Retrieval

BM25 (Best Match 25) is a probabilistic ranking function used for keyword-based document retrieval. It scores documents based on term frequency (how often a query term appears in the document) and inverse document frequency (how rare the term is across the corpus), with saturation functions that prevent any single term from dominating. BM25 is fast, interpretable, and works well for exact keyword matches but cannot capture semantic similarity — it will miss documents that use different words to express the same concept.

BM25 is part of the TF-IDF family of retrieval methods, collectively called sparse retrieval because most dimensions in the document representation are zero (only terms present in the document have non-zero weights).

## Hybrid Retrieval

Hybrid retrieval combines dense vector search with sparse keyword search (BM25) to capture both semantic relevance and exact-match relevance. A hybrid retriever runs both searches and merges the results using a weighted combination of scores. This approach is more robust than either method alone — vector search handles paraphrases and semantic queries while BM25 handles rare terms, acronyms, and exact entity names. Hybrid retrieval consistently outperforms either method alone on standard retrieval benchmarks.

## Semantic Search

Semantic search retrieves documents based on meaning and intent rather than exact keyword matches. By representing both documents and queries as dense vectors in the same embedding space, semantic search can return relevant results even when the query and document share no words in common. For example, a query for "car engine problems" might retrieve documents about "vehicle motor failures" through semantic search even though there is no keyword overlap.

## What is Chunking?

Chunking is the process of splitting documents into smaller pieces before indexing them in a vector store. Because embedding models have a maximum input length (context window) and because retrieval works better with focused, topically coherent pieces, documents are split into chunks of a few hundred to a few thousand tokens. Common chunking strategies include fixed-size chunking with overlap (splitting every N characters with M characters of overlap to preserve continuity), recursive character splitting (splitting on paragraph boundaries, then sentence boundaries, then word boundaries), and semantic chunking (splitting at topic boundaries detected by embedding similarity).

Chunk size affects the retrieval-context tradeoff: smaller chunks improve retrieval precision because each chunk is more topically focused, but the retrieved context may lack surrounding information needed for generation. Larger chunks provide more context but may include irrelevant content that distracts the model. A chunk overlap of 10-20% helps preserve continuity across chunk boundaries.

## How RAG Reduces Hallucination

Standard LLMs generate text based entirely on patterns learned during training (parametric memory). When asked about facts not well-represented in training data, or about recent events, they may confidently generate plausible-sounding but incorrect information — this is hallucination. RAG reduces hallucination by providing the model with explicit source documents at inference time. The model is instructed to answer only from the provided context, and citations allow users to verify claims against the original sources. This grounds generation in retrieved evidence rather than parametric memory.

## LangChain

LangChain is an open-source framework for building applications powered by large language models. It provides abstractions for chaining LLM calls together with tools, memory, and data sources. Key components include chains (sequences of calls), agents (LLMs that decide which actions to take), retrievers (interfaces for fetching relevant documents), vector stores (integrations with Chroma, Pinecone, FAISS, and others), and document loaders (for ingesting PDFs, web pages, databases, etc.).

LangChain's retrieval chain combines a retriever with an LLM and a prompt template to answer questions using retrieved context. The basic setup involves loading documents, splitting into chunks, embedding and storing in a vector store, creating a retriever from the vector store, and chaining the retriever with an LLM using a prompt that instructs the model to answer from context.

## LangGraph

LangGraph is a library built on top of LangChain for building stateful, multi-actor applications with language models. It models agent workflows as directed graphs where nodes represent processing steps (agents or functions) and edges represent transitions between steps. Unlike linear chains, LangGraph supports cycles, conditional branching, and parallel execution — enabling complex agentic behaviors like retrieval-then-validate-then-retry loops.

LangGraph uses a shared state object that flows through the graph. Each node receives the current state, performs computation, and returns an updated state. Conditional edges allow routing to different nodes based on the current state — for example routing to a retry node if retrieval quality is insufficient, or to an abstain node if validation fails. This makes LangGraph well suited for building robust agentic RAG pipelines with error handling and quality gates.

## Agentic RAG

Agentic RAG extends basic RAG pipelines with autonomous decision-making. Instead of a fixed retrieve-then-generate sequence, an agentic system can decide to reformulate the query if retrieval quality is poor, retrieve from multiple sources, validate the generated answer against citations, and abstain from answering if evidence is insufficient. These decisions are made by LLM-based agents or rule-based routers operating within a structured workflow graph.

Key stages in an agentic RAG pipeline typically include: query normalization and planning (decomposing complex queries into sub-queries), retrieval (fetching relevant chunks), adequacy checking (assessing whether retrieved evidence is sufficient), synthesis (generating a grounded answer), citation validation (verifying every claim is traceable to a source), and finalization or abstention.

## Retrieval Evaluation Metrics

Key metrics for evaluating RAG systems include:

Hit@k (Retrieval Hit Rate): The fraction of queries where the correct source document appears in the top-k retrieved chunks. Measures whether the retriever surfaces relevant content at all.

MRR (Mean Reciprocal Rank): The average of 1/rank where rank is the position of the first relevant result. Rewards retrieval systems that rank relevant documents higher.

Citation Precision: The fraction of citations in generated answers that match expected source documents. Measures whether the system cites appropriate sources.

Abstain Precision and Recall: Precision measures what fraction of abstentions were correct (the system correctly refused to answer unanswerable questions). Recall measures what fraction of unanswerable questions were correctly refused.

Adversarial Abstain Rate: The fraction of prompt injection and policy-violating queries that were correctly blocked.

## Confidence Scoring and Abstention

A RAG system should not always generate an answer — when retrieved evidence is insufficient or ambiguous, it should abstain rather than hallucinate. Confidence scoring assigns a numerical score to each answer based on retrieval quality, citation coverage, and model confidence. Low-confidence answers can be flagged for human review or replaced with an explicit abstention message. Abstention is particularly important for safety-critical applications where a wrong answer is worse than no answer.

## Reranking

Reranking is a two-stage retrieval approach where an initial set of candidate documents is retrieved using a fast method (BM25 or vector search) and then re-scored using a more expensive cross-encoder model. Cross-encoders jointly encode the query and document together, capturing fine-grained relevance signals that bi-encoder models miss. Reranking improves precision at the cost of additional latency, and is typically applied to the top-20 to top-100 initial candidates to produce the final top-k results passed to the generator.

## Dense Passage Retrieval vs Sparse Retrieval Tradeoffs

Dense retrieval (vector search) captures semantic meaning and handles paraphrases well but requires training an embedding model, is computationally expensive for large corpora, and can miss exact keyword matches for rare terms. Sparse retrieval (BM25) is fast, requires no training, handles rare terms and exact matches well, but cannot capture semantic similarity. In practice hybrid approaches that combine both consistently outperform either alone. The choice of chunk size, embedding model, and retrieval strategy are all hyperparameters that should be tuned for the specific corpus and query distribution.

## Fine-tuning vs RAG

Fine-tuning trains the model's weights on domain-specific data, baking knowledge into the model's parameters. RAG retrieves knowledge at inference time from an external corpus. Fine-tuning produces faster inference (no retrieval step) and can improve the model's style and instruction-following for a domain. However, fine-tuning is expensive, requires retraining when knowledge changes, and provides no citation mechanism. RAG enables dynamic knowledge updates without retraining, provides traceable citations, and works with any LLM — but adds retrieval latency and depends on retrieval quality. For most enterprise knowledge base applications, RAG is preferred because knowledge changes frequently and auditability is important.
