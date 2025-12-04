"""Reranking service with auto-detection of reranker architecture.

Supports two reranker architectures:
1. Bi-encoder + Projector (Jina style): Fast, requires projector.safetensors
2. Cross-encoder (Qwen3 style): More accurate, uses generative scoring
"""

import os
from typing import Any, Dict, List, Protocol, Union

import tiktoken
from huggingface_hub import snapshot_download

from ..utils.logger import logger
from .cross_encoder_reranker import CrossEncoderReranker
from .jina_mlx_reranker import MLXReranker
from .schema import RerankRequest, RerankResponse, RerankResult, RerankUsage


class RerankerProtocol(Protocol):
    """Protocol for reranker implementations."""

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_n: int | None = None,
        return_embeddings: bool = False,
    ) -> List[dict]: ...


def resolve_model_path(model_id: str) -> str:
    """Resolve a model ID to a local filesystem path.

    Handles:
    - HuggingFace model IDs (e.g., "jinaai/jina-reranker-v3-mlx")
    - Local paths (e.g., "/path/to/model" or "./model")
    """
    # If it's already a valid local path, use it
    if os.path.isdir(model_id):
        return model_id

    # Check if it looks like a HuggingFace model ID (contains /)
    if "/" in model_id and not model_id.startswith(("/", "./")):
        try:
            # Download/cache the model and return the local path
            local_path = snapshot_download(
                repo_id=model_id,
                local_files_only=False,  # Allow downloading if not cached
            )
            logger.info(f"Resolved HuggingFace model {model_id} to {local_path}")
            return local_path
        except Exception as e:
            logger.warning(f"Failed to resolve {model_id} via HuggingFace Hub: {e}")
            # Fall back to using as-is
            return model_id

    return model_id


def has_projector(model_path: str) -> bool:
    """Check if a model has a projector.safetensors file."""
    projector_path = os.path.join(model_path, "projector.safetensors")
    return os.path.isfile(projector_path)


def create_reranker(model_id: str, model_path: str) -> RerankerProtocol:
    """Create the appropriate reranker based on model architecture.

    Auto-detects:
    - If projector.safetensors exists: Use Jina-style bi-encoder
    - Otherwise: Use cross-encoder (generative scoring)
    """
    if has_projector(model_path):
        logger.info(f"Using bi-encoder reranker (projector found) for {model_id}")
        projector_path = os.path.join(model_path, "projector.safetensors")
        return MLXReranker(model_path=model_path, projector_path=projector_path)
    else:
        logger.info(f"Using cross-encoder reranker (no projector) for {model_id}")
        return CrossEncoderReranker(model_path=model_path)


class RerankService:
    """Service for reranking documents using MLX models.

    Automatically detects and uses the appropriate reranker architecture:
    - Bi-encoder + Projector (Jina): Fast, single forward pass
    - Cross-encoder (Qwen3): More accurate, one pass per document
    """

    def __init__(self):
        # Map of loaded models for caching
        self._models: Dict[str, RerankerProtocol] = {}
        # Default encoder for token counting
        try:
            self._default_tokenizer = tiktoken.get_encoding("cl100k_base")
        except Exception:
            try:
                self._default_tokenizer = tiktoken.get_encoding("p50k_base")
            except Exception:
                logger.warning(
                    "Could not load any tiktoken encoding, token counts may be inaccurate"
                )
                self._default_tokenizer = None

    def _get_model(self, model_id: str) -> RerankerProtocol:
        """Get or load a reranker model based on its ID."""
        if model_id not in self._models:
            logger.info(f"Loading reranker model: {model_id}")
            try:
                # Resolve model_id to local filesystem path (handles HuggingFace IDs)
                model_path = resolve_model_path(model_id)

                # Create appropriate reranker based on architecture
                reranker = create_reranker(model_id, model_path)
                self._models[model_id] = reranker

            except Exception as e:
                logger.error(f"Error loading reranker model {model_id}: {str(e)}")
                raise RuntimeError(f"Failed to load reranker model: {str(e)}")

        return self._models[model_id]

    def _count_tokens(self, query: str, documents: List[str]) -> int:
        """Count tokens in query and documents."""
        if self._default_tokenizer is None:
            # Simple approximation
            query_tokens = len(query.split())
            doc_tokens = sum(len(doc.split()) for doc in documents)
            return query_tokens + doc_tokens

        try:
            query_tokens = len(self._default_tokenizer.encode(query))
            doc_tokens = sum(len(self._default_tokenizer.encode(doc)) for doc in documents)
            return query_tokens + doc_tokens
        except Exception as e:
            logger.warning(f"Error counting tokens: {str(e)}. Using fallback method.")
            query_tokens = len(query.split())
            doc_tokens = sum(len(doc.split()) for doc in documents)
            return query_tokens + doc_tokens

    def _extract_documents(self, documents: Union[List[str], List[Dict[str, Any]]]) -> List[str]:
        """Extract text from documents (handle both strings and dicts)."""
        extracted = []
        for doc in documents:
            if isinstance(doc, str):
                extracted.append(doc)
            elif isinstance(doc, dict):
                # Try common text fields
                text = doc.get("text") or doc.get("content") or doc.get("document") or str(doc)
                extracted.append(text)
            else:
                extracted.append(str(doc))
        return extracted

    def rerank_documents(self, request: RerankRequest) -> RerankResponse:
        """Rerank documents by relevance to query."""
        model_id = request.model
        reranker = self._get_model(model_id)

        # Extract document text
        documents = self._extract_documents(request.documents)

        # Count tokens for usage info
        token_count = self._count_tokens(request.query, documents)

        try:
            # Rerank documents
            results = reranker.rerank(
                query=request.query,
                documents=documents,
                top_n=request.top_n,
                return_embeddings=False,
            )

            # Build response
            rerank_results = []
            for result in results:
                rerank_result = RerankResult(
                    index=result["index"],
                    relevance_score=result["relevance_score"],
                    document=result["document"] if request.return_documents else None,
                )
                rerank_results.append(rerank_result)

            response = RerankResponse(
                model=model_id,
                results=rerank_results,
                usage=RerankUsage(total_tokens=token_count),
            )

            return response

        except Exception as e:
            logger.error(f"Error reranking documents: {str(e)}", exc_info=True)
            raise RuntimeError(f"Failed to rerank documents: {str(e)}")
