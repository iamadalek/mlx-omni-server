from typing import Any, Dict, List, Union

import tiktoken

from ..utils.logger import logger
from .jina_mlx_reranker import MLXReranker
from .schema import RerankRequest, RerankResponse, RerankResult, RerankUsage


class RerankService:
    """Service for reranking documents using MLX models."""

    def __init__(self):
        # Map of loaded models for caching
        self._models: Dict[str, MLXReranker] = {}
        # Default encoder for token counting
        try:
            self._default_tokenizer = tiktoken.get_encoding("cl100k_base")
        except:
            try:
                self._default_tokenizer = tiktoken.get_encoding("p50k_base")
            except:
                logger.warning(
                    "Could not load any tiktoken encoding, token counts may be inaccurate"
                )
                self._default_tokenizer = None

    def _get_model(self, model_id: str) -> MLXReranker:
        """Get or load a reranker model based on its ID."""
        if model_id not in self._models:
            logger.info(f"Loading reranker model: {model_id}")
            try:
                # For Jina Reranker v3 MLX, model_id should be path to model directory
                # Default to current directory if just model name provided
                model_path = model_id if "/" in model_id else "."
                projector_path = f"{model_path}/projector.safetensors"
                reranker = MLXReranker(model_path=model_path, projector_path=projector_path)
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
