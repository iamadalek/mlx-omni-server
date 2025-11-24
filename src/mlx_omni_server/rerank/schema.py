from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field


class RerankRequest(BaseModel):
    model: str = Field(..., description="ID of the reranker model to use")
    query: str = Field(..., description="Search query to rank documents against")
    documents: Union[List[str], List[Dict[str, Any]]] = Field(
        ..., description="List of documents to rank (strings or dicts with 'text' field)"
    )
    top_n: Optional[int] = Field(
        None, description="Return only top N most relevant documents"
    )
    return_documents: bool = Field(
        True, description="Whether to return document text in response"
    )

    class Config:
        extra = "allow"

    def get_extra_params(self) -> Dict[str, Any]:
        """Get all extra parameters that aren't part of the standard API."""
        standard_fields = {"model", "query", "documents", "top_n", "return_documents"}
        return {k: v for k, v in self.model_dump().items() if k not in standard_fields}


class RerankResult(BaseModel):
    index: int = Field(..., description="Index in the original documents list")
    relevance_score: float = Field(..., description="Relevance score (higher = more relevant)")
    document: Optional[str] = Field(None, description="Document text if return_documents=True")


class RerankUsage(BaseModel):
    total_tokens: int = Field(..., description="Total tokens processed")


class RerankResponse(BaseModel):
    model: str = Field(..., description="Model used for reranking")
    results: List[RerankResult] = Field(..., description="Ranked documents")
    usage: RerankUsage = Field(..., description="Token usage information")
