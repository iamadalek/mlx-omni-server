from fastapi import APIRouter, HTTPException

from .rerank_service import RerankService
from .schema import RerankRequest, RerankResponse

router = APIRouter(tags=["rerank"])
rerank_service = RerankService()


@router.post("/rerank", response_model=RerankResponse)
@router.post("/v1/rerank", response_model=RerankResponse)
async def rerank_documents(request: RerankRequest) -> RerankResponse:
    """Rerank documents by relevance to a query.

    This endpoint takes a query and a list of documents, and returns the documents
    ranked by their relevance to the query. Uses MLX-based Jina Reranker v3 for
    fast, on-device reranking.
    """
    try:
        return rerank_service.rerank_documents(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
