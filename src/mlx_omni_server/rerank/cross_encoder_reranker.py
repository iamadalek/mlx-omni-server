"""Cross-encoder reranker for causal LM models (e.g., Qwen3 Reranker).

Unlike bi-encoder approaches (Jina), cross-encoders process query-document
pairs together, allowing the model to see their interaction directly.
This is typically more accurate but slower (one forward pass per document).
"""

import re
from typing import List, Optional

import mlx.core as mx
from mlx_lm import generate, load

from ..utils.logger import logger


# Prompt template for cross-encoder scoring
# The model is asked to output a single number 0-10
# We suppress thinking mode by starting the assistant response
SCORE_PROMPT_TEMPLATE = """<|im_start|>system
You are a relevance scoring expert. Given a query and a document, output a single relevance score from 0 to 10.
- 0 means completely irrelevant
- 10 means perfectly relevant
Respond with ONLY the number. No explanation, no thinking, just the score.<|im_end|>
<|im_start|>user
Query: {query}

Document: {document}

Score:<|im_end|>
<|im_start|>assistant
<think>

</think>

"""


def parse_score(output: str) -> float:
    """Parse a relevance score from model output.

    Handles various formats:
    - "7" -> 7.0
    - "7.5" -> 7.5
    - "Score: 7" -> 7.0
    - "The relevance is 7" -> 7.0
    """
    # Try to find a number in the output
    # First try to match just a number at the start
    match = re.match(r"^\s*(\d+(?:\.\d+)?)", output.strip())
    if match:
        return float(match.group(1))

    # Try to find any number in the output
    numbers = re.findall(r"\d+(?:\.\d+)?", output)
    if numbers:
        # Take the first number found
        score = float(numbers[0])
        # Clamp to 0-10 range
        return max(0.0, min(10.0, score))

    # Default to middle score if parsing fails
    logger.warning(f"Could not parse score from: {output!r}, defaulting to 5.0")
    return 5.0


class CrossEncoderReranker:
    """Cross-encoder reranker using causal LM models.

    This works with models like Qwen3 Reranker that don't have a projector
    and instead use the LM's generative capability to score relevance.
    """

    def __init__(self, model_path: str):
        """Initialize cross-encoder reranker.

        Args:
            model_path: Path to MLX model (local path or HuggingFace ID)
        """
        logger.info(f"Loading cross-encoder reranker from: {model_path}")
        self.model, self.tokenizer = load(model_path)
        self.model.eval()
        self.model_path = model_path

    def _score_pair(self, query: str, document: str) -> float:
        """Score a single query-document pair.

        Args:
            query: Search query
            document: Document to score

        Returns:
            Relevance score (0-10, normalized to 0-1 for output)
        """
        prompt = SCORE_PROMPT_TEMPLATE.format(query=query, document=document)

        # Generate score - we only need a few tokens
        output = generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=10,  # Just need a number
        )

        score = parse_score(output)
        # Normalize to 0-1 range for consistency with Jina reranker
        return score / 10.0

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_n: Optional[int] = None,
        return_embeddings: bool = False,
    ) -> List[dict]:
        """Rerank documents by relevance to a query.

        Args:
            query: Search query string
            documents: List of document strings to rank
            top_n: Return only top N results (default: all)
            return_embeddings: Not supported for cross-encoder (always None)

        Returns:
            List of dicts with keys:
                - document: Original document text
                - relevance_score: Similarity score (0-1, higher = more relevant)
                - index: Position in input documents list
                - embedding: Always None (not supported)
        """
        if return_embeddings:
            logger.warning("Cross-encoder reranker does not support return_embeddings")

        # Score each document
        scores = []
        for i, doc in enumerate(documents):
            score = self._score_pair(query, doc)
            scores.append((i, score, doc))
            logger.debug(f"Document {i} score: {score:.4f}")

        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)

        # Apply top_n limit
        if top_n is not None:
            scores = scores[:top_n]

        # Build results
        return [
            {
                "document": doc,
                "relevance_score": score,
                "index": idx,
                "embedding": None,
            }
            for idx, score, doc in scores
        ]


if __name__ == "__main__":
    # Example usage
    reranker = CrossEncoderReranker("galaxycore/qwen3-reranker-8b-mlx")

    query = "What are the health benefits of green tea?"
    documents = [
        "Green tea contains antioxidants that may help reduce inflammation.",
        "The weather forecast predicts rain tomorrow.",
        "Studies show green tea can improve brain function and metabolism.",
    ]

    results = reranker.rerank(query, documents)

    print("Cross-Encoder Reranker Results:")
    for result in results:
        print(f"Score: {result['relevance_score']:.4f}, Index: {result['index']}, Document: {result['document'][:60]}...")
