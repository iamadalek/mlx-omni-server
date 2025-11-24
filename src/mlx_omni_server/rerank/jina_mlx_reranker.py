import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load
import numpy as np
from typing import Optional, List, Dict, Tuple
from safetensors import safe_open


class MLPProjector(nn.Module):
    """MLP projector to project hidden states to embedding space."""
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(1024, 512, bias=False)
        self.linear2 = nn.Linear(512, 512, bias=False)

    def __call__(self, x):
        x = self.linear1(x)
        x = nn.relu(x)
        x = self.linear2(x)
        return x


def load_projector(projector_path: str) -> MLPProjector:
    """Load projector weights from safetensors file."""
    projector = MLPProjector()

    with safe_open(projector_path, framework="numpy") as f:
        w0 = f.get_tensor("linear1.weight")
        w2 = f.get_tensor("linear2.weight")

        projector.linear1.weight = mx.array(w0)
        projector.linear2.weight = mx.array(w2)

    return projector


def sanitize_input(text: str, special_tokens: Dict[str, str]) -> str:
    """Remove special tokens from input text."""
    for token in special_tokens.values():
        text = text.replace(token, "")
    return text


def format_docs_prompts_func(
    query: str,
    docs: list[str],
    instruction: Optional[str] = None,
    special_tokens: Dict[str, str] = {},
    no_thinking: bool = True,
) -> str:
    """Format query and documents into a prompt for the model."""
    query = sanitize_input(query, special_tokens)
    docs = [sanitize_input(doc, special_tokens) for doc in docs]

    prefix = (
        "<|im_start|>system\n"
        "You are a search relevance expert who can determine a ranking of the passages based on how relevant they are to the query. "
        "If the query is a question, how relevant a passage is depends on how well it answers the question. "
        "If not, try to analyze the intent of the query and assess how well each passage satisfies the intent. "
        "If an instruction is provided, you should follow the instruction when determining the ranking."
        "<|im_end|>\n<|im_start|>user\n"
    )
    suffix = "<|im_end|>\n<|im_start|>assistant\n"
    if no_thinking:
        suffix += "<think>\n\n</think>\n\n"

    doc_emb_token = special_tokens["doc_embed_token"]
    query_emb_token = special_tokens["query_embed_token"]

    prompt = (
        f"I will provide you with {len(docs)} passages, each indicated by a numerical identifier. "
        f"Rank the passages based on their relevance to query: {query}\n"
    )

    if instruction:
        prompt += f'<instruct>\n{instruction}\n</instruct>\n'

    doc_prompts = [f'<passage id="{i}">\n{doc}{doc_emb_token}\n</passage>' for i, doc in enumerate(docs)]
    prompt += "\n".join(doc_prompts) + "\n"
    prompt += f"<query>\n{query}{query_emb_token}\n</query>"

    return prefix + prompt + suffix


class MLXReranker:
    """MLX-based implementation of jina-reranker-v3."""

    def __init__(self, model_path: str = ".", projector_path: str = "projector.safetensors"):
        """
        Initialize MLX-based reranker.

        Args:
            model_path: Path to MLX-converted Qwen3 model (default: current directory)
            projector_path: Path to projector weights in safetensors format
        """
        # Load MLX model and tokenizer
        self.model, self.tokenizer = load(model_path)
        self.model.eval()

        # Load projector
        self.projector = load_projector(projector_path)

        # Special tokens
        self.special_tokens = {"query_embed_token": "<|rerank_token|>", "doc_embed_token": "<|embed_token|>"}
        self.doc_embed_token_id = 151670
        self.query_embed_token_id = 151671

    def _compute_single_batch(
        self,
        query: str,
        docs: List[str],
        instruction: Optional[str] = None,
    ) -> Tuple[mx.array, mx.array, mx.array]:
        """
        Compute embeddings for a single batch of documents.

        Returns:
            query_embeds: Query embeddings after projection [1, 1, 512]
            doc_embeds: Document embeddings after projection [1, num_docs, 512]
            scores: Cosine similarity scores [num_docs]
        """
        prompt = format_docs_prompts_func(
            query,
            docs,
            instruction=instruction,
            special_tokens=self.special_tokens,
            no_thinking=True,
        )

        # Tokenize using MLX tokenizer
        input_ids = self.tokenizer.encode(prompt)

        # Get hidden states from model
        hidden_states = self.model.model([input_ids])  # Shape: [1, seq_len, hidden_size]

        # Remove batch dimension
        hidden_states = hidden_states[0]  # Shape: [seq_len, hidden_size]

        # Convert input_ids to numpy for indexing
        input_ids_np = np.array(input_ids)

        # Find positions of special tokens
        query_embed_positions = np.where(input_ids_np == self.query_embed_token_id)[0]
        doc_embed_positions = np.where(input_ids_np == self.doc_embed_token_id)[0]

        # Extract embeddings at special token positions
        if len(query_embed_positions) > 0:
            query_pos = int(query_embed_positions[0])
            query_hidden = mx.expand_dims(hidden_states[query_pos], axis=0)  # [1, hidden_size]
        else:
            raise ValueError("Query embed token not found in input")

        if len(doc_embed_positions) > 0:
            # Gather all document embeddings
            doc_hidden = mx.stack([hidden_states[int(pos)] for pos in doc_embed_positions])  # [num_docs, hidden_size]
        else:
            raise ValueError("Document embed tokens not found in input")

        # Project embeddings
        query_embeds = self.projector(query_hidden)  # [1, 512]
        doc_embeds = self.projector(doc_hidden)  # [num_docs, 512]

        # Reshape for consistency
        query_embeds = mx.expand_dims(query_embeds, axis=0)  # [1, 1, 512]
        doc_embeds = mx.expand_dims(doc_embeds, axis=0)  # [1, num_docs, 512]

        # Compute cosine similarity scores
        query_expanded = mx.broadcast_to(query_embeds, doc_embeds.shape)  # [1, num_docs, 512]

        # Cosine similarity
        scores = mx.sum(doc_embeds * query_expanded, axis=-1) / (
            mx.sqrt(mx.sum(doc_embeds * doc_embeds, axis=-1)) *
            mx.sqrt(mx.sum(query_expanded * query_expanded, axis=-1))
        )  # [1, num_docs]

        return query_embeds, doc_embeds, scores

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_n: Optional[int] = None,
        return_embeddings: bool = False,
    ) -> List[dict]:
        """
        Rerank documents by relevance to a query.

        Args:
            query: Search query string
            documents: List of document strings to rank
            top_n: Return only top N results (default: all)
            return_embeddings: Include embeddings in output (default: False)

        Returns:
            List of dicts with keys:
                - document: Original document text
                - relevance_score: Similarity score (higher = more relevant)
                - index: Position in input documents list
                - embedding: Doc embedding if return_embeddings=True, else None
        """
        # Process all documents at once
        query_embeds, doc_embeds, scores = self._compute_single_batch(
            query, documents, instruction=None
        )

        # Convert to numpy
        doc_embeds_np = np.array(doc_embeds[0])  # [num_docs, 512]
        scores_np = np.array(scores[0])  # [num_docs]

        # Sort by relevance score (descending)
        scores_argsort = np.argsort(scores_np)[::-1]

        # Determine top_n
        if top_n is None:
            top_n = len(documents)
        else:
            top_n = min(top_n, len(documents))

        # Build results
        return [
            {
                'document': documents[scores_argsort[i]],
                'relevance_score': float(scores_np[scores_argsort[i]]),
                'index': int(scores_argsort[i]),
                'embedding': doc_embeds_np[scores_argsort[i]] if return_embeddings else None,
            }
            for i in range(top_n)
        ]


if __name__ == "__main__":
    # Example usage
    reranker = MLXReranker()

    query = "What are the health benefits of green tea?"
    documents = [
        "Green tea contains antioxidants called catechins that may help reduce inflammation and protect cells from damage.",
        "El precio del café ha aumentado un 20% este año debido a problemas en la cadena de suministro.",
        "Studies show that drinking green tea regularly can improve brain function and boost metabolism.",
        "Basketball is one of the most popular sports in the United States.",
        "绿茶富含儿茶素等抗氧化剂，可以降低心脏病风险，还有助于控制体重。",
        "Le thé vert est riche en antioxydants et peut améliorer la fonction cérébrale.",
    ]

    results = reranker.rerank(query, documents)

    print("MLX Reranker Results:")
    for result in results:
        print(f"Score: {result['relevance_score']:.4f}, Index: {result['index']}, Document: {result['document'][:80]}...")
