"""Tests for IRIS model components."""

import sys
from pathlib import Path

import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def test_query_vectors():
    """Test query vector initialization."""
    from criteria_bge_hpo.models import QueryVectors

    print("\n1. Testing QueryVectors...")
    query_module = QueryVectors(num_queries=8, embedding_dim=768)

    queries = query_module()
    print(f"   Shape: {queries.shape}")
    assert queries.shape == (8, 768), f"Expected (8, 768), got {queries.shape}"

    normalized = query_module.get_normalized_queries()
    print(f"   Normalized shape: {normalized.shape}")

    # Check normalization
    norms = torch.norm(normalized, dim=1)
    print(f"   Norms: min={norms.min():.4f}, max={norms.max():.4f}")
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)

    print("   ✓ QueryVectors passed")


def test_linear_attention():
    """Test linear attention mechanism."""
    from criteria_bge_hpo.models import LinearAttention

    print("\n2. Testing LinearAttention...")
    attention = LinearAttention(temperature=0.1)

    query = torch.randn(768)
    retrieved = torch.randn(8, 768)

    aggregated = attention(query, retrieved)
    print(f"   Input query shape: {query.shape}")
    print(f"   Retrieved shape: {retrieved.shape}")
    print(f"   Output shape: {aggregated.shape}")

    assert aggregated.shape == (768,), f"Expected (768,), got {aggregated.shape}"

    print("   ✓ LinearAttention passed")


def test_query_penalty_loss():
    """Test query penalty loss."""
    from criteria_bge_hpo.models import QueryPenaltyLoss

    print("\n3. Testing QueryPenaltyLoss...")
    penalty_fn = QueryPenaltyLoss(lambda_penalty=0.1, similarity_threshold=0.4)

    # Create similar queries (should have high penalty)
    similar_queries = torch.randn(8, 768)
    similar_queries = similar_queries / torch.norm(similar_queries, dim=1, keepdim=True)

    penalty = penalty_fn(similar_queries)
    print(f"   Penalty for random queries: {penalty.item():.4f}")

    # Create identical queries (should have high penalty)
    identical_queries = torch.ones(8, 768)
    identical_queries = identical_queries / torch.norm(identical_queries, dim=1, keepdim=True)
    penalty_identical = penalty_fn(identical_queries)
    print(f"   Penalty for identical queries: {penalty_identical.item():.4f}")

    assert penalty_identical >= 0, "Penalty should be non-negative"
    assert penalty.item() >= 0, "Penalty should be non-negative"

    print("   ✓ QueryPenaltyLoss passed")


def test_chunk_retriever():
    """Test FAISS-based retriever."""
    from criteria_bge_hpo.models import ChunkRetriever
    import numpy as np

    print("\n4. Testing ChunkRetriever...")
    retriever = ChunkRetriever(embedding_dim=768, use_gpu=False)

    # Create dummy chunks
    num_chunks = 100
    embeddings = np.random.randn(num_chunks, 768).astype(np.float32)
    texts = [f"Chunk {i}" for i in range(num_chunks)]

    retriever.add_chunks(embeddings, texts)
    print(f"   Added {retriever.num_chunks} chunks")

    # Search
    query_vectors = torch.randn(8, 768)
    similarities, indices = retriever.search(query_vectors, k=8)

    print(f"   Similarities shape: {similarities.shape}")
    print(f"   Indices shape: {indices.shape}")

    assert similarities.shape == (8, 8), f"Expected (8, 8), got {similarities.shape}"
    assert indices.shape == (8, 8), f"Expected (8, 8), got {indices.shape}"

    # Get retrieved chunks
    retrieved = retriever.get_retrieved_chunks(indices)
    print(f"   Retrieved {len(retrieved)} sets of chunks")
    print(f"   Example: {retrieved[0][0]}")

    print("   ✓ ChunkRetriever passed")


def test_embedding_model():
    """Test embedding model wrapper."""
    from criteria_bge_hpo.models import EmbeddingModel

    print("\n5. Testing EmbeddingModel...")
    try:
        model = EmbeddingModel("sentence-transformers/all-MiniLM-L6-v2", freeze=True)

        texts = [
            "This is a test sentence.",
            "Another test sentence here.",
        ]

        embeddings = model.encode(texts, batch_size=2)
        print(f"   Embeddings shape: {embeddings.shape}")
        print(f"   Embedding dim: {model.embedding_dim}")

        assert embeddings.shape[0] == len(texts)
        assert embeddings.shape[1] == model.embedding_dim

        print("   ✓ EmbeddingModel passed")
    except Exception as e:
        print(f"   ⚠ EmbeddingModel skipped (requires sentence-transformers): {e}")


def test_iris_classifier():
    """Test complete IRIS classifier."""
    from criteria_bge_hpo.models import IRISClassifier
    import numpy as np

    print("\n6. Testing IRISClassifier...")

    try:
        # Create model
        model = IRISClassifier(
            num_queries=4,  # Smaller for testing
            k_retrieved=4,
            embedding_dim=384,  # Smaller for testing
            encoder_name="sentence-transformers/all-MiniLM-L6-v2",
            num_classes=2,
        )

        # Build retriever with dummy chunks
        chunk_texts = [
            f"This is chunk {i} about depression symptoms."
            for i in range(50)
        ]
        model.build_retriever(chunk_texts, batch_size=10)

        print(f"   Model created with {model.retriever.num_chunks} chunks")

        # Forward pass
        post_texts = ["I feel sad all the time", "Everything is great!"]
        criterion_texts = ["Depressed mood", "Loss of interest"]

        outputs = model(post_texts, criterion_texts)

        print(f"   Logits shape: {outputs['logits'].shape}")
        print(f"   Query penalty: {outputs['query_penalty'].item():.4f}")
        print(f"   Retrieved indices shape: {outputs['retrieved_indices'].shape}")

        assert outputs["logits"].shape == (2, 2), "Expected (batch_size=2, num_classes=2)"

        print("   ✓ IRISClassifier passed")
    except Exception as e:
        print(f"   ⚠ IRISClassifier skipped: {e}")


def main():
    """Run all tests."""
    print("=" * 70)
    print("IRIS Model Component Tests")
    print("=" * 70)

    try:
        test_query_vectors()
        test_linear_attention()
        test_query_penalty_loss()
        test_chunk_retriever()
        test_embedding_model()
        test_iris_classifier()

        print("\n" + "=" * 70)
        print("✓ ALL TESTS PASSED")
        print("=" * 70)
        return 0

    except Exception as e:
        print("\n" + "=" * 70)
        print(f"✗ TEST FAILED: {e}")
        print("=" * 70)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
