from api.routes.common import build_classification_router
from classify.hierarchical_embeddings import EmbeddingHierarchicalClassifier

router = build_classification_router(
    prefix="/hierarchical-embeddings",
    tag="Hierarchical embeddings",
    classifier_cls=EmbeddingHierarchicalClassifier,
)
