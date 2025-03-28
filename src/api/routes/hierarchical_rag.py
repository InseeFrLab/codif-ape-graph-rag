from api.routes.common import build_classification_router
from classify.hierarchical_rag import RAGHierarchicalClassifier

router = build_classification_router(
    prefix="/hierarchical-rag",
    tag="Hierarchical RAG",
    classifier_cls=RAGHierarchicalClassifier,
)
