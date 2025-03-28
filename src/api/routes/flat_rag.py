from api.routes.common import build_classification_router
from classify.flat_rag import RAGFlatClassifier

router = build_classification_router(
    prefix="/flat-rag",
    tag="Flat RAG",
    classifier_cls=RAGFlatClassifier,
)
