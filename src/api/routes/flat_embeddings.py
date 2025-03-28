from api.routes.common import build_classification_router
from classify.flat_embeddings import EmbeddingFlatClassifier

router = build_classification_router(
    prefix="/flat-embeddings",
    tag="Flat Embeddings",
    classifier_cls=EmbeddingFlatClassifier,
)
