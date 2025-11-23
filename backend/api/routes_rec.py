from fastapi import APIRouter, Depends, Request, HTTPException

from ..content_recommender import ContentKNNRecommender
from ..metrics import evaluate_user_recommendations

router = APIRouter(tags=["recommendations"])


def get_recommender(request: Request) -> ContentKNNRecommender:
    recommender = getattr(request.app.state, "recommender", None)
    if recommender is None:
        raise HTTPException(status_code=500, detail="Recommender não inicializado")
    return recommender


@router.get("/ping")
def ping():
    return {"status": "ok"}


@router.get("/recommendations")
def get_recommendations(
    user_id: int,
    k: int = 10,
    recommender: ContentKNNRecommender = Depends(get_recommender),
):
    """
    Retorna recomendações + métricas (precision/recall/f1) para o usuário.
    As métricas são calculadas fazendo um split interno (train/test) nos
    likes desse usuário, só para efeito de avaliação.
    """
    result = evaluate_user_recommendations(
        
        recommender=recommender,
        user_id=user_id,
        k=k,
    )

    if not result.get("ok", False):
        raise HTTPException(status_code=400, detail=result.get("message", "Erro na avaliação"))


    response = {
        "user_id": result["user_id"],
        "k": result["k"],
        "metrics": result["metrics"],          
        "total_likes": result["total_likes"],
        "train_likes": result["train_likes"],
        "test_likes": result["test_likes"],
        "recommendations": result["recommendations"], 
    }

    return response
