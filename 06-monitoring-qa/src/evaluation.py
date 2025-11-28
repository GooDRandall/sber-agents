import logging
from typing import Optional, Dict, Any
import pandas as pd
from langsmith import Client
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    AnswerCorrectness,
    AnswerSimilarity,
    ContextRecall,
    ContextPrecision,
)
from ragas.metrics.base import MetricWithLLM, MetricWithEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig
from config import config
import rag

logger = logging.getLogger(__name__)

# Глобальные инициализированные метрики
_ragas_metrics = None
_ragas_run_config = None

def init_ragas_metrics():
    """
    Инициализация RAGAS метрик (один раз)
    
    По образцу референсного ноутбука (раздел 5.1)
    """
    global _ragas_metrics, _ragas_run_config
    
    if _ragas_metrics is not None:
        return _ragas_metrics, _ragas_run_config
    
    logger.info("Initializing RAGAS metrics...")
    
    # Настройка LLM для RAGAS
    if config.RAGAS_LLM_PROVIDER == "ollama":
        from langchain_ollama import ChatOllama
        langchain_llm = ChatOllama(
            model=config.RAGAS_LLM_MODEL,
            temperature=0,
            base_url=config.OLLAMA_BASE_URL
        )
        logger.info(f"Using Ollama LLM for RAGAS: {config.RAGAS_LLM_MODEL} at {config.OLLAMA_BASE_URL}")
    else:
        from langchain_openai import ChatOpenAI
        kwargs = {
            "model": config.RAGAS_LLM_MODEL,
            "temperature": 0
        }
        if config.OPENAI_BASE_URL:
            kwargs["base_url"] = config.OPENAI_BASE_URL
        langchain_llm = ChatOpenAI(**kwargs)
        logger.info(f"Using OpenAI-compatible LLM for RAGAS: {config.RAGAS_LLM_MODEL}")
    
    # Настройка embeddings для RAGAS на основе провайдера
    if config.RAGAS_EMBEDDING_PROVIDER == "ollama":
        from langchain_ollama import OllamaEmbeddings
        langchain_embeddings = OllamaEmbeddings(
            model=config.RAGAS_EMBEDDING_MODEL,
            base_url=config.OLLAMA_BASE_URL
        )
        logger.info(f"Using Ollama embeddings for RAGAS: {config.RAGAS_EMBEDDING_MODEL} at {config.OLLAMA_BASE_URL}")
    elif config.RAGAS_EMBEDDING_PROVIDER == "huggingface":
        from langchain_huggingface import HuggingFaceEmbeddings
        langchain_embeddings = HuggingFaceEmbeddings(
            model_name=config.RAGAS_EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        logger.info(f"Using HuggingFace embeddings for RAGAS: {config.RAGAS_EMBEDDING_MODEL}")
    else:
        from langchain_openai import OpenAIEmbeddings
        langchain_embeddings = OpenAIEmbeddings(model=config.RAGAS_EMBEDDING_MODEL)
        logger.info(f"Using OpenAI embeddings for RAGAS: {config.RAGAS_EMBEDDING_MODEL}")
    
    # Создаем метрики
    metrics = [
        Faithfulness(),
        AnswerRelevancy(strictness=1),
        AnswerCorrectness(),
        AnswerSimilarity(),
        ContextRecall(),
        ContextPrecision(),
    ]
    
    # Инициализируем метрики
    ragas_llm = LangchainLLMWrapper(langchain_llm)
    ragas_embeddings = LangchainEmbeddingsWrapper(langchain_embeddings)
    
    for metric in metrics:
        if isinstance(metric, MetricWithLLM):
            metric.llm = ragas_llm
        if isinstance(metric, MetricWithEmbeddings):
            metric.embeddings = ragas_embeddings
        run_config = RunConfig()
        metric.init(run_config)
    
    # Настройки для выполнения
    # Уменьшаем max_workers для снижения нагрузки на API и избежания rate limiting
    run_config = RunConfig(
        max_workers=2,  # Уменьшено с 4 до 2 для снижения rate limiting
        max_wait=180,
        max_retries=5  # Увеличено с 3 до 5 для лучшей обработки временных ошибок
    )
    
    _ragas_metrics = metrics
    _ragas_run_config = run_config
    
    logger.info(f"RAGAS metrics initialized: {', '.join([m.name for m in metrics])}")
    return _ragas_metrics, _ragas_run_config

def check_dataset_exists(dataset_name: str) -> bool:
    """
    Проверка существования датасета в LangSmith
    
    Args:
        dataset_name: имя датасета
    
    Returns:
        True если датасет существует
    """
    if not config.LANGSMITH_API_KEY:
        logger.error("LANGSMITH_API_KEY not set")
        return False
    
    try:
        client = Client()
        datasets = list(client.list_datasets(dataset_name=dataset_name))
        return len(datasets) > 0
    except Exception as e:
        logger.error(f"Error checking dataset: {e}")
        return False

def evaluate_dataset(dataset_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Главная функция evaluation RAG системы
    
    По образцу референсного ноутбука (раздел 5.2):
    1. Запуск эксперимента в LangSmith с blocking=False и сбор данных
    2. RAGAS batch evaluation
    3. Загрузка метрик как feedback в LangSmith
    
    Args:
        dataset_name: имя датасета (по умолчанию из конфига)
    
    Returns:
        dict с результатами evaluation
    """
    if not config.LANGSMITH_API_KEY:
        raise ValueError("LANGSMITH_API_KEY not set. Cannot run evaluation.")
    
    if dataset_name is None:
        dataset_name = config.LANGSMITH_DATASET
    
    logger.info(f"Starting evaluation for dataset: {dataset_name}")
    
    # Проверяем существование датасета
    if not check_dataset_exists(dataset_name):
        raise ValueError(f"Dataset '{dataset_name}' not found in LangSmith")
    
    # Инициализируем метрики
    ragas_metrics, ragas_run_config = init_ragas_metrics()
    
    client = Client()
    
    # ========== Шаг 1: Запуск эксперимента и сбор данных ==========
    logger.info("\n[1/3] Running experiment and collecting data...")
    
    # Создаем target функцию для нашего RAG
    def target(inputs: dict) -> dict:
        """Target функция для evaluation"""
        question = inputs["question"]
        
        # Используем существующую RAG цепочку
        # Передаем только вопрос (без истории для evaluation)
        from langchain_core.messages import HumanMessage
        result = rag.get_rag_chain().invoke({"messages": [HumanMessage(content=question)]})
        
        return {
            "answer": result["answer"],
            "documents": result["documents"]
        }
    
    # Собираем данные во время выполнения evaluate
    questions = []
    answers = []
    contexts_list = []
    ground_truths = []
    run_ids = []
    
    # evaluate() с blocking=False возвращает итератор
    for result in client.evaluate(
        target,
        data=dataset_name,
        evaluators=[],
        experiment_prefix="rag-evaluation",
        metadata={
            "approach": "RAGAS batch evaluation + LangSmith feedback",
            "model": config.MODEL,
            "embedding_model": config.EMBEDDING_MODEL,
        },
        blocking=False,
    ):
        run = result["run"]
        example = result["example"]
        
        # Получаем данные
        question = run.inputs.get("question", "")
        answer = run.outputs.get("answer", "")
        documents = run.outputs.get("documents", [])
        contexts = [doc.page_content if hasattr(doc, 'page_content') else str(doc) for doc in documents]
        ground_truth = example.outputs.get("answer", "") if example else ""
        
        questions.append(question)
        answers.append(answer)
        contexts_list.append(contexts)
        ground_truths.append(ground_truth)
        run_ids.append(str(run.id))
    
    logger.info(f"Experiment completed, collected {len(questions)} examples")
    
    # ========== Шаг 2: RAGAS evaluation ==========
    logger.info("\n[2/3] Running RAGAS evaluation...")
    
    # Создаем Dataset для RAGAS
    ragas_dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts_list,
        "ground_truth": ground_truths
    })
    
    # Запускаем evaluation
    ragas_result = evaluate(
        ragas_dataset,
        metrics=ragas_metrics,
        run_config=ragas_run_config,
    )
    
    ragas_df = ragas_result.to_pandas()
    
    logger.info("RAGAS evaluation completed")
    
    # Вычисляем средние значения метрик
    metrics_summary = {}
    
    # Проверяем наличие nan значений и логируем статистику
    logger.info("\nDetailed metrics statistics:")
    for metric in ragas_metrics:
        if metric.name in ragas_df.columns:
            metric_series = ragas_df[metric.name]
            nan_count = metric_series.isna().sum()
            valid_count = metric_series.notna().sum()
            total_count = len(metric_series)
            
            if nan_count > 0:
                logger.warning(
                    f"  {metric.name}: {nan_count}/{total_count} values are nan "
                    f"(likely due to API errors or rate limiting)"
                )
            
            # Вычисляем среднее только по валидным значениям
            if valid_count > 0:
                avg_score = metric_series.mean()  # mean() автоматически игнорирует nan
                metrics_summary[metric.name] = avg_score
                logger.info(f"  {metric.name}: {avg_score:.3f} (valid: {valid_count}/{total_count})")
            else:
                logger.error(f"  {metric.name}: All values are nan - metric failed completely")
                metrics_summary[metric.name] = float('nan')
    
    # ========== Шаг 3: Загрузка feedback в LangSmith ==========
    logger.info("\n[3/3] Uploading feedback to LangSmith...")
    
    feedback_count = 0
    skipped_count = 0
    
    for idx, run_id in enumerate(run_ids):
        row = ragas_df.iloc[idx]
        
        for metric in ragas_metrics:
            if metric.name in row:
                score = row[metric.name]
                
                # Пропускаем nan значения - не загружаем их в LangSmith
                if pd.isna(score):
                    skipped_count += 1
                    logger.debug(f"Skipping nan value for {metric.name} in run {run_id}")
                    continue
                
                try:
                    client.create_feedback(
                        run_id=run_id,
                        key=metric.name,
                        score=float(score),
                        comment=f"RAGAS metric: {metric.name}"
                    )
                    feedback_count += 1
                except Exception as e:
                    logger.error(f"Failed to create feedback for {metric.name} in run {run_id}: {e}")
                    skipped_count += 1
    
    logger.info(f"Feedback uploaded: {feedback_count} successful, {skipped_count} skipped (nan or errors)")
    
    return {
        "dataset_name": dataset_name,
        "num_examples": len(questions),
        "metrics": metrics_summary,
        "ragas_result": ragas_result,
        "run_ids": run_ids
    }

