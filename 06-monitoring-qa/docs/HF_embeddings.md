# Интеграция HuggingFace Embeddings

Пошаговая инструкция по добавлению поддержки HuggingFace embeddings для retriever и RAGAS.

## 📋 Зачем это нужно

- **Бесплатно** - модели работают локально, не требуют API ключей
- **Быстрее** - нет задержек на API вызовы
- **Приватность** - данные не уходят на сторонние серверы
- **Гибкость** - можно использовать любые модели с HuggingFace

## 🔧 Шаг 1: Добавить зависимости

### `pyproject.toml`

Добавьте в dependencies:

```toml
dependencies = [
    # ... существующие ...
    "langchain-huggingface>=0.1.0",  # Интеграция HuggingFace с LangChain
    "sentence-transformers>=3.0.0",  # Модели embeddings
]
```

**Важно:** `langchain-huggingface` - это официальный пакет LangChain для работы с HuggingFace моделями. `sentence-transformers` - библиотека самих моделей embeddings.

Затем установите:
```bash
uv sync
```

## ⚙️ Шаг 2: Обновить конфигурацию

### `src/config.py`

Добавьте после строки с `EMBEDDING_MODEL`:

```python
class Config:
    # ... существующие настройки ...
    
    # Embeddings настройки
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
    EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "openai")  # 'openai' или 'huggingface'
    
    # ... остальное без изменений ...
    
    # RAGAS evaluation настройки
    RAGAS_LLM_MODEL = os.getenv("RAGAS_LLM_MODEL", "gpt-4o")
    RAGAS_EMBEDDING_MODEL = os.getenv("RAGAS_EMBEDDING_MODEL", "text-embedding-3-large")
    RAGAS_EMBEDDING_PROVIDER = os.getenv("RAGAS_EMBEDDING_PROVIDER", "openai")  # 'openai' или 'huggingface'
```

## 📝 Шаг 3: Обновить indexer.py

### `src/indexer.py`

Замените функцию `create_vector_store()`:

```python
def create_vector_store(chunks: list):
    """Создание векторного хранилища"""
    
    # Выбор embeddings на основе провайдера
    if config.EMBEDDING_PROVIDER == "huggingface":
        from langchain_huggingface import HuggingFaceEmbeddings
        embeddings = HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'},  # или 'cuda' если есть GPU
            encode_kwargs={'normalize_embeddings': True}
        )
        logger.info(f"Using HuggingFace embeddings: {config.EMBEDDING_MODEL}")
    else:
        from langchain_openai import OpenAIEmbeddings
        embeddings = OpenAIEmbeddings(
            model=config.EMBEDDING_MODEL
        )
        logger.info(f"Using OpenAI embeddings: {config.EMBEDDING_MODEL}")
    
    vector_store = InMemoryVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings
    )
    logger.info(f"Created vector store with {len(chunks)} chunks")
    return vector_store
```

## 🎯 Шаг 4: Обновить evaluation.py

### `src/evaluation.py`

В функции `init_ragas_metrics()` замените создание embeddings:

```python
def init_ragas_metrics():
    """Инициализация RAGAS метрик (один раз)"""
    global _ragas_metrics, _ragas_run_config
    
    if _ragas_metrics is not None:
        return _ragas_metrics, _ragas_run_config
    
    logger.info("Initializing RAGAS metrics...")
    
    # Настройка LLM для RAGAS (всегда OpenAI-совместимый)
    langchain_llm = ChatOpenAI(model=config.RAGAS_LLM_MODEL, temperature=0)
    
    # Настройка embeddings для RAGAS на основе провайдера
    if config.RAGAS_EMBEDDING_PROVIDER == "huggingface":
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
    
    # ... остальное без изменений ...
```

## 📄 Шаг 5: Обновить env.example

### `env.example`

Добавьте примеры с HuggingFace:

```bash
# === Вариант с HuggingFace Embeddings (бесплатно, локально) ===

# OpenRouter LLM + HuggingFace Embeddings
# OPENAI_API_KEY=sk-or-v1-...
# OPENAI_BASE_URL=https://openrouter.ai/api/v1
# MODEL=openai/gpt-oss-20b:free
# MODEL_QUERY_TRANSFORM=openai/gpt-oss-20b:free
# EMBEDDING_PROVIDER=huggingface
# EMBEDDING_MODEL=intfloat/multilingual-e5-base
# RAGAS_LLM_MODEL=openai/gpt-oss-20b:free
# RAGAS_EMBEDDING_PROVIDER=huggingface
# RAGAS_EMBEDDING_MODEL=intfloat/multilingual-e5-base

# OpenAI LLM + HuggingFace Embeddings (экономия на embeddings)
# OPENAI_API_KEY=sk-proj-...
# MODEL=gpt-4.1
# MODEL_QUERY_TRANSFORM=gpt-4.1
# EMBEDDING_PROVIDER=huggingface
# EMBEDDING_MODEL=intfloat/multilingual-e5-base
# RAGAS_LLM_MODEL=gpt-4.1
# RAGAS_EMBEDDING_PROVIDER=huggingface
# RAGAS_EMBEDDING_MODEL=intfloat/multilingual-e5-base
```

## 🎓 Рекомендуемые модели HuggingFace

### Для русского языка:

Семейство E5 (оптимальны для CPU):
• multilingual-e5-large-instruct (560M, 1.1GB) - Retrieval: 68.23, Rank #7
• multilingual-e5-base (278M, 1.1GB) - Retrieval: 67.14, Rank #32 ⭐ НАШ ВЫБОР
• multilingual-e5-small (118M, 449MB) - Retrieval: 65.85, Rank #37


## 🧪 Шаг 6: Тестирование

### Пример 1: Полностью бесплатная конфигурация

```bash
# .env
TELEGRAM_TOKEN=your_token
OPENAI_API_KEY=sk-or-v1-...
OPENAI_BASE_URL=https://openrouter.ai/api/v1
MODEL=openai/gpt-oss-20b:free
MODEL_QUERY_TRANSFORM=openai/gpt-oss-20b:free

# HuggingFace embeddings (бесплатно)
EMBEDDING_PROVIDER=huggingface
EMBEDDING_MODEL=intfloat/multilingual-e5-base

# RAGAS тоже с HuggingFace
RAGAS_LLM_MODEL=openai/gpt-oss-20b:free
RAGAS_EMBEDDING_PROVIDER=huggingface
RAGAS_EMBEDDING_MODEL=intfloat/multilingual-e5-base
```
### Запуск:

```bash
make run
```

При первом запуске модель скачается автоматически (~2GB для multilingual-e5-large).

### Проверка:

1. Отправьте боту вопрос
2. Проверьте логи - должно быть:
   ```
   INFO - Using HuggingFace embeddings: intfloat/multilingual-e5-large
   ```

3. Запустите evaluation:
   ```
   /evaluate_dataset
   ```

## ⚠️ Важные замечания

1. **Первый запуск медленный** - модель скачивается (до 2GB)
2. **Требуется RAM** - большие модели требуют 4-8GB RAM
3. **Совместимость с RAGAS** - работает через LangchainEmbeddingsWrapper
4. **Индексация медленнее** - чем через API, но дешевле
5. **Качество зависит от модели** - тестируйте разные модели


## ✅ Чеклист интеграции

- [ ] Добавлены зависимости `langchain-huggingface` и `sentence-transformers` в `pyproject.toml`
- [ ] Установлены зависимости через `uv sync`
- [ ] Обновлен `src/config.py` с `EMBEDDING_PROVIDER` и `RAGAS_EMBEDDING_PROVIDER`
- [ ] Обновлен `src/indexer.py` с условным выбором embeddings
- [ ] Обновлен `src/evaluation.py` с условным выбором embeddings для RAGAS
- [ ] Обновлен `env.example` с примерами HuggingFace
- [ ] Выбрана модель embeddings
- [ ] Настроен `.env` файл
- [ ] Протестирован запуск бота
- [ ] Протестирован `/index` - индексация работает
- [ ] Протестирован `/evaluate_dataset` - RAGAS работает

## 🎯 Готово!

Теперь проект поддерживает гибкую настройку embeddings - можно использовать:
- OpenAI API (платно, быстро, качественно)
- HuggingFace (бесплатно, локально, гибко)
- Гибридные варианты (OpenAI LLM + HuggingFace embeddings)

Выбирайте конфигурацию в зависимости от бюджета и требований!
