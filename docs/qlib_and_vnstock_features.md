# Useful Features for StocketAI

## Table of Contents

- [Overview](#overview)
- [Data Fetching Features (vnstock)](#data-fetching-features-vnstock)
  - [Core Data Sources](#core-data-sources)
  - [Advanced Data Features](#advanced-data-features)
  - [API Components](#api-components)
- [Model Training & Prediction Features (qlib)](#model-training--prediction-features-qlib)
  - [Data Layer Features](#data-layer-features)
  - [Model Features](#model-features)
  - [Workflow Features](#workflow-features)
  - [Infrastructure Features](#infrastructure-features)
- [Integration Points for StocketAI](#integration-points-for-stocketai)
  - [Data Pipeline](#data-pipeline)
  - [Model Development](#model-development)
  - [Prediction & Evaluation](#prediction--evaluation)
- [Advanced Features Requiring Additional Libraries](#advanced-features-requiring-additional-libraries)
  - [1. Advanced Natural Language Processing & Sentiment Analysis](#1-advanced-natural-language-processing--sentiment-analysis)
  - [2. Social Media Data Collection & Analysis](#2-social-media-data-collection--analysis)

## Overview

This document outlines the key features from vnstock and qlib libraries that can be leveraged to build StocketAI, a comprehensive stock price prediction system using extensible data acquisition frameworks. The system supports any stock universe (VN30, custom lists, or any configurable stock universe) and emphasizes research flexibility, reproducibility, and quantitative finance methodologies with support for multiple data providers and extensible architecture.

## Data Fetching Features (vnstock)

### Core Data Sources

| Feature | Description | Reference |
|---------|-------------|-----------|
| Multiple Data Providers | VCI, TCBS, MSN, FMARKET for comprehensive data coverage | [Reference](docs/vnstock_api_documentation.md#data-sources) |
| Historical Price Data | Daily, weekly, monthly OHLCV data with adjustable time intervals | [Reference](docs/vnstock_api_documentation.md#historical-price-data) |
| Intraday Data | Real-time tick-by-tick trading data for high-frequency analysis | [Reference](docs/vnstock_api_documentation.md#intraday-data) |
| Company Information | Overview, profile, shareholders, officers, corporate events | [Reference](docs/vnstock_api_documentation.md#company-information) |
| Financial Reports | Balance sheets, income statements, cash flow statements, financial ratios | [Reference](docs/vnstock_api_documentation.md#financial-reports) |

### Advanced Data Features

| Feature | Description | Reference |
|---------|-------------|-----------|
| Stock Screening | 1700+ criteria for filtering stocks by financial metrics (P/E, P/B, ROE, etc.) | [Reference](docs/vnstock_api_documentation.md#stock-screening) |
| Market Indices | VN30, VN100, HNX, UPCOM tracking and constituent data | [Reference](docs/vnstock_api_documentation.md#market-indices) |
| International Markets | Forex rates, cryptocurrencies, global indices | [Reference](docs/vnstock_api_documentation.md#international-markets) |
| Derivatives & Fixed Income | Futures, covered warrants, government/corporate bonds | [Reference](docs/vnstock_api_documentation.md#derivatives--fixed-income) |

### API Components

| Component | Description | Reference |
|-----------|-------------|-----------|
| Vnstock Class | Unified interface for all data types | [Reference](docs/vnstock_api_documentation.md#vnstock) |
| Quote Class | Price and trading data functionality | [Reference](docs/vnstock_api_documentation.md#quote) |
| Company Class | Corporate information and events | [Reference](docs/vnstock_api_documentation.md#company) |
| Finance Class | Financial statements and ratios | [Reference](docs/vnstock_api_documentation.md#finance) |
| Listing Class | Market listings and symbol information | [Reference](docs/vnstock_api_documentation.md#listing) |
| Trading Class | Real-time trading data | [Reference](docs/vnstock_api_documentation.md#trading) |
| Screener Class | Advanced stock screening capabilities | [Reference](docs/vnstock_api_documentation.md#screener) |

## Model Training & Prediction Features (qlib)

### Data Layer Features

| Feature | Description | Reference |
|---------|-------------|-----------|
| Qlib Format Data | High-performance .bin format for financial data storage | [Reference](docs/qlib/component/data.rst#qlib-format-data) |
| Data Conversion | Scripts to convert CSV/Parquet to qlib format | [Reference](docs/qlib/component/data.rst#converting-csv-format-into-qlib-format) |
| Expression Engine | Formulaic alpha creation with operators (Ref, Mean, etc.) | [Reference](docs/qlib/component/data.rst#feature) |
| Data Processing | Normalization, NaN handling, feature engineering | [Reference](docs/qlib/component/data.rst#processor) |
| Data Handlers | Pre-built handlers like Alpha158, Alpha360 with 158/360 features | [Reference](docs/qlib/component/data.rst#data-handler) |
| Dataset Management | Flexible dataset preparation for different model types | [Reference](docs/qlib/component/data.rst#dataset) |

### Model Features

Qlib provides a comprehensive model zoo with various machine learning algorithms optimized for quantitative finance. Models are categorized by their learning approach and architecture.

#### Tree-based Models

| Model | Description | Best Use Case |
|-------|-------------|---------------|
| **LightGBM (LGBModel)** | Gradient boosting framework optimized for speed and memory efficiency with tree-based learning | High-performance baseline for stock return prediction with tabular financial features |
| **XGBoost** | Scalable and flexible gradient boosting implementation with regularization | Robust prediction when dealing with noisy financial data and feature interactions |

#### Neural Network Models

| Model | Description | Best Use Case |
|-------|-------------|---------------|
| **MLP (Multi-Layer Perceptron)** | Feed-forward neural network with multiple hidden layers for non-linear pattern learning | Capturing complex non-linear relationships in financial indicators and technical features |
| **LSTM (Long Short-Term Memory)** | Recurrent neural network designed to learn long-term dependencies in sequential data | Modeling temporal patterns in stock price movements and time series forecasting |
| **GRU (Gated Recurrent Unit)** | Simplified LSTM variant with fewer parameters while maintaining performance | Efficient sequential modeling for high-frequency trading signals and market microstructure |
| **TCN (Temporal Convolutional Network)** | Convolutional neural network for sequence modeling with parallel processing | Fast training on large-scale financial time series with local temporal patterns |

#### Transformer-based Models

| Model | Description | Best Use Case |
|-------|-------------|---------------|
| **TFT (Temporal Fusion Transformer)** | Attention-based architecture combining LSTM and transformer mechanisms | Multi-horizon forecasting with interpretable attention on different time scales |
| **TRA (Transformer with Attention)** | Transformer architecture with temporal relational reasoning | Capturing complex market dynamics and cross-asset relationships |
| **ADARNN (Adaptive Dynamic Attention RNN)** | RNN with adaptive attention mechanisms for dynamic feature selection | Markets with changing importance of different features over time |
| **HIST (Hierarchical Inference Strategy Transformer)** | Transformer with hierarchical structure for multi-scale analysis | Analyzing market trends at different time granularities simultaneously |
| **IGMTF (Improved Graph Multi-Task Transformer)** | Graph-based transformer for multi-task learning across assets | Portfolio optimization with inter-stock relationships and correlations |

#### Specialized Models

| Model | Description | Best Use Case |
|-------|-------------|---------------|
| **KRNN (Kernel Recurrent Neural Network)** | RNN with kernel methods for enhanced pattern recognition | Non-linear time series patterns that traditional RNNs struggle with |
| **Sandwich** | Multi-layer architecture combining different model types | Ensemble learning combining strengths of different approaches |
| **ADD (Adaptive Dynamic Deep learning)** | Deep learning model with adaptive parameter adjustment | Highly volatile markets requiring dynamic model adaptation |
| **DDG-DA (Dynamic Data Generation - Domain Adaptation)** | Meta-learning approach for domain adaptation across market conditions | Transfer learning between different market regimes or geographies |

#### Custom Models

| Model | Description | Best Use Case |
|-------|-------------|---------------|
| **Custom Model Support** | Extensible framework for implementing proprietary algorithms | Domain-specific models incorporating unique trading strategies or proprietary data |
| **Reinforcement Learning Models** | RL-based trading agents for decision-making under uncertainty | Optimal execution strategies and dynamic portfolio rebalancing |

| Feature | Description | Reference |
|---------|-------------|-----------|
| Learning Paradigms | Supervised learning and reinforcement learning | [Reference](docs/qlib/component/model.rst#introduction) |
| Model Training | Automated training with configurable parameters | [Reference](docs/qlib/component/model.rst#example) |
| Prediction Scoring | Stock rating and forecasting capabilities | [Reference](docs/qlib/component/model.rst#introduction) |
| Model Evaluation | Information Coefficient (IC), backtesting metrics | [Reference](docs/qlib/component/model.rst#introduction) |

### Workflow Features

| Feature | Description | Reference |
|---------|-------------|-----------|
| Automated Workflow | qrun command for end-to-end execution | [Reference](docs/qlib/component/workflow.rst#complete-example) |
| Experiment Management | Complete tracking of training, inference, and evaluation | [Reference](docs/qlib/component/workflow.rst#introduction) |
| Backtesting | Portfolio analysis with various strategies | [Reference](docs/qlib/component/workflow.rst#task-section) |
| Signal Analysis | Forecast signal evaluation and risk analysis | [Reference](docs/qlib/component/workflow.rst#task-section) |
| Strategy Implementation | Top-k dropout, custom trading strategies | [Reference](docs/qlib/component/strategy.rst#implemented-strategy) |

### Infrastructure Features

| Feature | Description | Reference |
|---------|-------------|-----------|
| Caching System | Memory and disk caching for performance optimization | [Reference](docs/qlib/component/data.rst#cache) |
| Data Health Checking | Automated validation of data integrity | [Reference](docs/qlib/component/data.rst#checking-the-health-of-the-data) |
| Multi-Market Support | China and US stock modes | [Reference](docs/qlib/component/data.rst#multiple-stock-modes) |
| Scalable Architecture | Loosely-coupled components for customization | [Reference](docs/qlib/component/workflow.rst#introduction) |

## Integration Points for StocketAI

### Data Pipeline

1. **Company Universe Definition**: Define and manage stock universe (VN30, custom lists, or any configurable stock universe) with extensible provider interface [[Reference](vnstock_api_documentation.md#vn30-constituent-list)]
2. **Multi-Provider Data Acquisition**: Use vnstock to fetch data from multiple providers (VCI, TCBS, MSN, FMARKET) with automatic fallback mechanisms
3. **Data Conversion**: Convert vnstock data to qlib format using dump_bin.py for high-performance processing [[Reference](docs/qlib/component/data.rst#converting-csv-format-into-qlib-format)]
4. **Feature Engineering**: Apply qlib data handlers for technical and fundamental feature generation [[Reference](docs/qlib/component/data.rst#data-handler)]
5. **Model Training**: Train models on processed datasets with cross-validation and hyperparameter optimization [[Reference](docs/qlib/component/model.rst#example)]

### Model Development

1. **Baseline Models**: Utilize qlib's Alpha158/Alpha360 features as baseline for stock return prediction [[Reference](docs/qlib/component/data.rst#data-handler)]
2. **Custom Features**: Implement domain-specific features using vnstock financial data and provider abstraction [[Reference](vnstock_api_documentation.md#financial-reports)]
3. **Model Comparison**: Train multiple models (GBDT, Neural Networks, Transformer-based) for comprehensive comparison [[Reference](docs/qlib/component/model.rst#custom-model)]
4. **Hyperparameter Optimization**: Optimize model performance using qlib's automated workflow system [[Reference](docs/qlib/component/workflow.rst#task-section)]

### Prediction & Evaluation

1. **Multi-Horizon Forecasting**: Generate price predictions for 1, 3, 6 month horizons with confidence intervals [[Reference](docs/qlib/component/model.rst#introduction)]
2. **Portfolio Backtesting**: Backtest strategies using qlib's portfolio analysis with trading constraints and risk management [[Reference](docs/qlib/component/strategy.rst#usage--example)]
3. **Performance Evaluation**: Evaluate using IC, Sharpe ratio, maximum drawdown, and directional accuracy metrics [[Reference](docs/qlib/component/model.rst#introduction)]
4. **Comparative Analysis**: Compare model performance across different stock universes and market conditions [[Reference](docs/qlib/component/workflow.rst#task-section)]


## Advanced Features Requiring Additional Libraries

While qlib and vnstock provide comprehensive support for most quantitative investment features, certain advanced capabilities require integration with specialized libraries and frameworks. This section identifies gaps and suggests complementary tools.

### 1. Advanced Natural Language Processing & Sentiment Analysis

**Gap in Current Libraries**:
- Limited built-in Vietnamese language processing capabilities
- No integrated transformer-based sentiment analysis for financial text
- Missing aspect-based sentiment analysis for company-specific mentions

**Recommended Libraries**:
- **transformers** (Hugging Face): Pre-trained BERT/RoBERTa models for Vietnamese financial text
- **VNPy** or **underthesea**: Vietnamese NLP toolkit for tokenization and POS tagging
- **TextBlob** or **VADER**: Rule-based sentiment analysis with Vietnamese language support
- **spaCy** with Vietnamese models: Industrial-strength NLP with custom financial entity recognition

**Implementation Approach**:
```python
# Example integration with vnstock + transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from vnstock import Vnstock

# Load Vietnamese financial sentiment model
tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base")
model = AutoModelForSequenceClassification.from_pretrained("financial-sentiment-vi")

# Get news data from vnstock
stock = Vnstock().stock(symbol='ACB', source='VCI')
news_data = stock.company.news()

# Process sentiment
sentiments = []
for news in news_data:
    inputs = tokenizer(news['content'], return_tensors="pt", truncation=True)
    outputs = model(**inputs)
    sentiment_score = outputs.logits.softmax(dim=1)
    sentiments.append(sentiment_score)
```

### 2. Social Media Data Collection & Analysis

**Gap in Current Libraries**:
- No direct social media API integrations
- Limited real-time social media monitoring capabilities
- Missing Vietnamese social media platform integrations (Zalo, Facebook Vietnam)

**Recommended Libraries**:
- **Tweepy**: Twitter API integration for global social sentiment
- **facebook-sdk**: Facebook Graph API for Vietnamese market sentiment
- **Selenium** or **BeautifulSoup**: Web scraping for social media data
- **snscrape**: Social media scraping without API limitations
- **SocialMediaAPI** frameworks for multi-platform integration

**Suggested Platforms**:
- Twitter API for global financial discussions
- Facebook Graph API for Vietnamese social sentiment
- Reddit API (PRAW) for retail investor sentiment
- Custom Zalo API integrations for Vietnamese social media
