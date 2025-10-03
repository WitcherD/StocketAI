
https://gemini.google.com/share/b4d03200c737

# **Expert Assessment of Qlib Models for Long-Horizon Stock Return Forecasting (T+126)**

## **I. Executive Synthesis: Optimal Strategies for  Alpha Generation**

### **1.1. Core Recommendation: The Hybrid Imperative for Long-Horizon Forecasting**

Forecasting stock returns over a 6-month horizon ( trading days) presents a unique set of challenges rooted in severe market non-stationarity and the necessity to capture complex, long-range temporal and cross-sectional dependencies.1 Predictive success in this domain is dictated not by raw magnitude of return, but by the stability and reliability of the alpha signal, conventionally measured by the Information Coefficient Information Ratio (ICIR) and the overall risk-adjusted return (Information Ratio, IR).2

Pure sequence models or stand-alone temporal models are frequently sub-optimal for this long horizon due to model decay caused by structural market shifts (Concept Drift).4 The analysis dictates that the most robust solution is a hybrid imperative: combining proactive adaptation mechanisms with advanced deep temporal architectures. Specifically, the optimal approach involves leveraging the sophisticated temporal modeling and multi-modal data fusion capabilities of the

**Temporal Fusion Transformer (TFT)**, whose long-term attention structure is highly effective for a 126-day lookahead 5, while stabilizing its training regimen using the proactive drift correction framework of

**DDG-DA (Data Distribution Generation for Predictable Concept Drift Adaptation)**.7

### **1.2. Summary of Performance Drivers and Projected Ranking**

For robust six-month alpha prediction, the primary metric shifts from instantaneous accuracy (Information Coefficient, IC) to resilience (ICIR and IR). The models are ranked below based on their theoretical capacity to maintain predictive stability and capture the underlying structural factors relevant to a strategic 6-month outlook.

**Theoretical Ranking for Long-Horizon Prediction (T+126):**

1. **DDG-DA (Proactive Adaptation/Meta-Learning):** This method holds the highest potential for maximizing risk-adjusted return (ICIR). Its ability to proactively model and prepare for future data distributions mitigates the performance degradation that inevitably occurs over a six-month period of market evolution.7  
2. **TFT (Multi-Horizon Attention):** Architecturally, TFT is superior for spanning the long prediction window. Its attention mechanisms effectively manage the temporal dependencies inherent in the 126-step forecast, and its dedicated fusion layers integrate static and known future covariates critical for long-term conditioning.5  
3. **HIST (Graph-based Relationality):** Effective six-month alpha is often driven by sustained macroeconomic or sectoral factors. HIST excels by modeling dynamic cross-sectional linkages through shared and latent concepts, capturing these structural trends that persist over an extended horizon.9  
4. **ADARNN (Reactive Adaptation):** ADARNN offers a strong, proven adaptive approach.11 However, its reactive mechanism, relying on aligning representations between older and newer data, is considered less potent for deep distributional shifts than DDG-DA's explicit prediction of the future market regime.11  
5. **IGMTF (Feature Transfer):** While focusing on efficient and decorrelated feature representation, IGMTF's primary utility is optimizing the correlation structure in multivariate time series data.12 For pure long-term prediction, this benefit is highly dependent on the quality of underlying feature engineering and is generally secondary to the adaptive or deep temporal modeling capabilities of the top contenders.  
6. **Sandwich (Emerging Hybrid):** Released recently in Qlib (May 2023\) 2, this model's specific financial architecture is not publicly documented with the same academic rigor as the others. Its ranking is pending rigorous empirical validation against established Qlib benchmarks like Alpha360/Alpha158.2

## **II. The Non-Stationary Challenge of Six-Month Forecasting**

### **2.1. Defining Long-Horizon Prediction () and the Role of Alpha**

Quantitative investment platforms such as Qlib are fundamentally designed to generate predictive scores, or *alpha signals*, which forecast the future excess return of stocks, rather than predicting the exact stock price.13 The objective is to rank stocks based on their expected outperformance, thereby informing portfolio construction.14

The prediction horizon, set at approximately 126 trading days (six months), fundamentally alters the modeling requirement. This horizon shifts the focus away from modeling high-frequency noise, intraday volatility, or short-term technical indicators. Instead, success depends on correctly capturing underlying, persistent drivers: changes in investor expectations, long-term confidence, structural regime shifts, and sustained macroeconomic trends.1 The factors driving stock movement over six months are substantially different from those influencing a one-day or one-week forecast.

### **2.2. Concept Drift and Market Non-Stationarity**

Financial markets exhibit pronounced non-stationary characteristics, meaning the statistical properties of the data stream change over time.15 This phenomenon, known as Concept Drift, defines a scenario where the joint distribution

 is not static.15 This instability can manifest in various ways, including abrupt changes in volatility regimes or shifts in macroeconomic trends.4

For the  prediction horizon, the implications of Concept Drift are severe. A model trained on a fixed historical window (even spanning 1-2 years) is practically guaranteed to encounter significant distributional shifts between its training period and the six-month future test period.4 The standard approach of traditional methods, which typically involve retraining only after drift has been detected, is inherently reactive.4 This latency in adaptation renders models trained using fixed historical weights obsolete and necessitates specialized adaptive techniques like DDG-DA and ADARNN.8 ADARNN specifically names this challenge

**Temporal Covariate Shift (TCS)**, acknowledging that market factors may diverge drastically from historical observations.11

### **2.3. Qlib Evaluation Framework for Long-Term Benchmarking**

Qlib provides a modular framework for quantitative research, including standardized metrics necessary for evaluating long-term alpha performance under realistic simulation environments.2

* **Information Coefficient (IC) and ICIR:** The IC measures the instantaneous rank correlation between the model's predicted score (alpha) and the actual future return.3 The  
  **ICIR (Information Coefficient Information Ratio)** is the mean IC divided by its standard deviation over the evaluation period, making it the essential metric for assessing the stability and robustness of the predictive signal over time.3 For  
   forecasting, models must demonstrate a consistently high ICIR, indicating reliable signal quality across different market conditions.  
* **Annualized Return (AR) and Information Ratio (IR):** These metrics quantify the utility of the signal when applied to portfolio construction and backtesting, accounting for transaction costs.2 AR measures the simulated profit, while the  
  **Information Ratio (IR)** measures the risk-adjusted return (excess return relative to benchmark divided by tracking error).2 For strategic, long-term investors, the IR is paramount as it validates the practical, risk-managed utility of the generated alpha signal, especially since accumulated profit metrics in Qlib are calculated by summation to avoid exponential skewing.3 The benchmark backtesting also includes critical measures such as Maximum Drawdown (MDD).2

## **III. Taxonomy I: Architectures Optimized for Market Adaptation**

The critical challenge for long-horizon prediction is survival—that is, maintaining predictive power when the market fundamentally changes. DDG-DA and ADARNN are explicitly designed to overcome Concept Drift.

### **3.1. DDG-DA: Proactive Distribution Generation for Predictable Concept Drift Adaptation**

DDG-DA (Data Distribution Generation for Predictable Concept Drift Adaptation) represents a state-of-the-art solution within Qlib's ecosystem, introduced in early 2022 as part of the meta-learning framework.18 Its fundamental objective is to move beyond reactive model adaptation by proactively modeling and predicting the evolution of the market's data distribution.7

#### **Mechanism and Stability for** 

The process operates in a meta-learning loop:

1. A predictor (meta-model) is trained to estimate the future data distribution, , based on the history of distribution shifts.15  
2. This prediction is utilized to generate optimized training samples for the subsequent forecasting task. This generation occurs via weighted sampling guided by a differentiable distribution distance metric, which is proven to be equivalent to KL-divergence.7  
3. The final forecasting model is then trained on this *future-aligned* resampled dataset.19

This methodology is rooted in the recognition that not all concept drift is random; some underlying factors influencing environmental evolution are predictable.7 By training the model on data that mimics the forthcoming test data distribution, DDG-DA ensures the forecasting model is maximally prepared for the market regime six months ahead, leading to substantially better signal quality (IC) and enhanced portfolio performance (AR/IR) compared to standard rolling retraining or reactive methods.8

The specialized nature of DDG-DA requires significant computational resources, often leading to large training data sizes and complexity in hyperparameter tuning.20 DDG-DA converts distribution distance optimization into a bi-level optimization problem using a proxy model.15 However, the requirement is to forecast over a six-month interval, meaning the full meta-learning loop and retraining procedures are typically executed far less frequently (e.g., quarterly or semi-annually, rather than daily). This low frequency of required training updates makes the inherent overhead of DDG-DA's meta-learning component practically acceptable for strategic, long-term implementation, effectively transforming a high-cost algorithm into an economically viable strategy for the

 horizon.

Furthermore, DDG-DA is designed to be model-agnostic in its distribution adaptation role.15 It stabilizes the

*data environment*, not the model structure itself. This unique capability means DDG-DA can function as a foundational stability layer for more complex models—including high-performance, deep temporal models like TFT or relational models like HIST—which might otherwise suffer from severe overfitting or catastrophic forgetting in highly dynamic markets. Implementing DDG-DA to preprocess data for a high-fidelity model like TFT offers the most resilient approach, combining maximum temporal prediction capability with maximum distributional stability.

### **3.2. ADARNN: Adaptive Learning and Forecasting for Time Series**

ADARNN (Adaptive Learning and Forecasting for Time Series), released in Qlib in November 2021 2, is a robust recurrent neural network (RNN)-based model explicitly designed to mitigate the effects of non-stationarity, or Temporal Covariate Shift (TCS), in financial time series.11

#### **Mechanism and Benchmark Context**

ADARNN leverages GRU layers combined with transfer-loss techniques.11 Its operation is divided into two phases: Temporal Distribution Characterization and Temporal Distribution Matching. The latter phase systematically bridges the distributional gap between past (older) regimes and newer market data using distance-based alignment metrics, such as Maximum Mean Discrepancy (MMD), CORAL, or COSINE similarity.11 This mechanism allows the model to continuously adapt to gradual changes in the feature space over time. ADARNN has been consistently benchmarked within the Qlib environment; for example, it has demonstrated a competitive annualized return of

 on the Alpha360 dataset.22

ADARNN’s stability profile for  is high, placing it above non-adaptive models. However, its core functionality involves aligning historical representations with recent data characteristics, making it inherently reactive to observed distribution shifts.11 This contrasts with the proactive approach of DDG-DA, which aims to predict the distributional characteristics of the

*future* test window.15 For a long prediction horizon like six months, where major economic or structural shifts are likely to occur, the capacity to anticipate the upcoming market regime (DDG-DA) is theoretically more valuable than optimizing alignment between past observed regimes (ADARNN).

The long-term reliability of the ADARNN alpha signal is heavily influenced by the hyperparameterization of the distribution matching algorithm. The choice and specific tuning of the MMD, CORAL, or COSINE distance metric directly govern how aggressively the model adjusts its representation space.11 Poorly tuned distance metrics may lead to over-correction, where the model inappropriately discards valuable historical information, or under-correction, where the distributional gap remains wide. Achieving a stable long-term

 signal thus requires extensive hyperparameter optimization focusing on the stability of the ICIR across multiple non-stationary periods.

## **IV. Taxonomy II: Architectures Optimized for Relational and Temporal Depth**

For strategic  forecasting, architectural depth—the ability to model long-range temporal dependencies and complex cross-sectional relationships—is essential.

### **4.1. TFT: Temporal Fusion Transformer for Multi-Horizon Forecasting**

The Temporal Fusion Transformer (TFT) is a sophisticated attention-based architecture specifically engineered for interpretable multi-horizon time series forecasting.5 Its design makes it uniquely suited for the

 problem by allowing the model to look far into the past and predict far into the future efficiently.

#### **Temporal Depth and Multi-Modal Fusion**

The TFT architecture combines two distinct mechanisms to handle time: sequence-to-sequence Recurrent Neural Networks (specifically LSTMs) handle short-term local processing, while an interpretable multi-head self-attention layer is employed to capture non-local, long-term dependencies across the sequence.6 This multi-head attention is crucial for

 because it mitigates the vanishing dependency problem common in sequential models (LSTMs/GRUs), ensuring that relevant information from 126 days in the past remains accessible and weighted correctly for the forecast.5

A primary advantage of TFT is its ability to handle heterogeneous, multi-modal input data.26 This includes:

1. **Static Covariates:** Time-invariant metadata about the asset (e.g., sector classification, market capitalization, permanent company characteristics).6  
2. **Known Future Inputs:** Time-varying features whose values are known across the entire prediction horizon (e.g., calendar effects, future trading holidays).6  
3. **Unknown Past/Future Inputs:** Standard market factors (e.g., historical returns, volumes).6

Gating mechanisms, implemented as Gated Residual Networks, are extensively used within the TFT to selectively suppress unnecessary computations, while Variable Selection Networks (VSNs) ensure only the most relevant features are processed at each time step, significantly boosting interpretability.6

The successful application of TFT to  prediction relies heavily on effectively integrating structural information. Long-term alpha signals are intrinsically linked to persistent company characteristics and sector groupings.30 By incorporating these structural attributes via dedicated static covariate encoders, TFT can condition its long-range temporal dynamics on stable identifiers, ensuring that the 6-month prediction is structurally sound. To implement this correctly in a Qlib workflow, the user must explicitly define the 6-month window by setting the model parameter

output\_chunk\_length to  trading days.29 Failure to configure this parameter appropriately will prevent the model from performing the required multi-horizon prediction.

### **4.2. HIST: Graph-based Framework for Concept Mining**

HIST (Hidden Information Aggregation for Stock Trend Forecasting), released in April 2022 2, is a Graph-based framework that moves beyond single-stock time series analysis to exploit the relationships and shared information among stocks.9

#### **Relational Modeling for Strategic Forecasting**

Traditional single-stock methods like ARIMA or basic LSTMs suffer because they ignore the cross-stock correlations and the non-linear, non-stationary nature of financial data.32 HIST overcomes this by mining concept-oriented shared information, utilizing both predefined concepts (e.g., sector indices or macro factors) and dynamically derived hidden concepts (latent factor groups).9 The framework simultaneously utilizes individual stock information and this mined shared information to predict stock trends.34

For the  horizon, HIST’s relational capability is exceptionally relevant. Long-term investment performance is frequently governed by sector rotation, thematic shifts, and shared risk premia.1 The dynamic nature of HIST’s concept modeling ensures that the relationships between stocks and these thematic drivers are not assumed to be stationary, overcoming a major limitation of prior work.10 The model’s ability to discover valuable hidden concepts that measure stock commonality beyond manually defined concepts provides a stable basis for sustained, structural alpha signals over a six-month window.

The performance of HIST is directly constrained by the richness of the input features used by Qlib, such as the Alpha158 or Alpha360 factor sets.35 If the underlying feature engineering does not adequately capture the fundamental, financial, and macroeconomic dimensions necessary for long-term prediction, the dynamically modeled concepts derived by HIST will be less indicative of genuine long-term market movements. Thus, maximizing HIST's utility for

 requires meticulous feature engineering aligned with long-term investment themes.

### **4.3. IGMTF: Inter-Group Mutual Information Guided Temporal Feature Transfer**

IGMTF (Inter-Group Mutual Information Guided Temporal Feature Transfer), also released in Qlib in April 2022 2, addresses the challenges of Multivariate Time Series Forecasting (MTSF) by focusing on optimizing the internal structure and information efficiency of the features themselves.12

#### **Feature Efficiency and Correlation Structure**

The motivation behind IGMTF stems from the realization that excessive integration of information in multivariate models can introduce superfluous data, thereby curtailing potential performance gains.12 IGMTF tackles this through information-theoretic objectives:

1. **Cross-variable Decorrelation Aware feature Modeling (CDAM):** Minimizes the mutual information between the latent representation of a single sequence and its accompanying multivariate sequence input, aiming to reduce informational redundancy.12  
2. **Temporal correlation Aware Modeling (TAM):** Maximizes the mutual information between adjacent sub-sequences of the forecasted and target series, capturing temporal structure beyond single-step forecasting.12

While maximizing information efficiency is theoretically advantageous for any complex model, the benefits of IGMTF's sophisticated cross-variable decorrelation may yield diminishing returns for the  prediction horizon relative to the model's computational cost. The calculation of mutual information metrics can be computationally intense.36 For capturing persistent 6-month macro themes, the dedicated graph structure and explicit concept mining in HIST may provide a more direct and efficient route to performance improvement than IGMTF's granular focus on decorrelation between latent variables. IGMTF is an elegant solution for optimizing feature representation but may not offer the most significant marginal gains for strategic long-term alpha generation compared to the robust adaptation capabilities of DDG-DA or the deep temporal structuring of TFT.

## **V. Analysis of the Emerging "Sandwich" Model**

### **5.1. Architectural Hypotheses and Context**

The Sandwich model, released alongside the KRNN model on May 26, 2023, is one of the newest innovations within the Qlib repository.2 It is implemented using PyTorch.2 However, unlike the other models derived from published papers (AAAI, ICLR), a definitive, publicly linked paper detailing the specific financial architecture of the Qlib "Sandwich" model is not readily apparent in the referenced documentation.2

Based on modern financial deep learning trends and the use of the term "sandwich structure" in related disciplines, the model is hypothesized to be a hybrid or ensemble framework.37 Such ensemble models are increasingly employed in financial time series forecasting to capture the mixed linear and non-linear characteristics of market data.37 A typical "sandwich" construction in this context involves stacking specialized layers: for example, a Convolutional Neural Network (CNN) layer for local feature extraction, followed by an LSTM or Transformer core for modeling spatiotemporal features, and potentially wrapping this core with an Autoregressive Moving Average (ARMA) component to capture linear autocorrelation.37 This design aims to create a robust model that fuses multiple information streams, addressing different aspects of the financial time series simultaneously.

### **5.2. Projected Performance Profile and Validation Necessity**

As a relatively new model in the Qlib suite, the Sandwich architecture inherently represents an attempt to establish a new state-of-the-art benchmark, aiming to surpass older architectures like ADARNN (2021) or HIST (2022).2 If the model indeed employs a robust hybrid structure designed for complexity and robustness, its performance potential for

 forecasting is high, as it seeks to address data heterogeneity challenges and potentially incorporate novel long-sequence handling techniques (as suggested by the use of "sandwich" terminology in large model temporal processing contexts 39).

Given the mandate for optimal long-term forecasting, the Sandwich model constitutes a mandatory candidate for immediate empirical validation. Its lack of detailed published benchmark data necessitates testing using a standard Qlib qrun workflow config.2 Priority must be placed on evaluating its stability metrics—ICIR and Information Ratio—over the 6-month prediction horizon, as raw IC values alone are insufficient indicators of long-term trading utility.2

## **VI. Comparative Performance and Optimization Pathways**

For predicting stock returns six months ahead, the intrinsic architecture of the forecasting model must align with the nature of the long-term signal: high non-stationarity and strong cross-sectional dependence. The following table summarizes the comparative strengths of the Qlib models evaluated for this specific strategic task.

Table 1: Architectural Taxonomy and Suitability for Long-Horizon Forecasting ()

| Model | Core Paradigm | Long-Term Mechanism | Non-Stationarity Focus | Projected AR/ICIR Stability () | Complexity/Cost |
| :---- | :---- | :---- | :---- | :---- | :---- |
| **DDG-DA** | Meta-Learning/Adaptation | Forecasts future distribution, Resampling (KL-Divergence) 7 | Proactive Adaptation to Predictable Drift 8 | **Highest, Most Resilient** | Very High (Meta-training loop) 20 |
| **TFT** | Transformer/Attention | Multi-Head Attention, Static/Future Covariate Fusion 5 | Long-Dependency Capture (Structure) | **Highest, Most Interpretable** | High (Transformer layers, VSNs) |
| **HIST** | Graph Neural Network (GNN) | Concept-Oriented Shared Information Mining 9 | Captures Dynamic Cross-Sectional Links | High (Reliable long-term thematic exposure) | High (Graph construction/training) |
| **ADARNN** | Recurrent Neural Network (GRU Core) | Temporal Distribution Matching (TCS Mitigation) 11 | Reactive Adaptation to Regime Shifts | Moderate to High (Proven baseline) 22 | Moderate (RNN/GRU \+ Transfer Loss) |
| **IGMTF** | Information Theory/DL | Mutual Information Guided Feature Transfer (CDAM/TAM) 12 | Feature Efficiency/Redundancy Reduction | Moderate (Depends on feature quality) | Moderate to High (MI calculation) |
| **Sandwich** | Hybrid Deep Learning | Layered/Ensemble structure (Inferred) 2 | Assumed optimization of feature fusion/robustness | TBD (Latest Qlib innovation) | TBD (Likely Moderate-High) |

### **6.1. Parameterization for Long-Horizon (T+126) Tuning**

Effective application of these models requires precise configuration within the Qlib workflow (qrun using YAML files) to align the model architecture with the long-term objective.

* **TFT Configuration:** The Temporal Fusion Transformer is optimized for multi-horizon forecasting, meaning the prediction window must be explicitly defined. For a 6-month prediction, the critical parameter is output\_chunk\_length, which must be set to  (or the exact number of desired trading days).29 Concurrently, the  
  input\_chunk\_length (lookback window) should be substantial—ideally encompassing 252 days (one year) or more—to provide the self-attention mechanism with adequate data to establish robust long-term dependencies.29  
* **DDG-DA Integration:** DDG-DA operates outside the conventional model training loop; it modifies the input data distribution based on a meta-training task.42 When deploying DDG-DA, researchers must focus on tuning the meta-learner hyperparameters, specifically the learning rates (  
  lr\_da, lr\_ma, and lr of the lower level model), which often requires specialized optimization techniques such as grid search.44 This complex, nested parameterization is handled via Qlib’s framework.42  
* **Relational Models (HIST/IGMTF):** For graph and information-theoretic models, the stability of the long-term forecast depends on capturing stable cross-sectional themes. This dictates that the data handler configuration must provide a lookback window sufficient to accurately construct the dynamic concept graph (HIST) or calculate stable mutual information metrics (IGMTF). A shorter lookback window risks capturing fleeting, short-term correlations that are irrelevant or detrimental to the 6-month signal.

## **VII. Deployment Considerations and Final Recommendations**

### **7.1. Model Selection Criteria: Stability over Magnitude**

The primary objective for a strategic 6-month alpha signal is not achieving the highest theoretical return but achieving the most stable risk-adjusted return. Volatility in the alpha signal can translate directly into poor execution, high turnover, and devastating drawdowns. Therefore, the selection criteria must prioritize resilience (ICIR) over raw magnitude (IC or AR).3

The analysis confirms the synergistic combination of DDG-DA and TFT offers the highest probability of fulfilling this mandate. DDG-DA provides the foundational **stability** required by proactively correcting for Concept Drift, thereby minimizing the predictable decay of performance over the long horizon.8 TFT provides the necessary

**architectural depth** and multi-modal integration to accurately model complex, long-range dependencies spanning 126 days, a task that recurrent or conventional models struggle to execute reliably.5 The implementation of DDG-DA as a mechanism to stabilize the training environment of the TFT model is therefore considered the optimal, high-quality strategy for this task.

### **7.2. Data Requirements and Feature Engineering**

The performance of all deep learning models, particularly those targeting a long horizon, is heavily dependent on the quality and domain relevance of the input factors. While Qlib provides robust datasets like Alpha360 and Alpha158 14, successful

 prediction requires a deliberate shift in feature engineering focus.46

The emphasis must move away from high-frequency and purely technical indicators toward factors that reflect macroeconomic trends, market expectations, sentiment, and fundamental valuation.1 The Temporal Fusion Transformer specifically highlights this requirement: for TFT to function optimally, its dedicated encoders must be fed meaningful

**static covariates** (e.g., sector, industry, market classification, long-term volatility profiles) and **known future inputs**.6 These structural features anchor the long-term forecast, providing necessary context that prevents the deep temporal model from drifting based solely on noisy historical price sequence data.

### **7.3. Actionable Implementation Plan in Qlib Workflow**

The deployment should follow a methodical, multi-stage workflow utilizing Qlib’s standard component interfaces (qrun).

Table 2: Actionable Implementation Plan Summary

| Phase | Action | Primary Model Focus | Qlib Component Reference |
| :---- | :---- | :---- | :---- |
| **Data Alignment (Proactive)** | Execute the DDG-DA meta-training phase to generate future-aligned, resampled training datasets. | DDG-DA | Utilize the structure found in examples/benchmarks\_dynamic/DDG-DA/workflow.py 43 |
| **Modeling (Deep Temporal)** | Configure and train the TFT model using the distribution-aligned data, explicitly setting the prediction horizon. | TFT | Configure TFTModel class parameters with output\_chunk\_length  29 |
| **Relational Validation** | Run the HIST model as a complementary check to identify and confirm the dynamics of key long-term hidden concepts and major macro themes impacting stock groups. | HIST | Leverage the underlying dynamic graph structure for factor interpretation 9 |
| **Benchmark Baseline** | Run ADARNN (adaptive baseline) and a traditional model like LightGBM for contextual comparison of Annualized Return and ICIR.22 | ADARNN, LightGBM | Use established workflow configurations from Qlib benchmarks 2 |
| **Final Validation** | Execute the integrated workflow (DDG-DA stabilized TFT) via qrun over multiple seeds (20 runs recommended for models with randomness).14 | All | Analyze PortAnaRecord output to prioritize high Information Ratio (IR) and low Maximum Drawdown (MDD) 2 |

#### **Источники**

1. arXiv:2307.08649v1 \[q-fin.ST\] 1 Jun 2023, дата последнего обращения: октября 3, 2025, [https://arxiv.org/pdf/2307.08649](https://arxiv.org/pdf/2307.08649)  
2. microsoft/qlib: Qlib is an AI-oriented Quant investment platform that aims to use AI tech to empower Quant Research, from exploring ideas to implementing productions. Qlib supports diverse ML modeling paradigms, including supervised learning, market dynamics modeling, and RL, and is now equipped with https://github.com/microsoft \- GitHub, дата последнего обращения: октября 3, 2025, [https://github.com/microsoft/qlib](https://github.com/microsoft/qlib)  
3. Evaluation & Results Analysis — QLib 0.9.6 documentation, дата последнего обращения: октября 3, 2025, [https://qlib.readthedocs.io/en/stable/component/report.html](https://qlib.readthedocs.io/en/stable/component/report.html)  
4. 12 posts tagged with "AI" \- Vadim's blog, дата последнего обращения: октября 3, 2025, [https://vadim.blog/tags/ai](https://vadim.blog/tags/ai)  
5. Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting, дата последнего обращения: октября 3, 2025, [https://research.google/pubs/temporal-fusion-transformers-for-interpretable-multi-horizon-time-series-forecasting/](https://research.google/pubs/temporal-fusion-transformers-for-interpretable-multi-horizon-time-series-forecasting/)  
6. On the Exploration of Temporal Fusion Transformers for Anomaly Detection with Multivariate Aviation Time-Series Data \- MDPI, дата последнего обращения: октября 3, 2025, [https://www.mdpi.com/2226-4310/11/8/646](https://www.mdpi.com/2226-4310/11/8/646)  
7. DDG-DA: Data Distribution Generation for Predictable Concept Drift Adaptation | Request PDF \- ResearchGate, дата последнего обращения: октября 3, 2025, [https://www.researchgate.net/publication/361768036\_DDG-DA\_Data\_Distribution\_Generation\_for\_Predictable\_Concept\_Drift\_Adaptation](https://www.researchgate.net/publication/361768036_DDG-DA_Data_Distribution_Generation_for_Predictable_Concept_Drift_Adaptation)  
8. Adapting Stock Forecasts with AI | Vadim's blog, дата последнего обращения: октября 3, 2025, [https://vadim.blog/ddg-da-adapting-ai](https://vadim.blog/ddg-da-adapting-ai)  
9. HIST: A Graph-based Framework for Stock Trend Forecasting via Mining Concept-Oriented Shared Information \- IDEAS/RePEc, дата последнего обращения: октября 3, 2025, [https://ideas.repec.org/p/arx/papers/2110.13716.html](https://ideas.repec.org/p/arx/papers/2110.13716.html)  
10. HIST: A Graph-based Framework for Stock Trend Forecasting via Mining Concept-Oriented Shared Information \- Semantic Scholar, дата последнего обращения: октября 3, 2025, [https://www.semanticscholar.org/paper/HIST%3A-A-Graph-based-Framework-for-Stock-Trend-via-Xu-Liu/b14e764109489f3a19e6823a9c18b8a4fc74339c](https://www.semanticscholar.org/paper/HIST%3A-A-Graph-based-Framework-for-Stock-Trend-via-Xu-Liu/b14e764109489f3a19e6823a9c18b8a4fc74339c)  
11. Adaptive Deep Learning in Quant Finance with Qlib's PyTorch AdaRNN \- Vadim's blog, дата последнего обращения: октября 3, 2025, [https://vadim.blog/qlib-ai-quant-workflow-adarnn](https://vadim.blog/qlib-ai-quant-workflow-adarnn)  
12. ENHANCING MULTIVARIATE TIME SERIES FORECAST- ING WITH MUTUAL INFORMATION-DRIVEN CROSS- VARIABLE AND TEMPORAL MODELING | OpenReview, дата последнего обращения: октября 3, 2025, [https://openreview.net/forum?id=gyJpajLkX2](https://openreview.net/forum?id=gyJpajLkX2)  
13. microsoft/qlib. Continue this conversation at http://localhost:3000?gist=7a2da9a3b46893e06f2ba4681988cb85 · GitHub, дата последнего обращения: октября 3, 2025, [https://gist.github.com/m0o0scar/7a2da9a3b46893e06f2ba4681988cb85](https://gist.github.com/m0o0scar/7a2da9a3b46893e06f2ba4681988cb85)  
14. дата последнего обращения: октября 3, 2025, [https://raw.githubusercontent.com/microsoft/qlib/main/examples/benchmarks/README.md](https://raw.githubusercontent.com/microsoft/qlib/main/examples/benchmarks/README.md)  
15. DDG-DA: Data Distribution Generation for Predictable Concept Drift ..., дата последнего обращения: октября 3, 2025, [https://cdn.aaai.org/ojs/20327/20327-13-24340-1-2-20220628.pdf](https://cdn.aaai.org/ojs/20327/20327-13-24340-1-2-20220628.pdf)  
16. DDG-DA: Data Distribution Generation for Predictable Concept Drift Adaptation \- arXiv, дата последнего обращения: октября 3, 2025, [https://arxiv.org/abs/2201.04038](https://arxiv.org/abs/2201.04038)  
17. Qlib: Quantitative Platform — QLib 0.9.6 documentation, дата последнего обращения: октября 3, 2025, [https://qlib.readthedocs.io/en/stable/introduction/introduction.html](https://qlib.readthedocs.io/en/stable/introduction/introduction.html)  
18. qlib \- Codesandbox, дата последнего обращения: октября 3, 2025, [http://codesandbox.io/p/github/microsoft/qlib](http://codesandbox.io/p/github/microsoft/qlib)  
19. DDG-DA:DATA DISTRIBUTION GENERATION FOR PREDICTABLE CONCEPT DRIFT ADAPTATION \- Wendi Li's Homepage, дата последнего обращения: октября 3, 2025, [https://wendili.org/files/DDGDA\_poster.pdf](https://wendili.org/files/DDGDA_poster.pdf)  
20. Release codes of incremental learning and DoubleAdapt by MogicianXD · Pull Request \#1560 · microsoft/qlib \- GitHub, дата последнего обращения: октября 3, 2025, [https://github.com/microsoft/qlib/pull/1560/files](https://github.com/microsoft/qlib/pull/1560/files)  
21. \[2108.04443\] AdaRNN: Adaptive Learning and Forecasting of Time Series \- arXiv, дата последнего обращения: октября 3, 2025, [https://arxiv.org/abs/2108.04443](https://arxiv.org/abs/2108.04443)  
22. xxiaowo/微软的AI量化平台-qlib \- benchmarks \- Gitee, дата последнего обращения: октября 3, 2025, [https://gitee.com/xxiaowo\_admin/qlib/blob/main/examples/benchmarks/README.md](https://gitee.com/xxiaowo_admin/qlib/blob/main/examples/benchmarks/README.md)  
23. \[1912.09363\] Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting \- arXiv, дата последнего обращения: октября 3, 2025, [https://arxiv.org/abs/1912.09363](https://arxiv.org/abs/1912.09363)  
24. Temporal Fusion Transformer — darts documentation \- GitHub Pages, дата последнего обращения: октября 3, 2025, [https://unit8co.github.io/darts/examples/13-TFT-examples.html](https://unit8co.github.io/darts/examples/13-TFT-examples.html)  
25. \[UQT\] C18: Temporal Fusion Transformer (TFT) based trading strategy | by Supriya Devidutta | Medium, дата последнего обращения: октября 3, 2025, [https://medium.com/@supriyadevidutta/understanding-quantitative-trading-c16-temporal-fusion-transformer-tbased-strategy-166f98cd2dab](https://medium.com/@supriyadevidutta/understanding-quantitative-trading-c16-temporal-fusion-transformer-tbased-strategy-166f98cd2dab)  
26. mlverse/tft: R implementation of Temporal Fusion Transformers \- GitHub, дата последнего обращения: октября 3, 2025, [https://github.com/mlverse/tft](https://github.com/mlverse/tft)  
27. Temporal Fusion Transformer for Time Series Classification: A Complete Walkthrough | by Yash Gupta | Medium, дата последнего обращения: октября 3, 2025, [https://medium.com/@eryash15/temporal-fusion-transformer-for-time-series-classification-a-complete-walkthrough-5c455f488047](https://medium.com/@eryash15/temporal-fusion-transformer-for-time-series-classification-a-complete-walkthrough-5c455f488047)  
28. Covariates — darts documentation \- GitHub Pages, дата последнего обращения: октября 3, 2025, [https://unit8co.github.io/darts/userguide/covariates.html](https://unit8co.github.io/darts/userguide/covariates.html)  
29. Temporal Fusion Transformer (TFT) — darts documentation, дата последнего обращения: октября 3, 2025, [https://unit8co.github.io/darts/generated\_api/darts.models.forecasting.tft\_model.html](https://unit8co.github.io/darts/generated_api/darts.models.forecasting.tft_model.html)  
30. Interpretable multi-horizon time series forecasting of cryptocurrencies by leverage temporal fusion transformer \- PMC, дата последнего обращения: октября 3, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC11605417/](https://pmc.ncbi.nlm.nih.gov/articles/PMC11605417/)  
31. Wentao-Xu/HIST: The source code and data of the paper "HIST: A Graph-based Framework for Stock Trend Forecasting via Mining Concept-Oriented Shared Information". \- GitHub, дата последнего обращения: октября 3, 2025, [https://github.com/Wentao-Xu/HIST](https://github.com/Wentao-Xu/HIST)  
32. HIST: A Graph-based Framework for Stock Trend Forecasting via Mining Concept-Oriented Shared Information \- arXiv, дата последнего обращения: октября 3, 2025, [https://arxiv.org/pdf/2110.13716](https://arxiv.org/pdf/2110.13716)  
33. Time Series Forecasting for Stock Market Prices \- Carroll Collected, дата последнего обращения: октября 3, 2025, [https://collected.jcu.edu/cgi/viewcontent.cgi?article=1148\&context=honorspapers](https://collected.jcu.edu/cgi/viewcontent.cgi?article=1148&context=honorspapers)  
34. \[2110.13716\] HIST: A Graph-based Framework for Stock Trend Forecasting via Mining Concept-Oriented Shared Information \- arXiv, дата последнего обращения: октября 3, 2025, [https://arxiv.org/abs/2110.13716](https://arxiv.org/abs/2110.13716)  
35. Data Layer: Data Framework & Usage — QLib 0.8.3 documentation, дата последнего обращения: октября 3, 2025, [https://qlib.readthedocs.io/en/v0.8.3/component/data.html](https://qlib.readthedocs.io/en/v0.8.3/component/data.html)  
36. Information Gain and Mutual Information for Machine Learning | by Amit Yadav \- Medium, дата последнего обращения: октября 3, 2025, [https://medium.com/biased-algorithms/information-gain-and-mutual-information-for-machine-learning-060a79f32981](https://medium.com/biased-algorithms/information-gain-and-mutual-information-for-machine-learning-060a79f32981)  
37. Financial Time Series Forecasting with the Deep Learning Ensemble Model \- MDPI, дата последнего обращения: октября 3, 2025, [https://www.mdpi.com/2227-7390/11/4/1054](https://www.mdpi.com/2227-7390/11/4/1054)  
38. Deep Learning-Based Financial Time Series Forecasting via Sliding Window and Variational Mode Decomposition \- arXiv, дата последнего обращения: октября 3, 2025, [https://arxiv.org/html/2508.12565v1](https://arxiv.org/html/2508.12565v1)  
39. arXiv:2212.10356v2 \[cs.CL\] 23 May 2023, дата последнего обращения: октября 3, 2025, [https://arxiv.org/pdf/2212.10356](https://arxiv.org/pdf/2212.10356)  
40. Forecast Model: Model Training & Prediction \- Qlib Documentation \- Read the Docs, дата последнего обращения: октября 3, 2025, [https://qlib.readthedocs.io/en/latest/component/model.html](https://qlib.readthedocs.io/en/latest/component/model.html)  
41. Transformer Model — darts documentation \- GitHub Pages, дата последнего обращения: октября 3, 2025, [https://unit8co.github.io/darts/generated\_api/darts.models.forecasting.transformer\_model.html](https://unit8co.github.io/darts/generated_api/darts.models.forecasting.transformer_model.html)  
42. Release 0.8.2 Microsoft \- Qlib Documentation, дата последнего обращения: октября 3, 2025, [https://qlib.readthedocs.io/\_/downloads/en/v0.8.2/pdf/](https://qlib.readthedocs.io/_/downloads/en/v0.8.2/pdf/)  
43. DDG-DA Nested Run Error · Issue \#996 · microsoft/qlib \- GitHub, дата последнего обращения: октября 3, 2025, [https://github.com/microsoft/qlib/issues/996](https://github.com/microsoft/qlib/issues/996)  
44. The official API of DoubleAdapt (KDD'23), an incremental learning framework for online stock trend forecasting, WITHOUT dependencies on the qlib package. \- GitHub, дата последнего обращения: октября 3, 2025, [https://github.com/SJTU-DMTai/DoubleAdapt](https://github.com/SJTU-DMTai/DoubleAdapt)  
45. Hyperparameter Optimization in AutoMM \- AutoGluon 1.4.1 documentation, дата последнего обращения: октября 3, 2025, [https://auto.gluon.ai/dev/tutorials/multimodal/advanced\_topics/hyperparameter\_optimization.html](https://auto.gluon.ai/dev/tutorials/multimodal/advanced_topics/hyperparameter_optimization.html)  
46. Feature Engineering for Financial Data: What Actually Matters? \- Medium, дата последнего обращения: октября 3, 2025, [https://medium.com/@TheDataDrivenDollar/feature-engineering-for-financial-data-what-actually-matters-4b3e81cc9564](https://medium.com/@TheDataDrivenDollar/feature-engineering-for-financial-data-what-actually-matters-4b3e81cc9564)  
47. Harnessing AI for Quantitative Finance with Qlib and LightGBM \- Vadim's blog, дата последнего обращения: октября 3, 2025, [https://vadim.blog/qlib-ai-quant-workflow-lightgbm](https://vadim.blog/qlib-ai-quant-workflow-lightgbm)