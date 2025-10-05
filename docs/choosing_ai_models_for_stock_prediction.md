# Choosing the Right AI Model for Stock Prediction: A Non-Expert's Journey

> Just Starting Out 💡

Hey everyone! Following up on my [previous post](https://dev.to/dbolotov/im-building-an-ai-to-predict-stocks-dk8) about building StocketAI, I wanted to dive deeper into how I'm picking AI models for stock prediction.

This research journey began with [Google Deep Research](https://gemini.google.com/share/b4d03200c737), which provided the initial analysis and comparison of different AI models for stock prediction. This comprehensive AI-powered research served as my starting point for understanding the landscape of available models and their capabilities.

I'm not a finance expert or a machine learning expert. I'm a solution architect who's learning as I go, relying heavily on AI tools and research to figure this out. So let me break down what I've learned in simple terms.

> **Important Disclaimer** ⚠️
>
> This analysis represents my current understanding based on the research and experimentation I've done so far. I might be wrong, and I fully expect to change my approach as I learn more, experiment with real data, and discover new techniques. Consider this a snapshot of an ongoing journey rather than definitive conclusions.

## The Big Challenge: Markets Keep Changing

Predicting stock prices 6 months ahead is really hard because **markets are constantly changing**. What worked last year might not work this year. The fancy term for this is "concept drift" - basically, the rules of the game keep changing.

Most AI models assume the future will look like the past, but that's not how stock markets work. Economic conditions change, new trends emerge, and what drove stock prices before might not matter anymore.

## My Research Journey: From Confusion to Clarity

After researching different AI models in [Qlib](https://github.com/microsoft/qlib) (a quantitative finance platform), here's what I learned:

### The Winner: A Hybrid Approach

The best approach seems to be combining two things:

1. **[DDG-DA (Data Distribution Generation for Predictable Concept Drift Adaptation)](https://arxiv.org/abs/2201.04038)** - A meta-learning technique that helps models adapt to changing market conditions by predicting future data distribution changes[^ddgda]
2. **[TFT (Temporal Fusion Transformer)](https://github.com/google-research/google-research/tree/master/tft)** - A state-of-the-art attention-based model for multi-horizon forecasting[^tft]

Think of DDG-DA as a "market change detector" and TFT as a "pattern finder." Together, they create a system that can handle the messy, ever-changing world of stock markets.

## Why This Combo Works (In Simple Terms)

### DDG-DA: The Market Change Detector

**DDG-DA** helps by:
- Predicting how market conditions might change in the future using meta-learning[^ddgda_impl]
- Adjusting the training data so the model learns from "future-like" scenarios through data distribution generation
- Basically preparing the model for surprises before they happen by proactively adapting to concept drift

It's like having a weather forecaster who not only tells you today's weather but also predicts how the climate might be changing over the next few months.

[^ddgda_impl]: Implemented in Qlib as a meta-model that works in four steps: (1) Calculate meta-information, (2) Train DDG-DA, (3) Inference to get guide information, (4) Apply guidance to forecasting models[^qlib_meta]

[^qlib_meta]: Qlib Meta Controller Documentation: https://github.com/microsoft/qlib/blob/main/docs/component/meta.rst

### TFT: The Pattern Finder

**TFT** is great because:
- It can look at long time periods (like 6 months of data) and find meaningful patterns using attention mechanisms[^tft_impl]
- It considers different types of information (like company basics, market trends, and economic indicators) through multi-modal input processing
- It doesn't just look at stock prices - it tries to understand the bigger picture using temporal fusion of different data sources

Imagine trying to predict someone's behavior not just by looking at their recent actions, but by understanding their personality, their environment, and the broader context of their life.

[^tft_impl]: Implemented in Qlib as a benchmark model with full TensorFlow implementation supporting multi-horizon forecasting and quantile regression[^qlib_tft]

[^qlib_tft]: Qlib TFT Benchmark Documentation: https://github.com/microsoft/qlib/tree/main/examples/benchmarks/TFT

## Other Models I Considered

I also looked at other options available in Qlib like:

- **[HIST (Heterogeneous Information Stock Transformer)](https://github.com/microsoft/qlib/blob/main/qlib/contrib/model/pytorch_hist.py)** - Uses concept stocks and relationship mining to find connections between different stocks and market sectors[^hist]
- **[ADARNN (Adaptive RNN)](https://github.com/microsoft/qlib/blob/main/qlib/contrib/model/pytorch_adarnn.py)** - Another model that adapts to changing conditions using transfer learning, but more reactive than proactive[^adarnn]
- **[Sandwich](https://github.com/microsoft/qlib/blob/main/qlib/contrib/model/pytorch_sandwich.py)** - A CNN-KRNN architecture designed for stock prediction[^sandwich]

The DDG-DA + TFT combo is the most reliable for long-term predictions.

[^hist]: HIST model in Qlib uses concept stocks to capture market sector relationships and improve prediction accuracy
[^adarnn]: ADARNN model in Qlib uses domain adaptation techniques to handle changing market conditions reactively
[^sandwich]: Sandwich model in Qlib combines CNN and KRNN (Kernel Recurrent Neural Network) for spatiotemporal feature extraction

## What This Means for StocketAI

For my VN30 stock prediction project, this means:

1. **I'll use the hybrid approach** - DDG-DA to handle market changes + TFT for the actual predictions
2. **Focus on 6-month predictions** - This approach works best for longer time horizons
3. **Keep it practical** - I want models that work well in the real world, not just in theory

## Building Confidence Through Multiple Models

What makes this even better is the idea of training multiple models and using meta-analysis to validate predictions. Instead of relying on just one model, we can train several different approaches and compare their results. The real confidence comes when multiple models - whether it's DDG-DA combined with TFT, or other approaches like HIST and ADARNN - all point to similar predictions. Only when we see that level of agreement across different modeling techniques do we really trust the forecast. This approach helps filter out the noise and gives us more reliable insights for making investment decisions.

## Next Steps in My Journey

I'm currently setting up experiments to test this hybrid approach with real VN30 data. I'll be using AI tools to help me configure everything properly and understand what the results mean.

The goal is still the same: help regular people like me make better investment decisions without needing a finance degree or years of trading experience.

## Come Join the Fun! 🎉

What do you think? Have you tried predicting stock prices with AI? What's been your experience? I'd love to hear from other non-experts who are figuring this out as they go!

Check out the StocketAI project on [GitHub](https://github.com/WitcherD/StocketAI) if you want to follow along with my experiments.

---

## References

[^ddgda]: Original DDG-DA paper: "DDG-DA: Data Distribution Generation for Predictable Concept Drift Adaptation" (https://arxiv.org/abs/2201.04038)
[^tft]: Original TFT paper: "Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting" (https://arxiv.org/abs/1912.09363)

*Built with ❤️ using AI assistance and some of coffee ☕*
