# SPX-VIX-Joint-Calibration-with-Neural-SDE
This project is based on the work by intermet / neural-spx-vix-calibration.

1. Compared to pure black-box neural networks, the Neural SDE framework preserves structural interpretability: it embeds neural networks directly into the drift and diffusion coefficients of a stochastic local volatility model, ensuring the prices and volatility dynamics follow Ito diffusion and remain Markovian.
2. Compared to simpler approaches (e.g., standard regresssions), this model guarantees internal consistency and theoretical absence of static arbitrage (such as calendar, butterfly or put-call parity arbitrage). Moreover, significant discrepancies between model valuations and market quotes can help identify relative-value mispricings for statistical arbitrage.
3. By simultaneously incorporating the risk-neutral distributions of future SPX values and future SPX implied volatilities, this model is well-suited for pricing and hedging SPX structured products.
4. The daily recalibration warm-started with the previous day's calibrated parameters can be computed overnight, offering a cost-effective setup for pricing and risk management next day.
