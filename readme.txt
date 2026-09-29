1. bs.py
Calculate price, greeks and implied volatility under BS model.

2. checkpoint.py
Record and read training history.

3. generatorQMC.py
Simulate stochastic local volatilaty model using quasi Monte Carlo and Euler discretization.

4. market.py
Calculate market volatility surface.

5. my_model.py
Construct neural SDE model.

6. plot.ipynb
Plot calibration results of market volatility smile and model volatility smile.

7. polynomial.py
Generate polynomial in regression of VIX extimator.

8. smile.py
Create a class to store data for generating volatility smiles and ploting.

9. train.py
Load in parameters, model and maturity dates for training.

10. train_utils.py
Sum up loss function and calculate dynamic weight.

11. utils.py
Calculate loss by parts (SPX, VIX).

12. vix.py
Run regression to compute model VIX.

13. vol.py
Calculate drift and diffusion in stochastic local volatility model.