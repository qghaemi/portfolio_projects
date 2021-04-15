# Capstone: Forecasting Bitcoin Price

![](https://i.guim.co.uk/img/media/6e51e9403e2f5f99687ef268535f660d557d53b1/0_286_5184_3110/master/5184.jpg?width=445&quality=45&auto=format&fit=max&dpr=2&s=eb826e034ded452e55c31a85d1ed2cf0)

Author: Q Ghaemi

## Executive Summary

When the internet was created and released in 1983, no one imagined what it would become. The developers of the internet did their best to build out as any scenarios as possible, including errors to help guide users in this new world. One such error that is rarely seen is the 402 error: payment error. When the internet was designed, the developers believed a payment protocol would be built on top of the internet. 26 years and a few financial crises later, enter Bitcoin. The digital currency built for the internet.

The data was collected from Yahoo Finance and came cleaned so there was minimal cleaning that needed to be done.

The components of EDA include a time plot, seasonal decomposition plot, stationarity check, and ACF/PACF charts. These various plots will help to determine which models are appropriate to forecast Bitcoin price. Based on findings in EDA: the data has an exponential trend, there is seasonality in the data at the monthly level, the data is not stationary but a one-period differencing will allow the data to be stationary, and the data is highly correlated with past values. Further details can be found below or in the EDA notebook.

The findings during EDA helped better inform the model selection process: ARIMA models, exponential smoothing models, seasonal extensions of both model types, and a neural network. The metrics being used to evaluate the models are their AIC scores as Residual Sum of Squares (RSS). The AIC (Akaie Information Criterion) measures how well the model measures reality; this is explaining how well the model works on the training data. The RSS harshly punishes incorrect forecasts which is why it was the chosen error term; this is explaining how well the model works on the testing data.

Each modeling notebook followed a similar structure: run a baseline model, the GridSearch over the hyperparameters in order to optimize for AIC and RSS. Each GridSearch was ran twice in order to optimize for the two evaluation metrics separately.

More details can be found below or in individual notebooks regarding the modeling process, but the too long did not read version is that the GRU neural network was able to successfully forecast the price of Bitcoin within a reasonable error level. Below is a graph that shows the forecasted price of Bitcoin, the error ranges (shaded area), and the true price of Bitcoin. 

![GRU](https://github.com/qghaemi/portfolio_projects/blob/main/bitcoin_forecasting_project/images/gru_fcast.png)

This is an extremely important finding as we can confirm the price of Bitcoin can be forecasted within a reasonable error level by only using the closing price. Extensions to this model will need to be developed in order to forecast for future dates, but in the mean time this model proves that a model should be able to reasonably forecast the price of Bitcoin.

The main recommendation is to build out the GRU network to forecast for the future, however there are some issues with the classes that were used during this process that would need to be reworked: `GridSearchCV`, `KerasRegressor`, and `TimeseriesGenerator`. A workaround to this problem would be to build a custom class to replace the `KerasRegressor`. The custom class should perform in the same way as the `KerasRegressor` and `TimeseriesGenerator` all in one class. Despite this custom class seeming very simple, this is an issue that multiple people are [attempting to solve](https://stackoverflow.com/questions/59118239/is-there-anyway-to-use-fit-generator-method-with-kerasregressor-wrapper). More details on this can be found in the recurrent neural network notebook.

### Contents: 

- [File Structure]()
- [Background]()
- [Data Collection & Cleaning]()
- [Data Dictionary]()
- [EDA]()
- [Modeling]()
- [Conclusions and Recommendations]()

## File Structure

- **data** folder - all data points and a cleaned, consolidated data file
- **images** folder - all images found throughout the project are saved in this folder
- **1_data_cleaning.ipynb** - data cleaning notebook
- **2_eda.ipynb** - Exploratory Data Analysis
- **3_arima_models.ipynb** - ARIMA and SARIMA modeling process
- **4_exponential_smoothing_models.ipynb** - Double and Triple Exponential Smoothing models
- **5_recurrent_neural_network.ipynb** - RNN models
- **presentation.pdf** - RNN models

Package requirements:
- pandas
- numpy
- seaborn
- matplotlib
- sklearn
- statsmodels
- tensorflow

## Background

The internet was created in 1983 and the original designers built out many functions that we still use to this day. An example of this are the 400 errors. Some of the common 400-errors include a 404 error (page not found) or a 403-error (forbidden page), but a lesser known 400-error is the 402-error: payment error. This means that when the internet was designed and built, the developers believed a payment protocol would be built on top of the internet. What that payment protocol would be, how it would operate, or any other details do not matter: the key is that the concept of a "money of the internet" was thought of very early on. Fast forward to 2009 and the creation of Bitcoin.

Bitcoin was created by a person or group named Satoshi Nakamoto (real identity still unknown) as a version of electronic cash that would allow online payments to be sent directly from one party to another without going through a financial institution; it was created as a direct response to the financial crisis in 2008 [read the full Bitcoin whitepaper here](https://bitcoin.org/bitcoin.pdf). The fact that Bitcoin does not require a financial institution created the world's first decentralized asset (and the world's first cryptocurrency - crypto for short). Decentralized asset means no one individual or state controls the asset, they cannot create more, freeze, or have any real impact on the asset. 

Nakamoto created a new, scarce, digital good. Bitcoins are created on the Bitcoin network in a process known as “mining”. Bitcoin mining can be thought of as gold mining (for simplicity sake we will not discuss the complex code behind the mining process, just understand that miners solve a complex math equation and are rewarded with Bitcoin). Satoshi designed the system to only have 21 million Bitcoins. This created a scarcity in an asset unlike gold (the maximum amount of gold on Earth is unknown) or the dollar (the Federal Reserve can simply print more money making the dollar not scarce). Vijay Boyapati has an excellent (but long) article detailing Bitcoin in greater detail and is worth reviewing for those who are curious to [learn more](https://vijayboyapati.medium.com/the-bullish-case-for-bitcoin-6ecc8bdecc1). Boyapati breaks down what is real money, how it has transformed throughout the history of civilized humans, and why Bitcoin is the superior asset.

While Bitcoin's value is highly contested and an argument can be made for both sides: Bitcoin has no value and Bitcoin is a valuable asset. This project will not debate this argument and will simply take the position of the latter and assumes anyone reading from this point on agrees. The key to understanding the growth rate of Bitcoin has been outlined by Michael Casey where he explains adoption theory with an S-curve (image below).

![adoption_s_curve](https://github.com/qghaemi/portfolio_projects/blob/main/bitcoin_forecasting_project/images/adoption_curve.png)

This S-curve that Casey shows is the adoption curve for new technology to be accepted by the masses. The curve is meant to be an exponential growth and then as adoption nears 100%, the curve begins to flatten out until 100% or near 100% adoption is reached. While many of the technologies shown have a fairly smooth trajectory up, unlike Bitcoin which has proven to be very volatile. Casey mentions this and points out that while it may look very volatile in the present, if/when Bitcoin reaches near 100% adoption the curve may resemble many of these other S-curves.

As Bitcoin has continued its rise since late-2020, more traditional financial institutions are incorporating Bitcoin: [Square](https://www.cnbc.com/2021/02/23/square-buys-170-million-worth-of-bitcoin.html#:~:text=Square%20bought%20%24170%20million%20worth%20of%20bitcoin%2C%20the%20company%20revealed,of%20the%20end%20of%202020.) and [Tesla](https://www.cnbc.com/2021/02/08/tesla-buys-1point5-billion-in-bitcoin.html) are two examples of publicly traded companies that have purchased Bitcoin with their cash reserves (and Tesla is even allowing customers to buy a new car with Bitcoin); Coinbase (a cryptocurrency trading platform) because [publicly listed on the NASDAQ on April 14, 2021](https://www.marketwatch.com/story/coinbase-ipo-everything-you-need-to-know-about-the-watershed-moment-in-crypto-11618350086); and now large financial institutions (like [Goldman Sachs](https://www.cnbc.com/2021/03/31/bitcoin-goldman-is-close-to-offering-bitcoin-to-its-richest-clients.html)) are exploring or already offering clients the opportunity to purchase Bitcoin. With traditional finance attempting to dive into the crypto world, some key differences need to be highlighted between traditional investment assets (cash, stocks, gold, bonds, real estate, etc) and cryptocurrencies. Going forward, the focus will be strictly on Bitcoin.

The largest and most important difference between these traditional assets and Bitcoin is the fact that Bitcoin is decentralized. This is significant for multiple reasons: no central bank can announce more Bitcoin will be created which would devalue the Bitcoin already in circulation (unlike the Federal Reserve and money printing they do), no central authority can shutdown trading of Bitcoin (unlike the GameStop fiasco from the beginning of 2021), payments are borderless meaning international transactions can happen quickly and cheaply, for nations with a destabilized fiat (government issued) currency Bitcoin could be an alternative, and decentralized currencies are immune to inflation of deflation (when the supply is fixed, like with Bitcoin). These points are the key differentiators (and benefits) of Bitcoin when compared to the cash or bonds. 

Unlike the stock market, crypto markets are operating 24/7. This means that the price Bitcoin is when you fall asleep is not the price it will be when you wake up. This can be nerve wracking for some but is a massive benefit in the long run. Unlike stocks or real estate, there is opportunity to collect an income from this asset (such as dividends or rent respectively). Bitcoin also does not have an earnings to report the way a stock would - it is similar to gold in this aspect. While gold has many uses (such as jewelry or in your iPhone), Bitcoin has no such uses, yet!

## Problem Statement

As Bitcoin's adoption rate continues to grow, it is important for new and speculative investors to be able to gauge where the current price of Bitcoin is on the adoption curve. 

Can a time series model be developed that will forecast the price of Bitcoin within a reasonable error level (error should not be more than 20% of current Bitcoin price)?  

*As of writing this, the price of Bitcoin is about $63,000: this means error (+/-) should not be more than $6,300.*

## Data Collection & Cleaning

All of the data that was used can be found at [finance.yahoo.com](https://finance.yahoo.com/quote/BTC-USD/history?period1=1410912000&period2=1617321600&interval=1d&filter=history&frequency=1d&includeAdjustedClose=true). The data was collected on April 2, 2021 which is why the last day that is shown in the data is April 1, 2021.

Cleaning the data was very straightforward as it came clean from Yahoo Finance. The main changes were reformatting the date column to DateTime and setting it as the Index. That and changing the columns from camel case to snake case were the only things done to clean the data.

## Data Dictionary

| Column Name | Description | Data type |
|--|--|--|
| date | Date | DateTimeIndex |
| high | High Price | float64 |
| low | Low Price | float64 |
| open | Opening Price | float64 |
| close | Closing Price | float64 |
| volume | Trading Volume | float64 |

## EDA

Based on findings in EDA: the data has an exponential trend, there is seasonality in the data at the monthly level, the data is not stationary but a one-period differencing will allow the data to be stationary, and the data is highly correlated with past values.

![monthly_seasonal_decomp](https://github.com/qghaemi/portfolio_projects/blob/main/bitcoin_forecasting_project/images/monthly_seasonal_decomp.png)

The seasonal decomposition chart above is made up of four sections from top to bottom: time plot, trend analysis, seasonality analysis, and residuals. Above is a seasonal decomposition of Bitcoin's closing price every 30-days. The trend in this example was an exponential trend based on the residuals; the residuals were consistently choppy throughout the entire time period when the exponential trend was analyzed, but when a linear trend was analyzed the residuals were much more pronounced from 2018-2021. While that could have suggested a linear trend until 2018 and then a switch to an exponential trend, there was no change in the underlying technology in 2018 - this was not the case. The argument could be made that Bitcoin gained mainstream notoriety in 2018, but that was the result of the price hitting new highs, not the other way around. 

The seasonal section of the seasonal decomposition chart was the most surprising finding: seasonality could be seen in the Bitcoin closing price at a monthly level. The seasonality became more pronounced as the time period expanded. This was an unexpected finding but helped craft a key point in model selection which was to include the seasonal extensions of the models selected.

![stationarity_visualization](https://github.com/qghaemi/portfolio_projects/blob/main/bitcoin_forecasting_project/images/stationarity_visualization.png)

In the above image, the average Bitcoin price over the time periods specified show the data is not stationary. An Augmented Dickey-Fuller (ADF) Test confirmed what the visualization showed, the data is not stationary. Differencing was performed in order to achieve stationarity, because the data has an exponential trend, both normal and logarithmic differencing was applied. Both types of differencing achieved stationarity, but the logged differenced data had a stronger ADF score. This determination gave the paved the way for using the logged version of the data for the ARIMA model.

![ACF](https://github.com/qghaemi/portfolio_projects/blob/main/bitcoin_forecasting_project/images/monthly_acf.png)

The key takeaways from the above monthly ACF chart are that the data is highly correlated even 90 days out and there is seasonality based on the scalloped/wave-like pattern in the chart. This combined with findings in the PACF chart (not pictured but full details in EDA notebook) helped inform the final parameters for the ARIMA model.

## Modeling

The findings during EDA helped better inform the model selection process: ARIMA models based on the high autocorrelation and partial-autocorrelation, exponential smoothing models due to the exponential trend, and the seasonal extensions of both model types due to seasonality. Additionally, exploring a neural network to forecast the bitcoin price will be explored to see if allowing the machine to train itself on the data could result in a better forecast compared to the machine learning models previously mentioned.

The metrics being used to evaluate the models are their AIC scores as Residual Sum of Squares (RSS). The AIC (Akaie Information Criterion) measures how well the model measures reality. This is explaining how well the model works on the training data (for those more familiar, this can be thought of as our training score). The RSS is the chosen error term as it harshly punishes incorrect forecasts, and does so in a more extreme way than other error metrics (such as MSE, MAE or RMSE). This is explaining how well the model works on the testing data (for those more familiar, this can be thought of as our testing score). 

The two types of ARIMA models that were ran were the ARIMA and SARIMA models. ARIMA stands for AutoRegressive, Integrated, Moving Average; SARIMA is the seasonal extension of the ARIMA model. More details about their hyperparameters and how they were selected can be found in the ARIMA notebook. 

![SARIMA](https://github.com/qghaemi/portfolio_projects/blob/main/bitcoin_forecasting_project/images/sarima_chart.png)

The RSS optimized SARIMA model was the best performing model from this section: it was the only model built that not only forecasted the price of Bitcoin higher, but also forecasted a series of price declines (pullbacks) that was similar to the real price of Bitcoin. The graph above helps visualize the SARIMA models' forecasts.

The next model type that was explored were exponential smoothing models due to the data having an exponential trend. The two exponential models that were run were Double Exponential Smoothing (Holt) and Triple Exponential Smoothing (Holt Winter). The Holt model is best used for univariate data (one variable) that has a trend. The Holt Winter model is the seasonal extension of the Holt model: it allows for the seasonality to be measured with a linear or exponential trend. 

One hyperparameter to highlight in these model types is the damped parameter. When set to True, the forecast will reduce the size of the trend over future time steps down to a straight line (no trend). This aligns well with the adoption curve that was shown in the background section. All of the RSS optimized exponential models had damped set to True which only further validated what was seen. More details about their hyperparameters and how they were selected can be found in the exponential smoothing notebook. 

![Holt](https://github.com/qghaemi/portfolio_projects/blob/main/bitcoin_forecasting_project/images/holt_chart.png)

Of the two exponential models tested, the Holt model returned a lower RSS score which was surprising since it did not account for seasonality in the data. Based on this finding, there is an interesting conclusion that can be drawn: the seasonality found in Bitcoin is specific to pullbacks only. The way this conclusion is drawn is based on how the exponential models forecast compared to how the SARIMA model forecasts. The exponential models are forecasting an exponential trend and are not concerned with minor pullbacks when the long-term trend is higher. As a result, the exponential model that incorporates seasonality is not helpful because the seasonality is reflective of short-term declines/pullbacks.

The final model that was created was a recurrent neural network (RNN). RNN's are best used for sequential data like time series or natural language processing (NLP). A neural network allows the machine the ability to train itself based on the data it is given. The two types of RNN's that were explored are gated recurrent network (GRU) and Long Short Term Memory (LSTM). These are two common network architectures that are helpful in solving time-series problems. The main difference between GRU and LSTM is that GRU has two gates that the data will go through while LSTM has three. Which architecture is best for the model depends on the data that is being modeled so both were created as an initial test before building the better one out deeper. More details about their differences and how each network was built can be found in the recurrent neural network notebook.

![RNN](https://github.com/qghaemi/portfolio_projects/blob/main/bitcoin_forecasting_project/images/gru_chart.png)

The above graph shows the forecasted price against the actual price similar to the previous graphs shown. While the forecast window is smaller (more details on why are in the RNN notebook), the forecasted prices are much more in line with the real price which is why this model returned the lowest RSS score. To bring it back to the problem statement, the RMSE brings the error back into the same units as the forecast (USD) and is 2,261.62. This is well below the $6,300 level that was deemed reasonable error which means this model is able to successfully forecast the price of Bitcoin within a reasonable error level.

## Conclusions & Recommendations

Below is a chart that shows the AIC and RSS for each model type created and what the model was optimized for. From the below chart the GRU neural network was the strongest 

| Model | Optimized | AIC | RSS |
|--|--|--|--|
| ARIMA | AIC | -9,803.978875555455 | 195,663,055,382.6728 |
| ARIMA | RSS | -9,763.586525933988 | 18,940,753,598.031494 |
| SARIMA | AIC | -9,775.04078021729 | 24,573,606,014.329235 |
| SARIMA | RSS | -8,821.249991627563 | 7,395,120,381.605706 |
| Holt | AIC | 31,511.65372812618 | 17,592,924,408.176796 |
| Holt | RSS | 32,622.519073158164 | 1,649,003,709.904847 |
| Holt Winter | AIC | 31,504.365238411654 | 20,400,967,764.329018 |
| Holt Winter | RSS | 31,594.712991325876 | 1,777,842,238.3718455 |
| GRU Baseline | RSS | N/A | 527,310,145.44964737* |
| LSTM Baseline | RSS | N/A | 9,479,719,403,159.566* |
| Optimized GRU | RSS | N/A | 306,895,354.31196207* |

*The RSS for GRU/LSTM neural networks are only calculated for a 60-day forecast while all other model's RSS is based on a 90-day forecast

After building out close to 8,000 different models, the GRU neural network was able to forecast the price of Bitcoin within a reasonable level. While not perfect, the GRU neural network had the smallest RSS error which also meant it had the smallest RMSE. The RMSE brings the error back to the same units as the forecast and was used to create a, error interval for the forecast. This error interval covered a total range of roughly 10% from the forecast, below the 20% threshold that was a part of the problem statement.

The main recommendation is to reformat the GRU neural network to forecast multiple steps ahead. The forecast can either be single-shot (makes the predictions all at once) or autoregressive (make one prediction at a time and feed the output back to the model). While autoregressive seems the best on paper, exploring both options may be beneficial to see if one or the other is better at forecasting the price of Bitcoin on quarter out.

Another recommendation would be to build a custom class to replace the `KerasRegressor`. The custom class should allow for the GRU GridSearch to be completed. While extremely helpful and it could be used for problems beyond just forecasting the price of Bitcoin, the GridSearch was focused on optimizing the above GRU network type. As previously discussed, this network is not providing forecasts multiple steps ahead which should be the priority. Once that network is built, a GridSearch over its hyperparameters should result in an optimized model.