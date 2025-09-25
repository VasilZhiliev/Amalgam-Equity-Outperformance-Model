# Amalgam-Equity-Outperformance-Model

This repository contains the full research code and thesis for “Classifying Stock Outperformance: A Machine Learning Framework for Dynamic Factor Allocation” (Vrije Universiteit Amsterdam, Duisenberg Honours Programme in Finance and Technology, 2025).

Overview

The Amalgam model is a machine learning framework designed to systematically navigate the Factor Zoo and generate persistent equity outperformance. It integrates 28 fundamental and market-based factor signals with granular industry classifications (based on the 74 GICS industries) to produce quarterly forecasts of stock outperformance. The model employs a rolling expanding-window design with oversampling of recent data, ensuring adaptability to evolving market regimes and style rotations.

Model: XGBoost classifier with probabilistic outputs

Data: CRSP & Compustat fundamental and market-based data (1961–2024)

Features: 29 factor proxies (Value, Quality, Profitability, Investment, Low Risk, etc.) + GICS Industry Categorical Signals

Target: Probabilistic indicator (% prob 0 to 100) of quarterly outperformance for a stock vs. both its industry peers and the value-weighted benchmark index

Key Results:

Out-of-sample period: Q!1 1981– Q4 2024 (44 years, quarterly rebalancing)

CAGR: 20.6% gross; 19.6% net of trading costs, vs. 8.9% for S&P500

Maximum Drawdown: (50.7%), vs. (56.8%) for S&P500

Annualized Alpha: 7.3% gross; 5% net of trading costs. 

Sharpe Ratio: 1.0 

Trading Costs: Average annual drag of ~0.92% based on a quarterly rebalancing frequency
