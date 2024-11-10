# Brent Oil Price Analysis Project

## Overview
This project analyzes the impact of significant events on Brent oil prices, focusing on political decisions, conflicts in oil-producing regions, global economic sanctions, and OPEC policy changes. The analysis aims to provide actionable insights for investors, analysts, and policymakers in the energy sector.

## Business Context
Developed for Birhan Energies, a leading energy sector consultancy firm, this project delivers data-driven insights to help stakeholders:
- Navigate complex global energy markets
- Make informed investment decisions
- Develop effective economic and energy security policies
- Optimize operational planning and supply chain management

## Features
- Historical price trend analysis from 1987 to 2022
- Change point detection for significant market events
- Statistical modeling using Bayesian inference
- Interactive dashboard for data visualization
- Comprehensive analysis of external factors affecting oil prices


## Project Structure

```
brent-oil-changepoint-analysis/
│
├── scripts/
│   ├── oil_analysis_utils.py
|   ├── economic_analysis.py
|   ├── price_forcasting.py
│   └── README.md
├── notebooks/
│   ├── exploratory_analysis.ipynb
│   ├── economic_indicators_analysis.ipynb
│   ├── price_forecasting.ipynb
│   └── README.md
├── brent-oil-dashboard/    # directory for flask dashboard
├── tests/
├── requirements.txt    # all dependencies 
└── README.md
```

## Data Sources
- Primary Brent oil price data (1987-2022)
- Economic indicators from World Bank api

