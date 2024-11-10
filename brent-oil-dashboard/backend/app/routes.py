from flask import Blueprint, jsonify
import pandas as pd
import numpy as np

main = Blueprint('main', __name__)

@main.route('/api/oil-prices', methods=['GET'])
def get_oil_prices():
    df = pd.read_csv('../data/Copy of BrentOilPrices.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    return jsonify({
        'dates': df['Date'].dt.strftime('%Y-%m-%d').tolist(),
        'prices': df['Price'].tolist()
    })

@main.route('/api/economic-indicators', methods=['GET'])
def get_economic_indicators():
    df = pd.read_csv('../data/final_merged_data.csv')
    return jsonify({
        'data': df.to_dict('records')
    })

@main.route('/api/correlation-analysis', methods=['GET'])
def get_correlation_analysis():
    df = pd.read_csv('../data/final_merged_data.csv')
    correlations = df.corr()['Price'].to_dict()
    return jsonify({
        'correlations': correlations
    })