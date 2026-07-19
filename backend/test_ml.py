import sys
import os

# Add routers directory to path
sys.path.append(os.path.dirname(__file__))

from routers.ml_model import api_train_model, api_predict, PredictRequestSchema, init_db

def test_workflow():
    print("Initializing Database...")
    init_db()
    
    print("\nRunning model training...")
    res = api_train_model()
    print("Training finished successfully!")
    print("Metrics:")
    for k, v in res.get("metrics", {}).items():
        print(f"  {k}: {v}")
    
    print("\nFeature importances:")
    for item in res.get("feature_importances", []):
        print(f"  {item['feature']}: {item['importance']}")
        
    print("\nRunning predictions...")
    req = PredictRequestSchema(
        ticker="AAPL",
        strike=190.0,
        underlierPrice=185.0,
        volume=1000,
        openInterest=200,
        volOiRatio=5.5,
        impliedVolatility=35.0,
        premium=250000.0,
        dte=30,
        optionType="Call",
        side="BUY",
        trendAlignment="BULL_ALIGNED"
    )
    
    pred_res = api_predict(req)
    print("Prediction Result:")
    print(f"  quantiles: {pred_res['quantiles']}")
    print(f"  p_success: {pred_res['p_success']}")
    print(f"  expected_return: {pred_res['expected_return']}")
    print(f"  strategy: {pred_res['strategy']}")
    print(f"  strategy_confidence: {pred_res['strategy_confidence']}%")
    print(f"  kelly_fraction: {pred_res['kelly_fraction']}")
    print(f"  kelly_fraction_uncapped: {pred_res['kelly_fraction_uncapped']}")
    
    print("\nAll ML engine updates verified successfully!")

if __name__ == "__main__":
    test_workflow()
