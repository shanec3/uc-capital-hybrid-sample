import pandas as pd
import pytest
from main import main

def test_hybrid_forecast():
    # Placeholder: replace with actual sample file path
    sample_file = 'sample_1month.csv'
    # Simulate CLI args
    import sys
    sys.argv = ['main.py', '--in_file', sample_file, '--out_dir', 'test_results', '--model', 'sarima_xgb']
    main()
    # Check output (replace with actual output file logic)
    df = pd.read_excel('test_results/HybridForecast.xlsx')
    assert not df['HybridForecast'].isnull().any() 