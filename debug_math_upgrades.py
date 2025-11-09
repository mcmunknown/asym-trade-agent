"""
Debug Script for Mathematical Upgrades
===================================

Simple debug script to test each component individually.
"""

import numpy as np
import pandas as pd
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_basic_spline():
    """Test basic spline functionality."""
    print("=== Testing Basic Spline ===")
    
    try:
        from spline_derivatives import SplineDerivativeAnalyzer
        
        # Simple test data
        x = np.linspace(0, 10, 100)
        y = np.sin(x) + 0.1 * np.random.normal(0, 1, 100)
        
        analyzer = SplineDerivativeAnalyzer(window_size=50)
        timestamps = pd.Series(x)
        prices = pd.Series(y)
        
        derivatives = analyzer.analyze_derivatives(prices, timestamps)
        
        print(f"✓ Spline derivatives calculated")
        print(f"  Price range: [{y.min():.3f}, {y.max():.3f}]")
        print(f"  Velocity range: [{derivatives['velocity'].min():.3f}, {derivatives['velocity'].max():.3f}]")
        print(f"  Fit quality: {derivatives['fit_quality']:.3f}")
        
        # Check for reasonable values
        if np.abs(derivatives['velocity']).max() > 10:
            print(f"  ⚠ Warning: High velocity values detected")
        
        return True
        
    except Exception as e:
        print(f"✗ Spline error: {e}")
        return False

def test_basic_wavelet():
    """Test basic wavelet functionality."""
    print("\n=== Testing Basic Wavelet ===")
    
    try:
        from wavelet_denoising import WaveletDenoiser
        
        # Simple test data
        x = np.linspace(0, 10, 100)
        y = np.sin(x) + 0.2 * np.random.normal(0, 1, 100)
        prices = pd.Series(y)
        
        denoiser = WaveletDenoiser(wavelet_family='db4')
        denoised = denoiser.denoise(prices)
        
        correlation = np.corrcoef(y, denoised)[0, 1]
        
        print(f"✓ Wavelet denoising completed")
        print(f"  Correlation: {correlation:.3f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Wavelet error: {e}")
        return False

def test_basic_emd():
    """Test basic EMD functionality."""
    print("\n=== Testing Basic EMD ===")
    
    try:
        from emd_denoising import EMDDenoiser
        
        # Simple test data
        x = np.linspace(0, 10, 100)
        y = np.sin(x) + 0.1 * x + 0.1 * np.random.normal(0, 1, 100)
        prices = pd.Series(y)
        
        denoiser = EMDDenoiser(max_imfs=5, emd_type='eemd')
        result = denoiser.decompose(prices, return_details=False)
        
        correlation = np.corrcoef(y, result)[0, 1]
        
        print(f"✓ EMD decomposition completed")
        print(f"  Correlation: {correlation:.3f}")
        
        return True
        
    except Exception as e:
        print(f"✗ EMD error: {e}")
        return False

def test_simple_calculus():
    """Test simple calculus analyzer without upgrades."""
    print("\n=== Testing Simple Calculus ===")
    
    try:
        from quantitative_models import CalculusPriceAnalyzer
        
        # Generate simple test data
        np.random.seed(42)
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        base_price = 100
        trend = 0.01 * np.arange(100)
        cycle = 5 * np.sin(2 * np.pi * np.arange(100) / 20)
        noise = np.random.normal(0, 2, 100)
        prices = base_price + trend + cycle + noise
        
        price_df = pd.Series(prices, index=dates)
        
        # Test original analyzer
        analyzer = CalculusPriceAnalyzer(
            lambda_param=0.6,
            use_spline_derivatives=False,
            use_wavelet_denoising=False
        )
        
        results = analyzer.analyze_price_curve(price_df)
        
        if not results.empty:
            print(f"✓ Simple calculus analysis completed")
            print(f"  Analysis points: {len(results)}")
            
            # Check key columns
            if 'velocity' in results.columns:
                vel_stats = results['velocity'].describe()
                print(f"  Velocity - Mean: {vel_stats['mean']:.3f}, Std: {vel_stats['std']:.3f}")
            
            if 'acceleration' in results.columns:
                acc_stats = results['acceleration'].describe()
                print(f"  Acceleration - Mean: {acc_stats['mean']:.3f}, Std: {acc_stats['std']:.3f}")
            
            return True
        else:
            print("✗ Simple calculus analysis failed")
            return False
        
    except Exception as e:
        print(f"✗ Simple calculus error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run debug tests."""
    print("Mathematical Upgrades Debug Script")
    print("=" * 40)
    
    tests = [
        ("Basic Spline", test_basic_spline),
        ("Basic Wavelet", test_basic_wavelet),
        ("Basic EMD", test_basic_emd),
        ("Simple Calculus", test_simple_calculus),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 40)
    print("DEBUG RESULTS")
    print("=" * 40)
    
    passed = 0
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name:<20}: {status}")
        if result:
            passed += 1
    
    print(f"\nDebug: {passed}/{len(tests)} basic tests passed")
    
    return passed == len(tests)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
