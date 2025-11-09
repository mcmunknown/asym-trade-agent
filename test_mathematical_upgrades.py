"""
Test Script for Mathematical Upgrades
================================

This script tests the new spline derivatives, wavelet denoising, 
and enhanced Taylor forecasting to ensure they work correctly.

Run: python test_mathematical_upgrades.py
"""

import os
os.environ.setdefault("NUMPY_SKIP_MAC_OS_CHECK", "1")

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from quantitative_models import CalculusPriceAnalyzer
from spline_derivatives import SplineDerivativeAnalyzer
from wavelet_denoising import WaveletDenoiser
from emd_denoising import EMDDenoiser
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@pytest.fixture(scope="module")
def prices_series():
    """
    Shared price series for pytest-based validation.
    """
    data = generate_test_data(days=45, trend=True, noise_level=0.015, seed=123)
    return data['price']

def generate_test_data(days=100, trend=True, noise_level=0.02, seed: int = 42):
    """
    Generate synthetic price data with known characteristics for testing.
    
    Args:
        days: Number of days of data
        trend: Whether to include a trend component
        noise_level: Standard deviation of noise
        seed: Random seed for reproducibility
        
    Returns:
        DataFrame with prices and components
    """
    rng = np.random.default_rng(seed)
    
    # Generate time series (daily data)
    dates = pd.date_range(start='2023-01-01', periods=days*24, freq='h')
    t = np.arange(len(dates))
    
    # Base price
    base_price = 50000
    
    # Trend component (logarithmic growth)
    if trend:
        trend_component = 0.0001 * t  # Small upward trend
    else:
        trend_component = 0.0
    
    # Oscillatory component (market cycles)
    cycle_component = 0.002 * np.sin(2 * np.pi * t / (24 * 7))  # Weekly cycle
    cycle_component += 0.001 * np.sin(2 * np.pi * t / (24))  # Daily cycle
    
    # Noise component
    noise = rng.normal(0, noise_level, len(t))
    
    # Combine components
    log_returns = trend_component + cycle_component + noise
    log_prices = np.log(base_price) + np.cumsum(log_returns)
    prices = np.exp(log_prices)
    
    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': dates,
        'price': prices,
        'trend': base_price * np.exp(np.cumsum(trend_component)),
        'cycle': cycle_component,
        'noise': noise
    })
    
    df.set_index('timestamp', inplace=True)
    
    logger.info(f"Generated {len(df)} price points: "
               f"trend={'included' if trend else 'none'}, "
               f"noise_level={noise_level}")
    
    return df

def run_spline_derivatives_check(prices):
    """
    Test spline derivative calculation against known synthetic data.
    """
    logger.info("\n=== Testing Spline Derivatives ===")
    
    try:
        analyzer = SplineDerivativeAnalyzer(window_size=50, adaptive_smoothing=True)
        derivatives = analyzer.analyze_derivatives(prices)
        
        if derivatives:
            velocity = derivatives['velocity']
            acceleration = derivatives['acceleration']
            
            # Basic validation
            logger.info(f"✓ Spline derivatives calculated successfully")
            logger.info(f"  Velocity range: [{np.min(velocity):.6f}, {np.max(velocity):.6f}]")
            logger.info(f"  Acceleration range: [{np.min(acceleration):.6f}, {np.max(acceleration):.6f}]")
            logger.info(f"  Fit quality: {derivatives['fit_quality']:.3f}")
            
            # Validate against finite differences (should be smoother)
            finite_diff_velocity = np.diff(prices.values)
            spline_smoothness = np.std(np.diff(velocity[-50:]))  # Higher-order smoothness
            finite_smoothness = np.std(np.diff(finite_diff_velocity[-50:]))
            
            logger.info(f"  Spline smoothness: {spline_smoothness:.6f}")
            logger.info(f"  Finite diff smoothness: {finite_smoothness:.6f}")
            logger.info(f"  Smoothness improvement: {(finite_smoothness - spline_smoothness)/finite_smoothness:.1%}")
            
            return True, derivatives
        else:
            logger.error("✗ Spline derivatives failed")
            return False, None
            
    except Exception as e:
        logger.error(f"✗ Spline derivatives error: {e}")
        return False, None

def run_wavelet_denoising_check(prices):
    """
    Test wavelet denoising and signal preservation.
    """
    logger.info("\n=== Testing Wavelet Denoising ===")
    
    try:
        denoiser = WaveletDenoiser(wavelet_family='db4', threshold_method='sure')
        denoised_prices = denoiser.denoise(prices, return_details=False)
        
        # Calculate signal quality metrics
        original_std = np.std(prices)
        denoised_std = np.std(denoised_prices)
        correlation = np.corrcoef(prices, denoised_prices)[0, 1]
        
        logger.info(f"✓ Wavelet denoising completed successfully")
        logger.info(f"  Original std: {original_std:.2f}")
        logger.info(f"  Denoised std: {denoised_std:.2f}")
        logger.info(f"  Correlation preserved: {correlation:.3f}")
        logger.info(f"  Noise reduction: {(original_std - denoised_std)/original_std:.1%}")
        
        # Get denoising statistics
        stats = denoiser.get_denoising_statistics()
        logger.info(f"  Average SNR improvement: {stats['avg_snr']:.3f}")
        logger.info(f"  Quality stability: {stats['quality_stability']:.3f}")
        
        return True, denoised_prices
        
    except Exception as e:
        logger.error(f"✗ Wavelet denoising error: {e}")
        return False, None

def run_emd_denoising_check(prices):
    """
    Test EMD decomposition and signal reconstruction.
    """
    logger.info("\n=== Testing EMD Denoising ===")
    
    try:
        denoiser = EMDDenoiser(max_imfs=8, emd_type='eemd')
        result = denoiser.decompose(prices, return_details=True)
        
        if isinstance(result, dict):
            denoised_prices = result['denoised']
            signal_imfs = result['signal_imfs']
            noise_imfs = result['noise_imfs']
            quality_metrics = result['quality_metrics']
            
            logger.info(f"✓ EMD decomposition completed successfully")
            logger.info(f"  Total IMFs: {len(signal_imfs) + len(noise_imfs)}")
            logger.info(f"  Signal IMFs: {len(signal_imfs)}")
            logger.info(f"  Noise IMFs: {len(noise_imfs)}")
            logger.info(f"  Signal correlation: {quality_metrics['correlation_preservation']:.3f}")
            logger.info(f"  SNR improvement: {quality_metrics['signal_to_noise']:.3f}")
            
            return True, denoised_prices
        else:
            logger.error("✗ EMD decomposition failed")
            return False, None
            
    except Exception as e:
        logger.error(f"✗ EMD denoising error: {e}")
        return False, None

def run_enhanced_calculus_analysis_check(prices):
    """
    Test the enhanced calculus analyzer with all upgrades enabled.
    """
    logger.info("\n=== Testing Enhanced Calculus Analysis ===")
    
    try:
        # Enhanced analyzer with all upgrades
        analyzer = CalculusPriceAnalyzer(
            lambda_param=0.6,
            snr_threshold=1.0,
            use_spline_derivatives=True,
            use_wavelet_denoising=True,
            spline_window=50,
            wavelet_type='db4'
        )
        
        # Run complete analysis
        results = analyzer.analyze_price_curve(prices)
        
        if not results.empty:
            logger.info(f"✓ Enhanced calculus analysis completed")
            logger.info(f"  Analysis points: {len(results)}")
            
            # Check key columns exist
            required_cols = ['smoothed_price', 'velocity', 'acceleration', 'enhanced_forecast']
            missing_cols = [col for col in required_cols if col not in results.columns]
            
            if missing_cols:
                logger.warning(f"  Missing columns: {missing_cols}")
            else:
                # Sample statistics
                velocity_stats = results['velocity'].describe()
                accel_stats = results['acceleration'].describe()
                forecast_stats = results['enhanced_forecast'].describe()
                
                logger.info(f"  Velocity - Mean: {velocity_stats['mean']:.6f}, "
                           f"Std: {velocity_stats['std']:.6f}")
                logger.info(f"  Acceleration - Mean: {accel_stats['mean']:.6f}, "
                           f"Std: {accel_stats['std']:.6f}")
                logger.info(f"  Forecast - Mean: {forecast_stats['mean']:.2f}, "
                           f"Std: {forecast_stats['std']:.2f}")
                
                # Count valid signals
                valid_signals = results['valid_signal'].sum()
                total_signals = len(results)
                signal_quality = valid_signals / total_signals if total_signals > 0 else 0
                
                logger.info(f"  Valid signals: {valid_signals}/{total_signals} ({signal_quality:.1%})")
            
            return True, results
        else:
            logger.error("✗ Enhanced calculus analysis failed")
            return False, None
            
    except Exception as e:
        logger.error(f"✗ Enhanced calculus analysis error: {e}")
        return False, None

def run_performance_comparison(prices):
    """
    Compare performance between original and enhanced methods.
    """
    logger.info("\n=== Performance Comparison ===")
    
    try:
        # Original analyzer (finite differences only)
        original_analyzer = CalculusPriceAnalyzer(
            lambda_param=0.6,
            use_spline_derivatives=False,
            use_wavelet_denoising=False
        )
        
        # Enhanced analyzer
        enhanced_analyzer = CalculusPriceAnalyzer(
            lambda_param=0.6,
            use_spline_derivatives=True,
            use_wavelet_denoising=True
        )
        
        import time
        
        # Time original method
        start_time = time.time()
        original_results = original_analyzer.analyze_price_curve(prices)
        original_time = time.time() - start_time
        
        # Time enhanced method
        start_time = time.time()
        enhanced_results = enhanced_analyzer.analyze_price_curve(prices)
        enhanced_time = time.time() - start_time
        
        if not original_results.empty and not enhanced_results.empty:
            # Compare noise levels
            original_velocity_std = original_results['velocity'].std()
            enhanced_velocity_std = enhanced_results['velocity'].std()
            
            original_accel_std = original_results['acceleration'].std()
            enhanced_accel_std = enhanced_results['acceleration'].std()
            
            # Compare signal quality
            original_valid_rate = original_results['valid_signal'].mean()
            enhanced_valid_rate = enhanced_results['valid_signal'].mean()
            
            logger.info(f"✓ Performance comparison completed")
            logger.info(f"  Original method time: {original_time:.3f}s")
            logger.info(f"  Enhanced method time: {enhanced_time:.3f}s")
            logger.info(f"  Time overhead: {(enhanced_time - original_time)/original_time:.1%}")
            logger.info(f"")
            logger.info(f"  Velocity noise - Original: {original_velocity_std:.6f}, "
                       f"Enhanced: {enhanced_velocity_std:.6f}")
            logger.info(f"  Velocity noise reduction: "
                       f"{(original_velocity_std - enhanced_velocity_std)/original_velocity_std:.1%}")
            logger.info(f"")
            logger.info(f"  Acceleration noise - Original: {original_accel_std:.6f}, "
                       f"Enhanced: {enhanced_accel_std:.6f}")
            logger.info(f"  Acceleration noise reduction: "
                       f"{(original_accel_std - enhanced_accel_std)/original_accel_std:.1%}")
            logger.info(f"")
            logger.info(f"  Valid signal rate - Original: {original_valid_rate:.1%}, "
                       f"Enhanced: {enhanced_valid_rate:.1%}")
            logger.info(f"  Signal quality improvement: "
                       f"{(enhanced_valid_rate - original_valid_rate)/original_valid_rate:.1%}")
            
            return True
        else:
            logger.error("✗ Performance comparison failed")
            return False
            
    except Exception as e:
        logger.error(f"✗ Performance comparison error: {e}")
        return False


# ---------------------------------------------------------------------------
# Pytest-facing test cases
# ---------------------------------------------------------------------------

def test_spline_derivatives(prices_series):
    success, derivatives = run_spline_derivatives_check(prices_series)
    assert success, "Spline derivative analyzer failed to produce stable derivatives."
    assert derivatives is not None and 'velocity' in derivatives
    assert len(derivatives['velocity']) == len(prices_series)


def test_wavelet_denoising(prices_series):
    success, denoised = run_wavelet_denoising_check(prices_series)
    assert success, "Wavelet denoising pipeline did not complete."
    assert denoised is not None
    assert len(denoised) == len(prices_series)


def test_emd_denoising(prices_series):
    success, denoised = run_emd_denoising_check(prices_series)
    assert success, "EMD denoising pipeline did not complete."
    assert denoised is not None
    assert len(denoised) == len(prices_series)


def test_enhanced_calculus_analysis(prices_series):
    success, results = run_enhanced_calculus_analysis_check(prices_series)
    assert success, "Enhanced calculus analyzer failed to return results."
    assert results is not None
    assert not results.empty
    for column in ['smoothed_price', 'velocity', 'acceleration', 'enhanced_forecast']:
        assert column in results.columns, f"Missing expected column '{column}'"


def test_performance_comparison(prices_series):
    assert run_performance_comparison(prices_series), "Performance comparison did not succeed."

def main():
    """
    Main test function for all mathematical upgrades.
    """
    logger.info("Starting Mathematical Upgrades Test Suite")
    logger.info("=" * 50)
    
    # Generate test data
    test_data = generate_test_data(days=30, trend=True, noise_level=0.03)
    prices = test_data['price']
    
    # Run all tests
    test_results = []
    
    # Test 1: Spline derivatives
    success, spline_results = run_spline_derivatives_check(prices)
    test_results.append(("Spline Derivatives", success))
    
    # Test 2: Wavelet denoising
    success, wavelet_results = run_wavelet_denoising_check(prices)
    test_results.append(("Wavelet Denoising", success))
    
    # Test 3: EMD denoising
    success, emd_results = run_emd_denoising_check(prices)
    test_results.append(("EMD Denoising", success))
    
    # Test 4: Enhanced calculus analysis
    success, calculus_results = run_enhanced_calculus_analysis_check(prices)
    test_results.append(("Enhanced Calculus Analysis", success))
    
    # Test 5: Performance comparison
    success = run_performance_comparison(prices)
    test_results.append(("Performance Comparison", success))
    
    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("TEST SUMMARY")
    logger.info("=" * 50)
    
    passed = 0
    for test_name, result in test_results:
        status = "✓ PASS" if result else "✗ FAIL"
        logger.info(f"{test_name:<30}: {status}")
        if result:
            passed += 1
    
    logger.info(f"\nOverall: {passed}/{len(test_results)} tests passed")
    
    if passed == len(test_results):
        logger.info("🎉 All mathematical upgrades are working correctly!")
        return 0
    else:
        logger.error("❌ Some mathematical upgrades need attention.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
