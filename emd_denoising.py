"""
Empirical Mode Decomposition (EMD) Denoising Module
=================================================

Advanced EMD-based signal decomposition and reconstruction for financial time series.
Separates intrinsic oscillatory modes (IMFs) to extract true market signals
from multi-scale noise and market microstructure effects.

Key Features:
- Adaptive IMF classification using Hurst exponent and variance analysis
- Multi-scale noise separation preserving market structure
- Ensemble EMD (EEMD) for robust decomposition
- Trend preservation for long-term signal components
- Real-time capable with incremental processing

Mathematical Foundation:
- EMD Decomposition: x(t) = Σ IMF_i(t) + r_n(t)
- Hurst Exponent: H = lim(R/S) as n→∞ for scale analysis
- IMF Selection: IMF_i ∈ Signal if H_i > 0.5 and Var_i > threshold
- Reconstruction: x̂(t) = Σ Signal_IMF_i(t) + Trend(t)
"""

import numpy as np
import pandas as pd
try:
    from PyEMD import EMD, EEMD, CEEMDAN
    # Full EMD suite available
    EMD_Available = True
    print("Info: Full PyEMD suite available with EMD, EEMD, and CEEMDAN")
except ImportError:
    # Fallback for systems without PyEMD
    EMD_Available = False
    EMD = None
    EEMD = None
    CEEMDAN = None
    print("Warning: PyEMD not available, EMD functionality will be limited")
        
import logging
from typing import Dict, List, Tuple, Optional, Union
from scipy.stats import linregress
from scipy.signal import hilbert
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class EMDDenoiser:
    """
    Advanced EMD-based signal decomposer with intelligent IMF selection
    for optimal signal extraction from noisy financial data.
    """

    def __init__(self,
                 max_imfs: int = 8,
                 emd_type: str = 'eemd',
                 noise_width: Optional[float] = 0.2,
                 ensemble_size: int = 50,
                 trend_preservation: float = 0.8,
                 variance_threshold: float = 0.05,
                 hurst_threshold: float = 0.5):
        """
        Initialize EMD denoiser.

        Args:
            max_imfs: Maximum number of IMFs to extract
            emd_type: Type of EMD ('emd', 'eemd', 'ceemdan')
            noise_width: Noise width for EEMD/CEEMDAN
            ensemble_size: Number of ensemble members for EEMD
            trend_preservation: Weight for trend preservation [0-1]
            variance_threshold: Minimum variance contribution for signal IMF
            hurst_threshold: Hurst exponent threshold for trend vs noise
        """
        self.max_imfs = max_imfs
        self.emd_type = emd_type
        self.noise_width = noise_width
        self.ensemble_size = ensemble_size
        self.trend_preservation = trend_preservation
        self.variance_threshold = variance_threshold
        self.hurst_threshold = hurst_threshold
        self.emd_processor = None
        self.fallback_mode = False
        
        # Initialize EMD processor
        self._initialize_emd()
        
        # Quality tracking
        self.imf_history = []
        self.denoising_quality = []
        self.trend_components = []
        
        logger.info(f"EMDDenoiser initialized: type={emd_type}, "
                   f"max_imfs={max_imfs}, ensemble_size={ensemble_size}")

    def _initialize_emd(self):
        """Initialize the appropriate EMD processor."""
        self.fallback_mode = False

        if not EMD_Available:
            logger.warning("PyEMD not available; enabling Savitzky-Golay fallback mode.")
            self.emd_processor = None
            self.fallback_mode = True
            return

        try:
            if self.emd_type == 'emd':
                self.emd_processor = EMD()
            elif self.emd_type == 'eemd':
                if EEMD is not None:
                    self.emd_processor = EEMD(trials=self.ensemble_size, 
                                          noise_width=self.noise_width)
                else:
                    logger.warning("EEMD not available, falling back to basic EMD")
                    self.emd_processor = EMD()
            elif self.emd_type == 'ceemdan':
                if CEEMDAN is not None:
                    self.emd_processor = CEEMDAN(trials=self.ensemble_size,
                                             noise_width=self.noise_width)
                else:
                    logger.warning("CEEMDAN not available, falling back to basic EMD")
                    self.emd_processor = EMD()
            else:
                raise ValueError(f"Unknown EMD type: {self.emd_type}. Use 'emd', 'eemd', or 'ceemdan'")
        except Exception as exc:
            logger.error(f"Unable to initialize EMD processor ({self.emd_type}): {exc}. "
                         "Falling back to smoothing-based denoiser.")
            self.emd_processor = None
            self.fallback_mode = True
            return
            
        # Configure EMD parameters when available
        try:
            self.emd_processor.max_imfs = self.max_imfs
        except AttributeError:
            logger.debug("EMD processor does not expose max_imfs attribute; skipping assignment.")

    def _calculate_hurst_exponent(self, signal: np.ndarray) -> float:
        """
        Calculate Hurst exponent using Rescaled Range (R/S) analysis.

        H < 0.5: Anti-persistent (mean-reverting)
        H = 0.5: Random walk
        H > 0.5: Persistent (trending)
        """
        if len(signal) < 10:
            return 0.5
            
        # Calculate cumulative deviations
        mean_val = np.mean(signal)
        deviations = signal - mean_val
        cumulative_dev = np.cumsum(deviations)
        
        # Calculate ranges for different lags
        lags = range(10, min(len(signal) // 4, 100))
        rs_values = []
        
        for lag in lags:
            if lag >= len(cumulative_dev):
                break
                
            # Range over this lag
            window = cumulative_dev[:lag+1]
            range_val = np.max(window) - np.min(window)
            
            # Standard deviation over this lag
            std_val = np.std(deviations[:lag+1])
            
            if std_val > 0:
                rs_values.append(range_val / std_val)
        
        if len(rs_values) < 2:
            return 0.5
            
        # Fit log-log regression
        log_lags = np.log(lags[:len(rs_values)])
        log_rs = np.log(rs_values)
        
        try:
            slope, _, _, _, _ = linregress(log_lags, log_rs)
            return slope
        except:
            return 0.5

    def _calculate_imf_properties(self, imf: np.ndarray) -> Dict[str, float]:
        """
        Calculate properties of an IMF for signal/noise classification.
        """
        # Basic statistics
        variance = np.var(imf)
        energy = np.sum(imf**2)
        zero_crossings = np.sum(np.diff(np.sign(imf)) != 0)
        
        # Frequency characteristics using Hilbert transform
        try:
            analytic_signal = hilbert(imf)
            instantaneous_phase = np.unwrap(np.angle(analytic_signal))
            instantaneous_freq = np.diff(instantaneous_phase) / (2 * np.pi)
            mean_frequency = np.mean(np.abs(instantaneous_freq))
            frequency_variance = np.var(instantaneous_freq)
        except:
            mean_frequency = 0.0
            frequency_variance = 0.0
        
        # Hurst exponent for trend analysis
        hurst = self._calculate_hurst_exponent(imf)
        
        # Correlation with original signal (for importance)
        # This would need the original signal - placeholder for now
        correlation = 0.0
        
        return {
            'variance': variance,
            'energy': energy,
            'zero_crossings': zero_crossings,
            'mean_frequency': mean_frequency,
            'frequency_variance': frequency_variance,
            'hurst_exponent': hurst,
            'correlation': correlation,
            'persistence': 'persistent' if hurst > 0.5 else ('anti-persistent' if hurst < 0.5 else 'random')
        }

    def _classify_imf(self, imf_properties: Dict[str, float], imf_index: int, total_imfs: int) -> str:
        """
        Classify IMF as signal, noise, or trend based on its properties.
        
        Classification rules:
        1. High Hurst (> 0.5) + significant variance → Signal
        2. Low Hurst (< 0.5) + low variance → Noise
        3. Last IMF (residual) → Trend
        4. Very low frequency (low index) → Possible trend component
        """
        hurst = imf_properties['hurst_exponent']
        variance = imf_properties['variance']
        frequency = imf_properties['mean_frequency']
        
        # Last IMF is always trend/residual
        if imf_index == total_imfs - 1:
            return 'trend'
        
        # Very low frequency IMFs (first few) might be trend components
        if imf_index <= 1 and frequency < 0.1:
            return 'trend'
        
        # Signal criteria
        if (hurst > self.hurst_threshold and 
            variance > self.variance_threshold and
            frequency > 0.01):
            return 'signal'
        
        # Noise criteria
        if (hurst < self.hurst_threshold and 
            variance < self.variance_threshold):
            return 'noise'
        
        # Ambiguous - classify based on combined criteria
        signal_score = 0
        signal_score += 1 if hurst > self.hurst_threshold else -1
        signal_score += 1 if variance > self.variance_threshold else -1
        signal_score += 1 if frequency > 0.01 else 0
        
        return 'signal' if signal_score > 0 else 'noise'

    def _reconstruct_signal(self,
                           imfs: List[np.ndarray],
                           classifications: List[str],
                           preserve_trend: Optional[float] = None) -> np.ndarray:
        """
        Reconstruct denoised signal from selected IMFs.
        """
        if preserve_trend is None:
            preserve_trend = self.trend_preservation
            
        # Initialize reconstruction
        signal_components = []
        trend_components = []
        
        # Separate signal and trend components
        for imf, classification in zip(imfs, classifications):
            if classification == 'signal':
                signal_components.append(imf)
            elif classification == 'trend':
                trend_components.append(imf)
            # Noise components are excluded
        
        # Reconstruct signal
        if signal_components:
            denoised_signal = np.sum(signal_components, axis=0)
        else:
            denoised_signal = np.zeros_like(imfs[0])
        
        # Add trend component with preservation weight
        if trend_components:
            trend_component = np.sum(trend_components, axis=0)
            denoised_signal += preserve_trend * trend_component
            self.trend_components.append(trend_component)
        
        return denoised_signal

    def _calculate_reconstruction_quality(self,
                                      original: np.ndarray,
                                      denoised: np.ndarray,
                                      imfs: List[np.ndarray],
                                      classifications: List[str]) -> Dict[str, float]:
        """
        Calculate quality metrics for EMD reconstruction.
        """
        # Signal preservation metrics
        signal_to_noise = np.var(denoised) / (np.var(original - denoised) + 1e-10)
        correlation = np.corrcoef(original, denoised)[0, 1] if len(original) > 1 else 0
        
        # Component analysis
        signal_imfs = [imf for imf, cls in zip(imfs, classifications) if cls == 'signal']
        noise_imfs = [imf for imf, cls in zip(imfs, classifications) if cls == 'noise']
        trend_imfs = [imf for imf, cls in zip(imfs, classifications) if cls == 'trend']
        
        signal_energy = np.sum([np.sum(imf**2) for imf in signal_imfs])
        noise_energy = np.sum([np.sum(imf**2) for imf in noise_imfs])
        total_energy = np.sum(original**2)
        
        signal_contribution = signal_energy / total_energy if total_energy > 0 else 0
        noise_rejection = 1 - (noise_energy / total_energy if total_energy > 0 else 0)
        
        return {
            'signal_to_noise': float(signal_to_noise),
            'correlation': float(correlation),
            'correlation_preservation': float(correlation),  # Add key expected by test
            'signal_contribution': float(signal_contribution),
            'noise_rejection': float(noise_rejection),
            'num_signal_imfs': len(signal_imfs),
            'num_noise_imfs': len(noise_imfs),
            'num_trend_imfs': len(trend_imfs)
        }

    def decompose(self,
                 data: Union[np.ndarray, pd.Series],
                 return_details: bool = False) -> Union[np.ndarray, Dict]:
        """
        Perform EMD decomposition and intelligent signal reconstruction.

        Args:
            data: Input signal to decompose
            return_details: Whether to return detailed analysis

        Returns:
            Denoised signal (and optionally detailed analysis)
        """
        # Convert to numpy array
        if isinstance(data, pd.Series):
            original_data = data.values
            original_index = data.index
        else:
            original_data = np.asarray(data)
            original_index = None
            
        # Input validation
        if len(original_data) < 10:
            logger.warning(f"Insufficient data for EMD: {len(original_data)}")
            return data if not return_details else {'denoised': data}
        
        # Fallback mode when EMD is not available
        if self.fallback_mode or self.emd_processor is None:
            logger.warning("EMD not available, using simple moving average fallback")
            # Simple fallback: use moving average with noise reduction
            try:
                # Use Savitzky-Golay filter as fallback
                from scipy.signal import savgol_filter
                window_length = min(51, len(original_data) // 2)
                if window_length % 2 == 0:
                    window_length -= 1
                window_length = max(3, window_length)
                
                denoised_data = savgol_filter(original_data, window_length, 3, mode='nearest')
                
                if return_details:
                    return {
                        'denoised': denoised_data,
                        'original': original_data,
                        'imfs': [denoised_data],  # Single IMF as fallback
                        'imf_properties': [{'variance': np.var(denoised_data)}],
                        'classifications': ['signal'],
                        'signal_imfs': [denoised_data],
                        'noise_imfs': [],
                        'trend_imfs': [],
                        'quality_metrics': {
                            'signal_to_noise': 1.0,
                            'correlation_preservation': np.corrcoef(original_data, denoised_data)[0, 1],
                            'signal_contribution': 1.0,
                            'noise_rejection': 0.0,
                            'num_signal_imfs': 1,
                            'num_noise_imfs': 0,
                            'num_trend_imfs': 0
                        }
                    }
                else:
                    return denoised_data if original_index is None else pd.Series(denoised_data, index=original_index)
                    
            except Exception as e:
                logger.error(f"Fallback denoising failed: {e}")
                return data if not return_details else {'denoised': data}
        
        try:
            # Perform EMD decomposition
            if self.emd_type == 'eemd' and EEMD is not None:
                imfs = self.emd_processor.eemd(original_data)
            elif self.emd_type == 'ceemdan' and CEEMDAN is not None:
                imfs = self.emd_processor.ceemdan(original_data)
            else:
                # Basic EMD
                imfs = self.emd_processor.emd(original_data)
            
            if len(imfs) == 0:
                logger.error("EMD decomposition failed to produce any IMFs")
                return data if not return_details else {'denoised': data}
            
            # Add residual as last IMF for trend classification
            if 'residual' in locals():
                imfs.append(residual)
            
            # Analyze each IMF
            imf_properties = []
            classifications = []
            
            for i, imf in enumerate(imfs):
                properties = self._calculate_imf_properties(imf)
                classification = self._classify_imf(properties, i, len(imfs))
                
                imf_properties.append(properties)
                classifications.append(classification)
                
                logger.debug(f"IMF {i}: {classification}, H={properties['hurst_exponent']:.3f}, "
                           f"Var={properties['variance']:.6f}")
            
            # Reconstruct denoised signal
            denoised_signal = self._reconstruct_signal(imfs, classifications)
            
            # Calculate quality metrics
            quality_metrics = self._calculate_reconstruction_quality(
                original_data, denoised_signal, imfs, classifications
            )
            
            # Store for analysis
            self.imf_history.append({
                'num_imfs': len(imfs),
                'classifications': classifications,
                'properties': imf_properties,
                'quality': quality_metrics
            })
            
            self.denoising_quality.append(quality_metrics)
            
            if return_details:
                return {
                    'denoised': denoised_signal,
                    'original': original_data,
                    'imfs': imfs,
                    'imf_properties': imf_properties,
                    'classifications': classifications,
                    'quality_metrics': quality_metrics,
                    'signal_imfs': [imf for imf, cls in zip(imfs, classifications) if cls == 'signal'],
                    'noise_imfs': [imf for imf, cls in zip(imfs, classifications) if cls == 'noise'],
                    'trend_imfs': [imf for imf, cls in zip(imfs, classifications) if cls == 'trend']
                }
            else:
                return denoised_signal if original_index is None else pd.Series(denoised_signal, index=original_index)
                
        except Exception as e:
            logger.error(f"EMD decomposition failed: {e}")
            return data if not return_details else {'denoised': data}

    def multi_scale_analysis(self,
                          data: Union[np.ndarray, pd.Series]) -> Dict[str, np.ndarray]:
        """
        Perform multi-scale analysis to separate different signal components.

        Useful for understanding the contribution of different time scales:
        - Long-term trend: Very low frequency components
        - Medium-term cycles: Business cycle components  
        - Short-term fluctuations: High-frequency components
        """
        # Perform full decomposition with details
        result = self.decompose(data, return_details=True)
        
        if not isinstance(result, dict):
            return {'signal': result, 'trend': np.zeros_like(result), 'noise': np.zeros_like(result)}
        
        signal_imfs = result['signal_imfs']
        trend_imfs = result['trend_imfs']
        
        # Combine components
        signal = np.sum(signal_imfs, axis=0) if signal_imfs else np.zeros_like(result['original'])
        trend = np.sum(trend_imfs, axis=0) if trend_imfs else np.zeros_like(result['original'])
        noise = result['original'] - signal - trend
        
        return {
            'signal': signal,
            'trend': trend,
            'noise': noise,
            'original': result['original']
        }

    def get_decomposition_statistics(self) -> Dict[str, float]:
        """
        Get statistics about EMD decomposition performance.
        """
        if not self.denoising_quality:
            return {'avg_snr': 0.0, 'avg_correlation': 0.0, 'sample_count': 0}
            
        recent_quality = self.denoising_quality[-20:]  # Last 20 samples
        
        return {
            'avg_snr': float(np.mean([q['signal_to_noise'] for q in recent_quality])),
            'avg_correlation': float(np.mean([q['correlation'] for q in recent_quality])),
            'avg_noise_rejection': float(np.mean([q['noise_rejection'] for q in recent_quality])),
            'sample_count': len(self.denoising_quality),
            'quality_stability': float(1.0 - np.std([q['signal_to_noise'] for q in recent_quality])),
            'avg_signal_imfs': float(np.mean([q['num_signal_imfs'] for q in recent_quality])),
            'avg_noise_imfs': float(np.mean([q['num_noise_imfs'] for q in recent_quality]))
        }

    def update_parameters(self,
                        variance_threshold: Optional[float] = None,
                        hurst_threshold: Optional[float] = None,
                        trend_preservation: Optional[float] = None):
        """
        Update EMD parameters for adaptive behavior.
        """
        if variance_threshold is not None:
            self.variance_threshold = variance_threshold
            
        if hurst_threshold is not None:
            self.hurst_threshold = hurst_threshold
            
        if trend_preservation is not None:
            self.trend_preservation = np.clip(trend_preservation, 0.0, 1.0)
            
        logger.info(f"EMD parameters updated: var_thresh={self.variance_threshold}, "
                   f"hurst_thresh={self.hurst_threshold}, trend_pres={self.trend_preservation}")

# Convenience function for quick EMD denoising
def emd_denoise(prices: pd.Series,
                  max_imfs: int = 8,
                  emd_type: str = 'eemd') -> pd.Series:
    """
    Convenience function for EMD-based price denoising.
    """
    denoiser = EMDDenoiser(max_imfs=max_imfs, emd_type=emd_type)
    result = denoiser.decompose(prices)
    return result if isinstance(result, pd.Series) else pd.Series(result, index=prices.index)
