"""
Position Side Logic - Single Source of Truth
============================================
Canonical implementation of position side determination for all trading strategies.

Mathematical Foundation:
- Trend Following: Trade in direction of velocity (momentum continuation)
- Mean Reversion: Trade against velocity (expect mean reversion)

Author: Quantitative System Reconstruction
Date: 2025-11-10
"""

from calculus_strategy import SignalType
import logging

logger = logging.getLogger(__name__)


def determine_position_side(signal_type: SignalType, velocity: float) -> str:
    """
    Canonical function for position side determination.
    
    This is the SINGLE SOURCE OF TRUTH for mapping (signal_type, velocity) → position_side.
    Used by:
    - risk_manager.calculate_dynamic_tp_sl()
    - live_calculus_trader._execute_trade()
    - Any position sizing logic
    
    Strategy Logic:
    ---------------
    1. Trend Following Signals (BUY, STRONG_BUY, POSSIBLE_LONG, SELL, STRONG_SELL):
       - BUY signals → long position (expect upward continuation)
       - SELL signals → short position (expect downward continuation)
    
    2. Mean Reversion Signals (NEUTRAL):
       - velocity > 0 (rising/overbought) → short position (expect pullback)
       - velocity < 0 (falling/oversold) → long position (expect bounce)
       
       Mathematical Justification:
       - In range-bound markets, price oscillates around mean
       - Extreme deviations (high |velocity|) tend to revert
       - Trade against momentum for mean reversion edge
    
    Args:
        signal_type: Type of trading signal from calculus analysis
        velocity: Instantaneous rate of change ($/second)
    
    Returns:
        position_side: "long" or "short"
    
    Examples:
        >>> determine_position_side(SignalType.BUY, 0.5)
        'long'
        >>> determine_position_side(SignalType.NEUTRAL, 0.001866)  # Rising
        'short'  # Mean reversion: expect pullback
        >>> determine_position_side(SignalType.NEUTRAL, -0.001866)  # Falling
        'long'   # Mean reversion: expect bounce
    """
    # Trend Following Signals: Trade with momentum
    if signal_type in [SignalType.BUY, SignalType.STRONG_BUY, SignalType.POSSIBLE_LONG]:
        return "long"
    
    if signal_type in [SignalType.SELL, SignalType.STRONG_SELL]:
        return "short"
    
    # Mean Reversion Signal: Trade against momentum
    if signal_type == SignalType.NEUTRAL:
        # Velocity >= 0 (rising or flat) → short (expect mean reversion down)
        # Velocity < 0 (falling) → long (expect mean reversion up)
        return "short" if velocity >= 0 else "long"
    
    # Default fallback: Trend following for other signal types
    return "long" if velocity > 0 else "short"


def determine_trade_side(signal_type: SignalType, velocity: float) -> str:
    """
    Determine trade execution side (Buy/Sell) from position side.
    
    Mapping:
        position_side = "long" → trade_side = "Buy"
        position_side = "short" → trade_side = "Sell"
    
    Args:
        signal_type: Type of trading signal
        velocity: Instantaneous rate of change
    
    Returns:
        trade_side: "Buy" or "Sell"
    """
    position_side = determine_position_side(signal_type, velocity)
    return "Buy" if position_side == "long" else "Sell"


def validate_position_consistency(
    signal_type: SignalType,
    velocity: float,
    intended_side: str,
    position_side: str
) -> tuple[bool, str]:
    """
    Validate that intended trade matches canonical position side logic.
    
    This catches inconsistencies where different parts of the code
    calculate position side differently.
    
    Args:
        signal_type: Trading signal type
        velocity: Current velocity
        intended_side: What the code wants to do ("Buy"/"Sell" or "long"/"short")
        position_side: What calculate_dynamic_tp_sl calculated
    
    Returns:
        (is_valid, error_message)
    """
    canonical_side = determine_position_side(signal_type, velocity)
    
    # Normalize intended_side to position terminology
    if intended_side in ["Buy", "long"]:
        intended_position = "long"
    elif intended_side in ["Sell", "short"]:
        intended_position = "short"
    else:
        return False, f"Invalid intended_side: {intended_side}"
    
    if intended_position != canonical_side:
        return False, (
            f"Position side mismatch: "
            f"Canonical={canonical_side}, "
            f"Intended={intended_position}, "
            f"signal_type={signal_type.name}, "
            f"velocity={velocity:.6f}"
        )
    
    if position_side != canonical_side:
        return False, (
            f"TP/SL calculated for wrong side: "
            f"Canonical={canonical_side}, "
            f"TP/SL_calculated_for={position_side}, "
            f"signal_type={signal_type.name}"
        )
    
    return True, "Position side consistent"
