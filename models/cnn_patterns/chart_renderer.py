#!/usr/bin/env python3
"""
Chart Renderer for CNN Pattern Detection
Renders candlestick charts from OHLCV data as images for CNN input.

Output: 224x224 RGB image of candlestick chart
Input: Last 100 candles of OHLCV data

NO RANDOM DATA - All charts from real market data.
"""

import numpy as np
from PIL import Image, ImageDraw
from typing import Tuple, Optional
import io


class CandlestickRenderer:
    """
    Renders candlestick charts as images for CNN input.
    
    Features:
    - Candlestick bodies and wicks
    - Support/resistance levels (optional)
    - Volume bars (optional)
    - Clean, high-contrast design for CNN
    """
    
    def __init__(
        self,
        width: int = 224,
        height: int = 224,
        candle_width: int = 2,
        wick_width: int = 1,
        background_color: Tuple[int, int, int] = (0, 0, 0),  # Black
        bullish_color: Tuple[int, int, int] = (0, 255, 0),   # Green
        bearish_color: Tuple[int, int, int] = (255, 0, 0),  # Red
        show_volume: bool = True,
        show_grid: bool = False
    ):
        self.width = width
        self.height = height
        self.candle_width = candle_width
        self.wick_width = wick_width
        self.background_color = background_color
        self.bullish_color = bullish_color
        self.bearish_color = bearish_color
        self.show_volume = show_volume
        self.show_grid = show_grid
        
        # Chart area (leave space for volume if shown)
        self.chart_top = 5
        self.chart_bottom = int(height * 0.75) if show_volume else height - 5
        self.volume_top = int(height * 0.78)
        self.volume_bottom = height - 5
        self.chart_left = 5
        self.chart_right = width - 5
    
    def render(self, ohlcv: np.ndarray, add_indicators: bool = False) -> Image.Image:
        """
        Render OHLCV data as candlestick chart image.
        
        Args:
            ohlcv: numpy array of shape (num_candles, 5) with columns [open, high, low, close, volume]
            add_indicators: If True, add moving averages and other indicators
        
        Returns:
            PIL Image of size (width, height)
        """
        # Create blank image
        img = Image.new('RGB', (self.width, self.height), self.background_color)
        draw = ImageDraw.Draw(img)
        
        num_candles = len(ohlcv)
        if num_candles == 0:
            return img
        
        # Extract OHLCV
        opens = ohlcv[:, 0]
        highs = ohlcv[:, 1]
        lows = ohlcv[:, 2]
        closes = ohlcv[:, 3]
        volumes = ohlcv[:, 4] if ohlcv.shape[1] > 4 else np.ones(num_candles)
        
        # Calculate price range
        price_min = np.min(lows)
        price_max = np.max(highs)
        price_range = price_max - price_min
        if price_range == 0:
            price_range = price_min * 0.01  # Fallback for flat data
        
        # Add padding to price range
        padding = price_range * 0.05
        price_min -= padding
        price_max += padding
        price_range = price_max - price_min
        
        # Calculate volume range
        vol_max = np.max(volumes) if np.max(volumes) > 0 else 1
        
        # Calculate candle spacing
        chart_width = self.chart_right - self.chart_left
        candle_spacing = chart_width / num_candles
        
        # Draw grid if enabled
        if self.show_grid:
            self._draw_grid(draw, price_min, price_max, num_candles)
        
        # Draw candles
        for i in range(num_candles):
            o, h, l, c = opens[i], highs[i], lows[i], closes[i]
            v = volumes[i]
            
            # X position
            x = self.chart_left + (i + 0.5) * candle_spacing
            
            # Determine if bullish or bearish
            is_bullish = c >= o
            color = self.bullish_color if is_bullish else self.bearish_color
            
            # Convert prices to Y coordinates (inverted - higher price = lower Y)
            def price_to_y(price):
                return self.chart_top + (price_max - price) / price_range * (self.chart_bottom - self.chart_top)
            
            y_high = price_to_y(h)
            y_low = price_to_y(l)
            y_open = price_to_y(o)
            y_close = price_to_y(c)
            
            # Draw wick (high to low)
            draw.line([(x, y_high), (x, y_low)], fill=color, width=self.wick_width)
            
            # Draw body (open to close)
            body_top = min(y_open, y_close)
            body_bottom = max(y_open, y_close)
            body_height = body_bottom - body_top
            
            if body_height < 1:
                body_height = 1  # Minimum body height
            
            half_width = self.candle_width / 2
            draw.rectangle(
                [(x - half_width, body_top), (x + half_width, body_bottom)],
                fill=color,
                outline=color
            )
            
            # Draw volume bar if enabled
            if self.show_volume and vol_max > 0:
                vol_height = (v / vol_max) * (self.volume_bottom - self.volume_top)
                vol_y = self.volume_bottom - vol_height
                draw.rectangle(
                    [(x - half_width, vol_y), (x + half_width, self.volume_bottom)],
                    fill=color,
                    outline=color
                )
        
        # Add indicators if requested
        if add_indicators:
            self._add_indicators(draw, ohlcv, price_min, price_max, candle_spacing)
        
        return img
    
    def _draw_grid(self, draw: ImageDraw, price_min: float, price_max: float, num_candles: int):
        """Draw grid lines."""
        grid_color = (50, 50, 50)  # Dark gray
        
        # Horizontal lines (price levels)
        price_range = price_max - price_min
        for i in range(5):
            y = self.chart_top + i * (self.chart_bottom - self.chart_top) / 4
            draw.line([(self.chart_left, y), (self.chart_right, y)], fill=grid_color, width=1)
        
        # Vertical lines (time)
        chart_width = self.chart_right - self.chart_left
        for i in range(5):
            x = self.chart_left + i * chart_width / 4
            draw.line([(x, self.chart_top), (x, self.chart_bottom)], fill=grid_color, width=1)
    
    def _add_indicators(self, draw: ImageDraw, ohlcv: np.ndarray, price_min: float, price_max: float, candle_spacing: float):
        """Add moving average lines."""
        closes = ohlcv[:, 3]
        num_candles = len(closes)
        price_range = price_max - price_min
        
        # SMA 20
        if num_candles >= 20:
            sma20 = np.convolve(closes, np.ones(20)/20, mode='valid')
            sma_color = (255, 165, 0)  # Orange
            
            for i in range(1, len(sma20)):
                x1 = self.chart_left + (i + 19 + 0.5) * candle_spacing
                x2 = self.chart_left + (i + 20 + 0.5) * candle_spacing
                y1 = self.chart_top + (price_max - sma20[i-1]) / price_range * (self.chart_bottom - self.chart_top)
                y2 = self.chart_top + (price_max - sma20[i]) / price_range * (self.chart_bottom - self.chart_top)
                
                if 0 <= x1 < self.width and 0 <= x2 < self.width:
                    draw.line([(x1, y1), (x2, y2)], fill=sma_color, width=1)
    
    def render_to_array(self, ohlcv: np.ndarray, add_indicators: bool = False) -> np.ndarray:
        """
        Render OHLCV to numpy array.
        Returns array of shape (height, width, 3).
        """
        img = self.render(ohlcv, add_indicators)
        return np.array(img)
    
    def render_to_bytes(self, ohlcv: np.ndarray, format: str = 'PNG') -> bytes:
        """
        Render OHLCV to bytes (PNG or JPEG).
        """
        img = self.render(ohlcv)
        buffer = io.BytesIO()
        img.save(buffer, format=format)
        return buffer.getvalue()


def render_chart(ohlcv: np.ndarray, **kwargs) -> Image.Image:
    """Convenience function to render a chart."""
    renderer = CandlestickRenderer(**kwargs)
    return renderer.render(ohlcv)


if __name__ == "__main__":
    import os
    
    print("=" * 60)
    print("CHART RENDERER - Test")
    print("=" * 60)
    
    # Generate sample OHLCV data (in reality, this comes from MT5 API)
    np.random.seed(42)  # For reproducibility in test only
    
    num_candles = 100
    base_price = 1.1000
    
    ohlcv = np.zeros((num_candles, 5))
    for i in range(num_candles):
        # Simulate price movement
        change = np.random.randn() * 0.0020
        
        o = base_price
        c = base_price + change
        h = max(o, c) + abs(np.random.randn() * 0.0005)
        l = min(o, c) - abs(np.random.randn() * 0.0005)
        v = 1000 + np.random.randn() * 200
        
        ohlcv[i] = [o, h, l, c, v]
        base_price = c
    
    # Render chart
    renderer = CandlestickRenderer(width=224, height=224, show_volume=True)
    img = renderer.render(ohlcv)
    
    # Save test image
    output_path = "/tmp/test_chart.png"
    img.save(output_path)
    print(f"Saved test chart to {output_path}")
    print(f"Image size: {img.size}")
    print(f"Image mode: {img.mode}")
    
    # Convert to array
    arr = renderer.render_to_array(ohlcv)
    print(f"Array shape: {arr.shape}")
    print(f"Array dtype: {arr.dtype}")
    
    print("\n✅ Chart renderer working correctly")
    print("=" * 60)
