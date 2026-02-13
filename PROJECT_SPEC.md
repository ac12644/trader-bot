# PROJECT SPECIFICATION v3
## Aggressive Crypto Futures Day Trading Bot
### Controlled Aggression Framework — $1,000 Starting Capital

---

## 1. OBJECTIVE

### Primary Objective

Build a fully automated crypto futures day trading bot engineered for aggressive capital compounding while maintaining hard risk controls that prevent account liquidation.

### Core Philosophy

This is **controlled aggression**. The bot exploits volatility expansion events with asymmetric risk-reward. It does not predict. It does not gamble. It reacts to confirmed market structure with strict discipline.

Profit comes from a small number of large winning trades. Most trades will lose. The system survives because losses are small and capped.

### Target Metrics (Honest)

| Metric | Value |
|---|---|
| Starting Capital | $1,000 |
| Target Monthly Return (good month) | 5-12% |
| Exceptional Month (strong trend) | 15-25% |
| Flat/Negative Month | Expected regularly |
| Max Acceptable Drawdown | 20% |
| Daily Loss Cap | 5% ($50 at start) |
| Weekly Loss Cap | 12% |
| Risk of Ruin Target | < 15% annually |
| Minimum Trades Before Scaling | 100 |

These numbers assume honest execution with fees, slippage, and regime variation. Do NOT optimize the bot to hit higher numbers — that leads to overfitting.

---

## 2. NON-NEGOTIABLE RULES

The bot must NEVER:

- Implement martingale or any loss-doubling logic
- Average down on a losing position
- Remove or widen a stop loss after placement
- Increase leverage dynamically
- Allow more than 3 open trades simultaneously
- Ignore fees or slippage in backtesting
- Use look-ahead bias or future candle data
- Trade after daily loss limit is hit
- Trade on unconfirmed (still open) candles
- Increase position size after a winning streak
- Enter without confirmed stop loss placement
- Hold a position longer than 4 hours
- Trade during known macro events without filter

These are hard constraints. If any optimization violates these, reject it.

---

## 3. TARGET ENVIRONMENT

| Parameter | Value |
|---|---|
| Asset Class | Crypto Perpetual Futures |
| Pairs | BTCUSDT, ETHUSDT only |
| Exchange | User-specified (Binance/Bybit) |
| Entry Timeframe | 5m (confirmed close only) |
| Trend Filter Timeframe | 1h |
| Regime Filter Timeframe | 1h ATR |
| Language | Python 3.11+ |
| Architecture | Async (asyncio + uvloop) |
| Deployment | Docker on VPS |
| Database | SQLite (upgrade to PostgreSQL at scale) |

### Why Only BTC and ETH

- Deepest liquidity — tightest spreads
- Most reliable execution
- Least susceptible to single-whale manipulation
- Sufficient volatility for breakout systems
- Avoid low-cap altcoins entirely — spread, slippage, and manipulation destroy small accounts

---

## 4. ARCHITECTURE

```
/core
    engine.py              # Main event loop and orchestration
    state_manager.py       # State persistence and crash recovery
/data
    websocket_client.py    # Exchange websocket with reconnect logic
    candle_builder.py      # Aggregates ticks into confirmed candles
    spread_monitor.py      # Real-time bid-ask spread tracking
/indicators
    technical.py           # ATR, EMA, ADX, Bollinger, volume metrics
    regime.py              # Chop detection, volatility regime classification
/strategies
    base_strategy.py       # Abstract interface all strategies implement
    volatility_breakout.py # Strategy 1: Expansion breakout
    pullback_continuation.py # Strategy 2: Trend pullback
    squeeze_breakout.py    # Strategy 3: Range compression breakout
/filters
    trend_filter.py        # Higher timeframe trend gate
    chop_filter.py         # ADX + ATR percentile chop detection
    spread_filter.py       # Pre-entry spread validation
    funding_filter.py      # Funding rate cost check
    event_filter.py        # Macro event calendar blackout
    session_filter.py      # Low-liquidity hour avoidance
    correlation_filter.py  # BTC/ETH single-exposure enforcement
/risk
    risk_manager.py        # Position sizing, daily loss, drawdown tracking
    kill_switch.py         # Emergency exit conditions
/execution
    order_manager.py       # Order placement, partial fills, retries
    exchange_adapter.py    # CCXT wrapper with error handling
/portfolio
    position_manager.py    # Open position tracking and reconciliation
    pnl_tracker.py         # Trade logging, daily/weekly/monthly P&L
/config
    settings.yaml          # All configurable parameters
    events_calendar.yaml   # Known macro event dates/times
/backtest
    backtest_engine.py     # Historical simulation with realistic costs
    metrics.py             # Sharpe, profit factor, drawdown, expectancy
    monte_carlo.py         # Drawdown probability simulation
```

Every module is independent. Strategy generates signals. Risk validates them. Execution places orders. Portfolio tracks state. No module reaches into another's responsibilities.

---

## 5. SYSTEM FLOW

```
1. Websocket receives tick data
2. Spread monitor updates current bid-ask spread
3. Candle builder aggregates into 5m candle
4. On confirmed candle close:
   a. Update all indicators (ATR, EMA, ADX, Bollinger, volume)
   b. Regime filter checks: is market tradeable?
      - ADX > 20 on 1h?
      - ATR percentile above 30th of 30-day range?
      - Not in low-liquidity session?
      - No macro event blackout active?
      → If ANY filter fails: skip, no signal generated
   c. Strategy generates signal (long/short/none)
   d. Correlation filter checks: BTC+ETH combined exposure
   e. Risk manager validates:
      - Daily loss limit not hit?
      - Weekly loss limit not hit?
      - Position size within limits?
      - Max concurrent positions not exceeded?
      - Cooldown period not active?
   f. Funding rate filter: is funding cost acceptable?
   g. Spread filter: is current spread within limits?
   h. Execution module places order (limit IOC preferred)
   i. If filled: immediately place stop loss order on exchange
   j. Portfolio updates state, persists to database
5. Position monitor runs continuously:
   a. Trailing stop management after 1R profit
   b. Time-based exit: close if held > 4 hours
   c. Stale position exit: close if < 1R profit after 2 hours
   d. Kill switch monitoring (abnormal slippage, disconnect, spread spike)
6. On exit: log trade, update P&L, check daily limits
```

Signal generation must NEVER execute trades directly. The flow is always: Signal → Filters → Risk → Execution.

---

## 6. FILTER SYSTEM (CRITICAL)

Filters prevent the bot from trading in conditions where breakout strategies have negative expectancy. Most bot failures come from **trading when they shouldn't**, not from bad entry logic.

### 6A. Chop Filter

Breakout strategies bleed in ranging markets. This filter prevents that.

**Rules:**
- Calculate ADX on 1h timeframe
- If ADX < 20: **no trading** (market is ranging/choppy)
- Calculate ATR percentile: current 1h ATR vs rolling 30-day ATR distribution
- If ATR percentile < 30th percentile: **no trading** (volatility too low for breakouts)
- If 3 consecutive breakout entries hit stop loss: **pause trading for 4 hours** (likely in chop)

### 6B. Spread Filter

During volatile moments, spreads widen and destroy fill quality.

**Rules:**
- Track rolling 100-tick average spread
- Before entry: check current spread vs average
- If spread > 2x average: **delay entry by 1 candle**
- If spread > 3x average: **skip trade entirely**

### 6C. Funding Rate Filter

Perpetual futures charge/pay funding every 8 hours. Trading against high funding is a hidden cost.

**Rules:**
- Before entry, fetch current funding rate
- If funding > 0.05% against your trade direction: **reduce position size by 30%**
- If funding > 0.10% against your direction: **skip trade entirely**
- If funding > 0.05% in your favor: **standard position size** (funding is tailwind)

### 6D. Session Filter

Crypto has distinct liquidity profiles throughout the day.

**Rules:**
- No new entries between **00:00-04:00 UTC** on weekdays (lowest liquidity)
- No new entries between **04:00-06:00 UTC** on weekends
- These are the most common times for manipulation wicks on thin order books

### 6E. Event Filter

Macro events cause whipsaws that destroy breakout entries.

**Rules:**
- Maintain a calendar of known events: FOMC, CPI, PPI, NFP, major ETH/BTC upgrades, exchange maintenance windows
- No new entries **30 minutes before** event
- No new entries **15 minutes after** event
- If unrealized P&L on any open position swings > 3% in a single 5m candle: **emergency close all positions**

### 6F. Correlation Filter (BTC/ETH Exposure Control)

BTC and ETH are 80-90% correlated. Two positions in the same direction = doubled risk.

**Rules:**
- Treat BTC + ETH as a **single exposure bucket**
- Max 1 position if both are in the same direction
- If BTC long is open and ETH long signal fires: **skip ETH trade**
- Opposite direction allowed (BTC long + ETH short = hedged, acceptable)
- Combined notional exposure across both must never exceed single-trade risk limits

---

## 7. STRATEGY SET

Three strategies. Each exploits a different volatility pattern. The bot does not need all three active from day one — start with Strategy 1 only, add others after 100+ trades validate the framework.

### Strategy 1: Volatility Expansion Breakout (Primary)

**Purpose:** Capture explosive momentum moves when price breaks out of consolidation on expanding volume.

**Long Entry Conditions (ALL must be true):**
1. 5m candle CLOSES above the highest high of the last 20 candles
2. Volume on that candle >= 1.7x the 20-period average volume
3. 1h EMA 21 > EMA 55 (trend is bullish)
4. 1h EMA 21 slope is positive (trend is accelerating, not flattening)
5. 1h ATR is above 30th percentile of 30-day distribution
6. ADX on 1h > 20

**Short Entry Conditions:**
- Mirror logic: 5m close below 20-candle low, volume spike, 1h EMA 21 < EMA 55, negative slope, same volatility/ADX requirements.

**Exit Rules:**
- Stop Loss: 1.4 x ATR(14) on 5m, placed immediately on exchange after fill
- Take Profit: 2.5R fixed target
- Trailing Stop: After position reaches 1R profit, trail stop at 1.2 x ATR below/above current price, updating every new 5m candle
- Time Exit: If position open > 4 hours, close at market
- Stale Exit: If position open > 2 hours and unrealized P&L < 1R, close at market

**Frequency:** Max 2 entries per symbol per day. Max 4 total entries per day across all symbols.

**Expected Performance:**
- Win rate: ~38-44%
- Average win (after trailing): ~1.6-2.0R
- Average loss: 1R
- Positive expectancy over 200+ trade sample

### Strategy 2: Pullback Continuation

**Purpose:** Enter an established trend after a minor retracement to a structural level, rather than chasing the breakout.

**Long Entry Conditions (ALL must be true):**
1. 1h EMA 21 > EMA 55 (confirmed uptrend)
2. 5m price retraces to within 0.3% of the 1h EMA 21 (touching or near the dynamic support)
3. 5m candle prints a bullish reversal pattern: engulfing, hammer, or pin bar
4. Volume on the reversal candle >= 1.3x average (buyers stepping in)
5. Price is still above the 1h EMA 55 (pullback, not reversal)

**Short Entry Conditions:**
- Mirror logic for downtrends.

**Exit Rules:**
- Stop Loss: Below the swing low of the pullback (the lowest low of the retracement) + 0.5 x ATR(14) buffer
- Take Profit: 2R
- Trailing Stop: After 1R, trail at 1.0 x ATR
- Time Exit: 4 hours max
- Stale Exit: 2 hours if < 0.8R

**Frequency:** Max 2 entries per symbol per day.

**Expected Performance:**
- Win rate: ~46-52%
- Average win: ~1.4-1.8R
- Higher hit rate, lower R-multiple vs Strategy 1

### Strategy 3: Range Compression Breakout (Volatility Squeeze)

**Purpose:** Trade the explosive move that follows a period of unusually low volatility (compression leads to expansion).

**Entry Conditions (ALL must be true):**
1. Bollinger Band width (20-period, 2 std) on 5m is at its lowest value in the last 120 candles (10 hours)
2. 5m candle CLOSES outside the upper or lower Bollinger Band
3. Volume on that candle >= 2.0x average (strong participation)
4. ADX on 1h is rising (directional movement beginning)
5. Direction aligned with 1h trend (EMA 21 vs EMA 55)

**Exit Rules:**
- Stop Loss: Opposite Bollinger Band at time of entry (the compressed range width is the risk)
- Take Profit: 3R (squeezes produce outsized moves)
- Trailing Stop: After 1.5R, trail at 1.5 x ATR
- Time Exit: 4 hours max
- Only ONE attempt per squeeze event — if stopped out, do not re-enter the same squeeze

**Frequency:** These setups are rare. Expect 2-5 per week across both symbols.

**Expected Performance:**
- Win rate: ~35-40%
- Average win: ~2.0-2.8R
- Low frequency but highest R-multiple

---

## 8. RISK MANAGEMENT

This is the most important section. Strategy can be mediocre. Risk management cannot.

### 8A. Position Sizing

```
Position Size = (Account Equity x Risk %) / Stop Distance

Where:
- Account Equity = current account balance (not including unrealized P&L)
- Risk % = per-trade risk (see scaling table below)
- Stop Distance = entry price - stop price (in price units)
```

Must account for:
- Exchange maker/taker fees: assume **0.05% per side** (0.10% round trip)
- Slippage: assume **0.08%** per entry (this is realistic for volatile breakout entries)
- Contract minimum order size
- Tick size precision

The position size calculation must subtract expected fees from the risk budget:
```
Effective Risk = Risk Budget - (Position Size x Fee Rate x 2) - (Position Size x Slippage)
```

### 8B. Risk Per Trade (Scaling Table)

Risk scales based on **trade count and proven performance**, not arbitrary account milestones.

| Phase | Condition | Risk Per Trade |
|---|---|---|
| Phase 0 | First 50 trades | 1.5% |
| Phase 1 | 50+ trades, profit factor > 1.3 | 2.0% |
| Phase 2 | 100+ trades, profit factor > 1.4, max DD < 15% | 2.5% |
| Phase 3 | 200+ trades, profit factor > 1.5, Sharpe > 1.0 | 3.0% (max ever) |

If at any phase the drawdown exceeds 15%, **drop back one phase** until recovery.

Never exceed 3% per trade regardless of account size or performance.

### 8C. Leverage

- Fixed at **3x** for Phase 0 and Phase 1
- May increase to **5x** at Phase 2 only if drawdown metrics support it
- Never auto-increased
- Never above 5x under any circumstances

### 8D. Daily Loss Cap

- **5% of account equity**
- When hit: all open positions closed at market, no new entries until next UTC 00:00
- This is enforced programmatically, not optionally

### 8E. Weekly Loss Cap

- **12% of account equity** (measured from Monday 00:00 UTC)
- When hit: no trading until next Monday 00:00 UTC
- Prevents catastrophic losing weeks from compounding

### 8F. Consecutive Loss Cooldown

- After **3 consecutive losing trades**: pause trading for **2 hours**
- After **5 consecutive losing trades in one day**: stop trading for the day
- This prevents chop from draining the account through rapid-fire losses

### 8G. Drawdown Reduction

- If account drawdown from peak exceeds **15%**: reduce risk per trade to **1%** until new equity high
- If drawdown exceeds **20%**: reduce risk to **0.5%** and send alert for manual review
- If drawdown exceeds **25%**: **halt all trading**, require manual restart

### 8H. Exposure Limits

| Limit | Value |
|---|---|
| Max concurrent positions | 3 |
| Max same-direction BTC+ETH | 1 (correlation rule) |
| Max total capital deployed | 40% of equity |
| Max total open risk | 6% of equity |

---

## 9. EXECUTION ENGINE

### 9A. Order Types

- **Entry**: Limit IOC (Immediate or Cancel) preferred
  - Set limit price at signal price + 0.02% buffer (for longs)
  - If not filled: skip trade. Do NOT chase with market orders.
  - Exception: if signal is extremely strong (volume > 3x avg), use market order
- **Stop Loss**: Stop-market order placed on the exchange immediately after fill confirmation
  - Never use a "soft" stop that depends on the bot being online
  - The exchange enforces the stop even if the bot crashes
- **Take Profit**: Limit order at TP level, placed after fill
- **Trailing Stop**: Managed by the bot — update stop-market order on exchange every new 5m candle when in trailing mode

### 9B. Fill Validation

After every entry order:
- Confirm fill price vs expected price
- If slippage > 0.10%: log warning, reduce next trade size by 20%
- If slippage > 0.20%: log alert, consider pausing for 30 minutes (execution environment degraded)

### 9C. Partial Fill Handling

- If order partially fills and remaining unfilled within 30 seconds: cancel remainder
- Manage the partial fill as a normal position with proportionally sized stop

### 9D. Rate Limiting

- Respect exchange API rate limits (Binance: 1200 req/min, Bybit: varies)
- Implement exponential backoff on 429 errors
- Never spam orders

---

## 10. KILL SWITCH (EMERGENCY SAFETY)

The kill switch overrides everything. When triggered, it closes all positions at market and halts the bot.

**Trigger Conditions:**

| Condition | Action |
|---|---|
| Websocket disconnect > 15 seconds | Close all positions, halt |
| Exchange API error 3 times in 60 seconds | Close all positions, halt |
| Spread > 5x normal average | Close all positions, halt |
| Single-candle unrealized P&L swing > 3% | Close all positions, halt |
| Account equity drops below 70% of starting capital | Close all positions, halt permanently |
| Slippage on stop execution > 0.5% | Log critical alert, reduce risk to minimum |
| Funding rate spikes > 0.3% | Close affected position |

After kill switch trigger: require **manual restart with acknowledgment**.

---

## 11. STATE MANAGEMENT AND CRASH RECOVERY

### 11A. State Persistence

Every state change must be written to SQLite before the action completes:
- Open positions (entry price, stop, TP, size, timestamp)
- Daily P&L running total
- Weekly P&L running total
- Trade count
- Consecutive loss counter
- Current phase and risk level
- Kill switch status

### 11B. Startup Reconciliation

On every bot startup:
1. Query all open positions from the exchange API
2. Compare with local database state
3. If orphaned position found on exchange (not in local DB):
   - Apply emergency stop at 2x ATR from current price
   - Log alert
   - Manage as normal position
4. If local DB shows position but exchange doesn't:
   - Mark as closed, investigate fill history
   - Log discrepancy
5. Verify daily/weekly loss counters against actual trade history
6. Only begin normal trading after reconciliation passes

### 11C. Heartbeat

- Bot sends heartbeat to a monitoring endpoint every 60 seconds
- If heartbeat missed for 3 minutes: external alert triggers (email/Telegram)
- This catches silent crashes

---

## 12. BACKTESTING REQUIREMENTS

### 12A. Data

- Minimum 2 years of 5m candle data for BTCUSDT and ETHUSDT
- Data must include: open, high, low, close, volume
- Source: exchange historical API or verified third-party provider
- Check for gaps, duplicates, and timezone consistency before use

### 12B. Realistic Cost Model

All backtests MUST include:
- Taker fee: 0.05% per side (0.10% round trip)
- Maker fee: 0.02% per side (when limit orders used)
- Slippage: **0.08% per entry** (breakout entries have worse fills)
- Slippage: **0.03% per exit** (stops and TPs)
- Funding rate: apply historical funding rates to positions held across 8h marks

If a backtest looks profitable without these costs but unprofitable with them, the strategy has no real edge. Discard it.

### 12C. Required Metrics

Every backtest must output:

| Metric | Minimum Acceptable |
|---|---|
| Total trades | > 300 |
| Profit Factor | > 1.4 |
| Sharpe Ratio (annualized) | > 1.0 |
| Max Drawdown | < 25% |
| Max Consecutive Losses | documented (expect 6-10) |
| Win Rate | > 35% |
| Average Win / Average Loss | > 1.5 |
| Expectancy per trade | > 0.15R |
| Recovery Factor (net profit / max DD) | > 2.0 |

### 12D. Walk-Forward Validation

- Split data: 70% in-sample, 30% out-of-sample
- Optimize parameters on in-sample ONLY
- Validate on out-of-sample WITHOUT any changes
- If out-of-sample profit factor drops below 1.2: strategy is likely overfit
- Perform rolling walk-forward: 6-month train, 2-month test, roll forward by 2 months

### 12E. Monte Carlo Simulation

After backtesting:
- Run 1,000 Monte Carlo simulations by randomly reordering trades
- Record the distribution of max drawdowns
- The 95th percentile max drawdown must be < 30%
- If it exceeds this, the system is too risky for $1,000 capital

### 12F. Regime Analysis

Break backtest results into:
- Strong trending periods (ADX > 30)
- Moderate trend (ADX 20-30)
- Choppy/ranging (ADX < 20)

The system should show:
- Positive expectancy in trending regimes
- Near-zero or slightly negative in choppy regimes (filters should prevent most chop trades)
- If the system makes most profit in chop: something is wrong, likely overfitted

---

## 13. PERFORMANCE LOGGING

### 13A. Per-Trade Log

Every trade must record:
- Entry timestamp, exit timestamp
- Symbol, direction (long/short)
- Entry price, exit price
- Stop loss price, take profit price
- Position size (contracts and USD)
- Strategy that generated the signal
- Realized P&L (USD and %)
- Fees paid
- Slippage (expected vs actual fill)
- Exit reason: stop loss / take profit / trailing stop / time exit / stale exit / kill switch / daily limit
- Holding duration
- R-multiple achieved
- Funding rate cost if applicable

### 13B. Daily Summary

- Total trades
- Win/loss count
- Daily P&L (USD and %)
- Running drawdown from peak
- Risk usage (% of daily limit consumed)
- Filter rejection counts (how many signals were blocked by each filter)

### 13C. Weekly/Monthly Summary

- Aggregate P&L
- Sharpe ratio (rolling 30 days)
- Profit factor (rolling 30 days)
- Max drawdown (rolling)
- Strategy breakdown (which strategy contributed what)
- Regime classification for the period

---

## 14. WHAT THIS BOT IS NOT

- NOT a martingale or grid bot
- NOT a market maker
- NOT a prediction AI or ML model
- NOT a signal copier or social trading tool
- NOT an arbitrage system
- NOT a high-frequency bot (it trades 0-4 times per day, not 100s)
- NOT designed for altcoins, memecoins, or low-liquidity tokens

It is a **structured volatility breakout system** that makes money by catching strong directional moves with asymmetric risk-reward, and survives by strict loss control and filtering.

---

## 15. EXPECTED REALITY

Be honest about what this system will do:

- **40-50% of months will be flat or slightly negative.** This is normal.
- **Losing streaks of 6-10 trades will happen.** The math allows for it.
- **A single strong trending week can make the entire month's return.** Most profit is concentrated.
- **Choppy markets will produce zero signals for days.** This is the filters working correctly.
- **Drawdowns of 10-15% are routine.** They are not bugs.
- **The bot will NOT trade every day.** Some days have zero valid setups.

Do NOT "fix" any of the above by loosening filters or adding more trades. The edge IS selectivity.

---

## 16. DEVELOPMENT PHASES

### Phase 1: Backtesting Engine (Week 1-2)
- Build candle data pipeline
- Implement all indicators
- Build Strategy 1 (Volatility Breakout) only
- Build all filters
- Build realistic cost model
- Run backtests, validate metrics
- Run Monte Carlo simulation
- Run walk-forward validation
- **Gate: Do not proceed unless profit factor > 1.4 and Monte Carlo 95th percentile DD < 30%**

### Phase 2: Paper Trading (Week 3-4)
- Connect to exchange websocket
- Build execution engine (paper mode — no real orders)
- Build state management and crash recovery
- Run Strategy 1 live on paper for 30 days minimum
- Compare paper results to backtest expectations
- **Gate: Paper results must be within 30% of backtest metrics**

### Phase 3: Micro Live (Week 5-8)
- Deploy with 10% of capital ($100)
- Risk per trade: 1.5% of deployed capital ($1.50)
- Run for minimum 50 trades
- Validate fill quality, slippage, fees vs assumptions
- **Gate: Live profit factor > 1.2 and no unexpected behavior**

### Phase 4: Scale to Full Capital
- Gradually increase deployed capital: $100 → $250 → $500 → $1,000
- Each step requires 20+ trades at the current level with positive expectancy
- Add Strategy 2 after 100 total live trades if Strategy 1 is validated
- Add Strategy 3 after 200 total live trades

**No skipping phases. No shortcuts.**

---

## 17. TECHNOLOGY STACK

| Layer | Tool | Why |
|---|---|---|
| Exchange API | CCXT | Unified interface, supports all major exchanges |
| Websocket | Native exchange WS + websockets lib | Low latency real-time data |
| Async Runtime | asyncio + uvloop | Non-blocking execution |
| Indicators | NumPy + TA-Lib (or pandas-ta) | Fast, reliable technical calculations |
| Data Handling | Polars (preferred) or Pandas | Candle manipulation and analysis |
| Backtesting | Vectorbt | Fast vectorized backtesting |
| Optimization | Optuna | Hyperparameter search (use sparingly) |
| Database | SQLite (Phase 1-3), PostgreSQL (Phase 4+) | State persistence |
| Configuration | YAML | Human-readable settings |
| Logging | structlog | Structured, parseable logs |
| Monitoring | Telegram bot API | Alerts and daily summaries |
| Deployment | Docker + docker-compose | Reproducible, restartable |

---

## 18. CONFIGURATION (settings.yaml structure)

```yaml
exchange:
  name: "binance"  # or "bybit"
  testnet: true     # MUST be true until Phase 3
  api_key: "${EXCHANGE_API_KEY}"
  api_secret: "${EXCHANGE_API_SECRET}"

trading:
  symbols: ["BTCUSDT", "ETHUSDT"]
  leverage: 3
  max_leverage: 5

risk:
  risk_per_trade_pct: 1.5          # Phase 0 default
  daily_loss_cap_pct: 5.0
  weekly_loss_cap_pct: 12.0
  max_concurrent_positions: 3
  max_same_direction_correlated: 1
  max_total_exposure_pct: 40.0
  max_total_open_risk_pct: 6.0
  consecutive_loss_cooldown_count: 3
  consecutive_loss_cooldown_hours: 2
  drawdown_reduction_threshold_pct: 15.0
  drawdown_halt_threshold_pct: 25.0

strategy:
  breakout:
    lookback_candles: 20
    volume_multiplier: 1.7
    atr_period: 14
    stop_atr_multiplier: 1.4
    take_profit_r: 2.5
    trailing_start_r: 1.0
    trailing_atr_multiplier: 1.2
    max_entries_per_symbol_per_day: 2

  pullback:
    ema_proximity_pct: 0.3
    volume_multiplier: 1.3
    stop_atr_buffer: 0.5
    take_profit_r: 2.0
    trailing_start_r: 1.0
    trailing_atr_multiplier: 1.0
    max_entries_per_symbol_per_day: 2

  squeeze:
    bb_period: 20
    bb_std: 2.0
    bb_width_lookback: 120
    volume_multiplier: 2.0
    take_profit_r: 3.0
    trailing_start_r: 1.5
    trailing_atr_multiplier: 1.5

filters:
  trend:
    ema_fast: 21
    ema_slow: 55
  chop:
    adx_threshold: 20
    atr_percentile_threshold: 30
    atr_percentile_window_days: 30
    consecutive_failure_pause_hours: 4
    consecutive_failure_count: 3
  spread:
    rolling_window_ticks: 100
    delay_multiplier: 2.0
    skip_multiplier: 3.0
  funding:
    reduce_threshold_pct: 0.05
    reduce_size_pct: 30
    skip_threshold_pct: 0.10
  session:
    blackout_periods_utc:
      - start: "00:00"
        end: "04:00"
        days: ["mon", "tue", "wed", "thu", "fri"]
      - start: "04:00"
        end: "06:00"
        days: ["sat", "sun"]
  event:
    pre_event_blackout_minutes: 30
    post_event_blackout_minutes: 15
    emergency_close_swing_pct: 3.0

execution:
  order_type: "limit_ioc"
  limit_buffer_pct: 0.02
  slippage_warning_pct: 0.10
  slippage_pause_pct: 0.20
  partial_fill_cancel_seconds: 30
  max_holding_hours: 4
  stale_position_hours: 2
  stale_position_min_r: 1.0

kill_switch:
  ws_disconnect_seconds: 15
  api_error_count: 3
  api_error_window_seconds: 60
  spread_multiplier: 5.0
  equity_floor_pct: 70.0
  funding_spike_pct: 0.30
  stop_slippage_alert_pct: 0.50

scaling:
  phases:
    - min_trades: 0
      max_trades: 50
      risk_pct: 1.5
      max_leverage: 3
    - min_trades: 50
      min_profit_factor: 1.3
      risk_pct: 2.0
      max_leverage: 3
    - min_trades: 100
      min_profit_factor: 1.4
      max_drawdown_pct: 15.0
      risk_pct: 2.5
      max_leverage: 5
    - min_trades: 200
      min_profit_factor: 1.5
      min_sharpe: 1.0
      risk_pct: 3.0
      max_leverage: 5

backtest:
  fee_taker_pct: 0.05
  fee_maker_pct: 0.02
  slippage_entry_pct: 0.08
  slippage_exit_pct: 0.03
  include_funding: true
  monte_carlo_runs: 1000
  monte_carlo_max_dd_95_pct: 30.0
  walk_forward_train_months: 6
  walk_forward_test_months: 2

logging:
  level: "INFO"
  trade_log: "logs/trades.jsonl"
  daily_log: "logs/daily.jsonl"
  system_log: "logs/system.log"

alerts:
  telegram:
    enabled: true
    bot_token: "${TELEGRAM_BOT_TOKEN}"
    chat_id: "${TELEGRAM_CHAT_ID}"
  alert_on:
    - trade_entry
    - trade_exit
    - daily_summary
    - kill_switch
    - drawdown_warning
    - phase_change
```

---

## 19. FINAL INSTRUCTIONS FOR DEVELOPMENT

1. **Build incrementally.** Start with the data pipeline and backtesting engine. Don't touch live execution until backtests pass all gates.

2. **Every number is configurable.** Hardcode nothing. All thresholds, multipliers, and limits come from settings.yaml.

3. **Test edge cases obsessively.** What happens when the exchange returns an error? What happens when websocket drops mid-trade? What happens when a candle has zero volume? What happens when the bot starts with an orphaned position?

4. **Log everything.** You cannot debug a trading bot without comprehensive logs. Every decision (trade taken, trade rejected, filter activated) must be logged with the reason.

5. **If any optimization increases short-term profit but increases drawdown instability: reject it.** Preserve capital first. Grow second.

6. **The bot should do nothing most of the time.** A day with zero trades is not a bug — it means the filters are working. Do not "fix" low trade frequency by loosening filters.

7. **Backtest results are the ceiling, not the floor.** Live performance will always be worse than backtest. Design for that gap.

8. **This is a marathon, not a sprint.** The first 100 trades are about proving the system works, not about making money.
