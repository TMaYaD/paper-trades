import click
import requests
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
import json
from datetime import datetime

POOL_CACHE_FILE = "pool_cache.json"
RATE_LIMIT_DELAY = 1.5  # seconds between API calls

def _load_pool_cache():
    if os.path.exists(POOL_CACHE_FILE):
        with open(POOL_CACHE_FILE, 'r') as f:
            return json.load(f)
    return {}

def _save_pool_cache(cache):
    with open(POOL_CACHE_FILE, 'w') as f:
        json.dump(cache, f, indent=2)

def get_top_pool(token_address):
    """Fetches the most liquid DEX pool for a token.
    Returns cached result if available, otherwise fetches from API with rate limiting."""
    cache = _load_pool_cache()
    if token_address in cache:
        print(f"  Using cached pool for {token_address[:8]}...")
        return cache[token_address]

    time.sleep(RATE_LIMIT_DELAY)
    url = f"https://api.geckoterminal.com/api/v2/networks/solana/tokens/{token_address}/pools"
    res = requests.get(url).json()
    try:
        pool_address = res['data'][0]['attributes']['address']
        cache[token_address] = pool_address
        _save_pool_cache(cache)
        return pool_address
    except (KeyError, IndexError):
        raise RuntimeError(f"Could not locate an active liquidity pool for {token_address}")


def get_historical_closes(pool_address, start_date_str, token_address, limit=1000):
    """Fetches hourly OHLCV data, checking a local CSV cache first to minimize API calls."""
    cache_file = f"cache_{token_address}.csv"

    target_dt = pd.to_datetime(start_date_str)
    target_timestamp = int(target_dt.timestamp())

    # 1. Load existing cache
    if os.path.exists(cache_file):
        df_cache = pd.read_csv(cache_file)
        df_cache['timestamp'] = pd.to_datetime(df_cache['timestamp'])
        cache_oldest_ts = int(df_cache['timestamp'].min().timestamp())
        cache_newest_ts = int(df_cache['timestamp'].max().timestamp())
        all_dfs = [df_cache]
        print(f"  -> Loaded local cache: {len(df_cache)} records. (Oldest: {df_cache['timestamp'].min()})")
    else:
        cache_oldest_ts = float('inf')
        cache_newest_ts = float('-inf')
        all_dfs = []

    before_timestamp = None
    now_ts = int(time.time())

    # Skip fetching new data if cache already covers from target to within the last hour
    if cache_newest_ts >= now_ts - 3600 and cache_oldest_ts <= target_timestamp:
        print("  -> Cache is up to date. No API calls needed.")
        final_df = pd.concat(all_dfs).drop_duplicates(subset=['timestamp'])
        final_df = final_df.sort_values('timestamp').reset_index(drop=True)
        return final_df

    # If cache exists but has a gap to now, start fetching only the missing recent data
    if cache_newest_ts > 0 and cache_newest_ts < now_ts - 3600:
        print(f"  -> Cache newest: {pd.to_datetime(cache_newest_ts, unit='s')}. Fetching gap to now...")

    while True:
        url = f"https://api.geckoterminal.com/api/v2/networks/solana/pools/{pool_address}/ohlcv/hour?limit={limit}&token={token_address}&currency=usd"
        if before_timestamp is not None:
            url += f"&before_timestamp={before_timestamp - 1}"

        headers = {'User-Agent': 'Mozilla/5.0'}
        time.sleep(2.1)
        res = requests.get(url, headers=headers)

        if res.status_code == 429:
            print("[!] HTTP 429: Rate limit hit. Sleeping for 30 seconds...")
            time.sleep(30)
            continue

        res_json = res.json()

        try:
            ohlcv_list = res_json['data']['attributes']['ohlcv_list']
        except KeyError:
            raise RuntimeError(f"API Error for pool {pool_address}. Response: {res_json}")

        if not ohlcv_list:
            break

        df = pd.DataFrame(ohlcv_list, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        batch_oldest_ts = int(df['timestamp'].min())
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')

        all_dfs.append(df[['timestamp', 'close']])
        print(f"  -> Fetched {len(df)} hours down to {df['timestamp'].min()}")

        # --- Break & Skip Conditions ---

        # Condition 1: We successfully passed the user's target start date
        if batch_oldest_ts <= target_timestamp:
            break

        # Condition 2: Kill-switch for pool origin
        if before_timestamp is not None and batch_oldest_ts >= before_timestamp:
            break

        # Condition 3: Smart Cache Cursor Skip
        # If the batch we just downloaded overlaps with our cache's newest records...
        if batch_oldest_ts <= cache_newest_ts:
            # Check if the cache already covers the rest of the historical target
            if cache_oldest_ts <= target_timestamp:
                print("  -> Cache covers the remaining history. Stopping fetch.")
                break
            else:
                # The cache doesn't go back far enough. Skip the API cursor over the cached
                # segment and resume downloading from the absolute oldest cached record.
                print(f"  -> Skipping API cursor through cached data to {pd.to_datetime(cache_oldest_ts, unit='s')}")
                before_timestamp = cache_oldest_ts
                continue

        # Update the cursor normally
        before_timestamp = batch_oldest_ts

    if not all_dfs:
        raise ValueError("No historical data could be retrieved.")

    # 2. Merge, Sort, and Clean Duplicates
    final_df = pd.concat(all_dfs).drop_duplicates(subset=['timestamp'])
    final_df = final_df.sort_values('timestamp').reset_index(drop=True)

    # 3. Save the updated dataset back to the CSV cache
    final_df.to_csv(cache_file, index=False)

    return final_df


@click.command()
@click.option('--token-a', default="So11111111111111111111111111111111111111112", help='Mint address for Base Token (e.g., SOL).')
@click.option('--token-b', default="SKRbvo6Gf7GondiT3BbTfuRDPqLWei4j2Qy2NPGZhW3", help='Mint address for Quote Token (e.g., SKR).')
@click.option('--start-date', type=click.DateTime(formats=["%Y-%m-%d %H:%M:%S", "%Y-%m-%d"]), default="2026-01-21 09:00:00", help='Start datetime (YYYY-MM-DD HH:MM:SS)')
def main(token_a, token_b, start_date):
    print(f"--- Strategy Initialization ---")
    print(f"Base Token A: {token_a}")
    print(f"Quote Token B: {token_b}")
    print(f"Start Date:   {start_date}")

    # 1. Fetch Pools
    print("\nLocating optimal liquidity pools via GeckoTerminal...")
    pool_a = get_top_pool(token_a)
    pool_b = get_top_pool(token_b)

    # 2. Fetch Data
    print("Downloading historical hourly data...")
    df_a = get_historical_closes(pool_a, start_date, token_a).rename(columns={'close': 'price_a'})
    df_b = get_historical_closes(pool_b, start_date, token_b).rename(columns={'close': 'price_b'})

    # 3. Merge and Format
    df = pd.merge(df_a, df_b, on='timestamp', how='inner')
    df = df.sort_values('timestamp')
    df = df.set_index('timestamp')

    # 4. Filter by Start Date
    df = df.loc[start_date:]

    if df.empty:
        print("\n[!] Error: Dataset is empty after applying start-date filter.")
        print("    Remember: The free API only returns the last ~41 days of hourly data.")
        return

    # Resample to daily frequency
    df_daily = df.resample('D').first().dropna()

    if len(df_daily) < 2:
        print("\n[!] Error: Not enough daily data points to run a backtest.")
        return

    # 5. Backtest Execution — run both strategies in parallel
    print("Executing backtest simulation...")

    price_a_t0 = df_daily.iloc[0]['price_a']
    price_b_t0 = df_daily.iloc[0]['price_b']
    swap_fee = 0.005 # 0.5% assumed network/slippage penalty

    # Ungated strategy
    bal_a_ug = 1.0
    bal_b_ug = price_a_t0 / price_b_t0

    # Gated strategy
    bal_a_g = 1.0
    bal_b_g = price_a_t0 / price_b_t0

    # Baseline: 50-50 hold (1 token A + equivalent token B, no trading)
    hold_a = 1.0
    hold_b = price_a_t0 / price_b_t0

    init_value = bal_a_ug + (bal_b_ug * price_b_t0) / price_a_t0
    dates = [df_daily.index[0]]
    history_ungated = [init_value]
    history_gated = [init_value]
    history_baseline = [init_value]
    history_trigger = []
    history_pre_swap = []

    for i in range(1, len(df_daily)):
        price_a = df_daily['price_a'].iloc[i]
        price_b = df_daily['price_b'].iloc[i]

        prev_price_a = df_daily['price_a'].iloc[i-1]
        prev_price_b = df_daily['price_b'].iloc[i-1]

        # Calculate 24h Relative Movement
        ret_a = (price_a - prev_price_a) / prev_price_a
        ret_b = (price_b - prev_price_b) / prev_price_b
        trigger = ret_b - ret_a

        pre_swap_value = bal_a_g + (bal_b_g * price_b) / price_a
        history_pre_swap.append(pre_swap_value)
        history_trigger.append(trigger)

        # Ungated: always swap
        if ret_a > ret_b:
            sell = bal_a_ug / 2
            bal_a_ug -= sell
            bal_b_ug += (sell * price_a * (1 - swap_fee)) / price_b
        elif ret_b > ret_a:
            sell = bal_b_ug / 2
            bal_b_ug -= sell
            bal_a_ug += (sell * price_b * (1 - swap_fee)) / price_a

        # Gated: only swap if movement exceeds fee
        if abs(trigger) >= swap_fee:
            if ret_a > ret_b:
                sell = bal_a_g / 2
                bal_a_g -= sell
                bal_b_g += (sell * price_a * (1 - swap_fee)) / price_b
            elif ret_b > ret_a:
                sell = bal_b_g / 2
                bal_b_g -= sell
                bal_a_g += (sell * price_b * (1 - swap_fee)) / price_a

        dates.append(df_daily.index[i])
        history_ungated.append(bal_a_ug + (bal_b_ug * price_b) / price_a)
        history_gated.append(bal_a_g + (bal_b_g * price_b) / price_a)
        history_baseline.append(hold_a + (hold_b * price_b) / price_a)

    # Trade P&L: pre_swap[i] - pre_swap[i-1] = how yesterday's trade performed by today
    history_trade_pnl = [history_pre_swap[i] - history_pre_swap[i-1]
                         for i in range(1, len(history_pre_swap))]
    history_trigger_aligned = history_trigger[1:]

    # 6. Output Results
    # Normalize against baseline
    norm_ungated = [u / b for u, b in zip(history_ungated, history_baseline)]
    norm_gated = [g / b for g, b in zip(history_gated, history_baseline)]

    print("\n--- Final Results ---")
    print(f"Ungated Final Value: {history_ungated[-1]:.4f} (vs baseline: {norm_ungated[-1]:.4f})")
    print(f"Gated Final Value:   {history_gated[-1]:.4f} (vs baseline: {norm_gated[-1]:.4f})")
    print(f"Baseline (50-50 Hold): {history_baseline[-1]:.4f}")

    # Identify loss days: normalized gated value dropped vs previous day
    print("\n--- Loss Days (Gated strategy underperformed baseline day-over-day) ---")
    print(f"{'Date':<22} {'Trigger':>9} {'Gated':>9} {'Baseline':>9} {'Norm':>9} {'Prev Norm':>10} {'Delta':>9} {'Swapped':>8}")
    print("-" * 95)
    loss_count = 0
    for i in range(1, len(dates)):
        delta = norm_gated[i] - norm_gated[i-1]
        if delta < 0:
            loss_count += 1
            trigger = history_trigger[i-1] if i-1 < len(history_trigger) else 0
            swapped = "Yes" if abs(trigger) >= swap_fee else "No"
            print(f"{str(dates[i]):<22} {trigger:>+9.4f} {history_gated[i]:>9.4f} {history_baseline[i]:>9.4f} "
                  f"{norm_gated[i]:>9.4f} {norm_gated[i-1]:>10.4f} {delta:>+9.4f} {swapped:>8}")
    print(f"\nTotal loss days: {loss_count} / {len(dates)-1}")

    # 7. Plotting
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    fig.suptitle("Daily 50% Momentum Rebalancing Simulation", fontsize=14, fontweight='bold')

    axes[0].plot(dates, norm_ungated, label="Ungated (always swap)", color='blue', linewidth=2)
    axes[0].plot(dates, norm_gated, label=f"Gated (|trigger| >= {swap_fee})", color='purple', linewidth=2)
    axes[0].axhline(y=1.0, color='red', linestyle='--', label="Baseline (50-50 Hold)")
    axes[0].set_ylabel("Value vs Baseline")
    axes[0].set_xlabel("Date")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="upper left")

    # Subplot 2: Scatter — trigger vs next-day trade P&L
    colors = ['green' if pnl >= 0 else 'red' for pnl in history_trade_pnl]
    axes[1].scatter(history_trigger_aligned, history_trade_pnl, c=colors, alpha=0.6, edgecolors='black', linewidth=0.3)
    axes[1].axhline(y=0.0, color='black', linestyle='--', alpha=0.3)
    axes[1].axvline(x=0.0, color='black', linestyle='--', alpha=0.3)
    axes[1].set_xlabel("Trigger (ret_b - ret_a)")
    axes[1].set_ylabel("Prev-Day Trade P&L (in Token A)")
    axes[1].set_title("Trade Signal vs Prev-Day Outcome (pre-swap to pre-swap)")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()

