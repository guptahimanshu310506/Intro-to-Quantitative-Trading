import pandas as pd

class AutomatedMarketMaking:
    def __init__(self, tick_size=0.1, lot_size=2):
        self.tick_size  = tick_size
        self.lot_size   = lot_size
        self.reset_simulator()
#!/usr/bin/env python3
"""
Robust, risk-aware market-maker (ASCII only).
Outputs submission.csv for the first 3000 timestamps.
"""

import sys
from collections import deque
import numpy as np
import pandas as pd


class AdaptiveMarketMaking:
    """Core engine."""

    def __init__(
        self,
        tick_size=0.1,
        lot_size=2,
        max_inventory=20,
        ema_half_life=30,      # midprice volatility EMA half-life (rows)
        flow_window=40,        # recent trade imbalance
        k_vol=1.2,             # vol -> half spread
        base_half=0.15,        # static half-spread cushion
        k_inv=0.5,             # inventory in spread
        inv_skew=2,          # inventory skew multiplier
        flow_skew=2,         # flow skew multiplier
        book_skew=2,         # depth-imbalance skew multiplier
        cool_ticks=2,          # widen filled side
        cool_steps=3,          # for N rows
    ):
        self.tick = tick_size
        self.lot = lot_size
        self.max_inv = max_inventory

        self.vol_alpha = 2.0 / (ema_half_life + 1.0)
        self.flow_window = flow_window

        self.k_vol = k_vol
        self.base_half = base_half
        self.k_inv = k_inv
        self.inv_skew = inv_skew
        self.flow_skew = flow_skew
        self.book_skew = book_skew

        self.cool_ticks = cool_ticks
        self.cool_steps = cool_steps

        self.reset_simulator()

    # ---------- helpers --------------------------------------------------
    def _round_tick(self, p):
        return round(p / self.tick) * self.tick

    def _get_row(self, ob_df, ts):
        if ts in ob_df.index:
            return ob_df.loc[ts]
        return ob_df[ob_df["timestamp"] == ts].iloc[0]

    # ---------- state ----------------------------------------------------
    def reset_simulator(self):
        self.inventory = 0
        self.active_bid = None
        self.active_ask = None
        self.valid_from = None

        self.ema_mid = None
        self.ema_var = None

        self.flow_q = deque(maxlen=self.flow_window)

        self.cool_side = None
        self.cool_left = 0

        self.row_counter = 0

    # ---------- volatility update ---------------------------------------
    def _update_vol(self, mid):
        if self.ema_mid is None:
            self.ema_mid = mid
            self.ema_var = 0.0
            return
        diff = mid - self.ema_mid
        self.ema_mid += self.vol_alpha * diff
        self.ema_var = (1.0 - self.vol_alpha) * (self.ema_var + self.vol_alpha * diff * diff)

    # ---------- trade processing ----------------------------------------
    def process_trades(self, ts, trades_t):
        if self.valid_from is None or ts < self.valid_from:
            return self.inventory

        filled = False
        if self.active_bid is not None:
            sells = trades_t[trades_t.side == "sell"]
            if not sells.empty and sells.price.max() <= self.active_bid:
                self.inventory += self.lot
                self.active_bid = None
                self.cool_side = "bid"
                self.cool_left = self.cool_steps
                filled = True

        if self.active_ask is not None:
            buys = trades_t[trades_t.side == "buy"]
            if not buys.empty and buys.price.min() >= self.active_ask:
                self.inventory -= self.lot
                self.active_ask = None
                self.cool_side = "ask"
                self.cool_left = self.cool_steps
                filled = True

        if filled:
            self.valid_from = float("inf")
        return self.inventory

    # ---------- quote builder -------------------------------------------
    def _build_quote(self, row, inv, flow_imb):
        # top two levels for book imbalance
        bid1 = row.bid_1_price
        ask1 = row.ask_1_price
        bid1_sz = row.bid_1_size
        ask1_sz = row.ask_1_size

        bid2 = row.bid_2_price
        ask2 = row.ask_2_price
        bid2_sz = row.bid_2_size
        ask2_sz = row.ask_2_size

        depth_bid = bid1_sz + bid2_sz
        depth_ask = ask1_sz + ask2_sz
        if depth_bid + depth_ask > 0:
            book_imb = (depth_bid - depth_ask) / (depth_bid + depth_ask)
        else:
            book_imb = 0.0

        mid = (bid1 + ask1) / 2.0
        self._update_vol(mid)
        sigma = np.sqrt(self.ema_var) if self.ema_var is not None else 0.0

        half = max(self.tick,
                   self.base_half,
                   self.k_vol * sigma,
                   self.k_inv * abs(inv) * self.tick)

        # inventory + flow + book skew (ticks)
        skew_ticks = (
            -self.inv_skew * inv / self.max_inv
            + self.flow_skew * flow_imb
            + self.book_skew * book_imb
        )

        bid = mid - half + skew_ticks * self.tick
        ask = mid + half + skew_ticks * self.tick

        # cool-down widens only filled side
        if self.cool_left > 0:
            if self.cool_side == "bid":
                bid -= self.cool_ticks * self.tick
            else:
                ask += self.cool_ticks * self.tick

        # stay at least one tick from best on that side
        bid = min(bid, bid1 - self.tick)
        ask = max(ask, ask1 + self.tick)

        bid = self._round_tick(bid)
        ask = self._round_tick(ask)
        if bid >= ask:
            bid -= self.tick
            ask += self.tick
        return bid, ask

    # ---------- public strategy (checker calls this) --------------------
    def strategy(self, ob_df, tr_df, inventory, ts):
        row = self._get_row(ob_df, ts)
        self.row_counter += 1

        # update flow imbalance
        trades_now = tr_df[tr_df["timestamp"] == ts]
        if not trades_now.empty:
            sign = trades_now.side.map({"buy": 1, "sell": -1})
            self.flow_q.append(sign.mean())
        flow_imb = sum(self.flow_q) / len(self.flow_q) if self.flow_q else 0.0

        bid, ask = self._build_quote(row, inventory, flow_imb)

        # late session flattening
        if self.row_counter > 0.95 * 3000:
            if inventory > 0:
                bid = None
            elif inventory < 0:
                ask = None

        # hard inventory guards
        if inventory >= self.max_inv:
            bid = None
        if inventory <= -self.max_inv:
            ask = None

        # cool-down countdown
        if self.cool_left > 0:
            self.cool_left -= 1
            if self.cool_left == 0:
                self.cool_side = None

        return bid, ask

    # ---------- back-test helpers ---------------------------------------
    def update_quote(self, ts, bid, ask):
        self.active_bid = bid
        self.active_ask = ask
        self.valid_from = ts + 1

    def run(self, ob_df, tr_df):
        self.reset_simulator()

        ob_df = ob_df.head(3000).copy()
        tr_df = tr_df.head(3000).copy()

        ob_df.set_index("timestamp", inplace=True)
        tr_groups = tr_df.groupby("timestamp")

        quotes = []
        for ts in ob_df.index:
            trades_t = tr_groups.get_group(ts) if ts in tr_groups.groups else pd.DataFrame()
            inv = self.process_trades(ts, trades_t)

            bid, ask = self.strategy(ob_df, tr_df, inv, ts)
            self.update_quote(ts, bid, ask)

            quotes.append(
                {
                    "timestamp": ts,
                    "bid_price": bid if bid is not None else "",
                    "ask_price": ask if ask is not None else "",
                }
            )
        return pd.DataFrame(quotes)


# ---------- alias for checker -------------------------------------------
class AutomatedMarketMaking(AdaptiveMarketMaking):
    pass


# ---------- I / O --------------------------------------------------------
def get_paths():
    if len(sys.argv) >= 3:
        return sys.argv[1], sys.argv[2]
    return input().strip(), input().strip()


if __name__ == "__main__":
    ob_path, tr_path = get_paths()
    ob_df = pd.read_csv(ob_path)
    tr_df = pd.read_csv(tr_path)

    amm = AutomatedMarketMaking(tick_size=0.1, lot_size=2)
    submission = amm.run(ob_df, tr_df)
    submission.to_csv("submission.csv", index=False)

    def reset_simulator(self):
        self.inventory  = 0
        self.active_bid = None
        self.active_ask = None
        self.valid_from = None

    def update_quote(self, timestamp, bid_price, ask_price):
        # Post or update your quote at timestamp It takes effect at t+1
        self.active_bid = bid_price
        self.active_ask = ask_price
        self.valid_from = timestamp + 1

    def process_trades(self, timestamp, trades_at_t):
        # Process all public trades at timestamp Returns updated inventory
        if self.valid_from is None or timestamp < self.valid_from:
            return self.inventory

        filled = False

        # sellside fill against your bid
        sells = trades_at_t[trades_at_t.side == 'sell']
        if self.active_bid is not None and not sells.empty:
            if self.active_bid >= sells.price.max():
                self.inventory += self.lot_size
                self.active_bid = None
                filled = True

        # buyside fill against your ask
        buys = trades_at_t[trades_at_t.side == 'buy']
        if self.active_ask is not None and not buys.empty:
            if self.active_ask <= buys.price.min():
                self.inventory -= self.lot_size
                self.active_ask = None
                filled = True

        if filled:
            # deactivate until next update
            self.valid_from = float('inf')

        return self.inventory

    def strategy(self, ob_df, tr_df, inventory, t):
        #USER LOGIC override this in a subclass or monkey-patch it
        #Must return bid ask multiples of self tick size with bid less than ask.
        return None, None

    def run(self, ob_df, tr_df):
        self.reset_simulator()
        quotes = []

        all_ts = sorted(ob_df.timestamp.unique())
        for t in all_ts:
            trades_t = tr_df[tr_df.timestamp == t]
            inv      = self.process_trades(t, trades_t)

            bid, ask = self.strategy(ob_df, tr_df, inv, t)

            self.update_quote(t, bid, ask)

            quotes.append({
                'timestamp': t,
                'bid_price': bid,
                'ask_price': ask
            })

        return pd.DataFrame(quotes)

if __name__ == "__main__":
    ob_obj = pd.read_csv(input().strip())
    tr_obj = pd.read_csv(input().strip())
    
    #pick top 3k timestamps
    ob_obj = ob_obj.head(3000); 
    tr_obj = tr_obj.head(3000);

    amm = AutomatedMarketMaking(tick_size=0.1, lot_size=2)

    df_submission = amm.run(ob_obj, tr_obj)
    df_submission.to_csv('submission.csv', index=False)