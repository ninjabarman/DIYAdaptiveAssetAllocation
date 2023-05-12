import numpy as np
import pandas as pd
import vectorbt as vbt  # version=0.23.0
from pyrb import RiskBudgeting
from vectorbt.portfolio.enums import SizeType, Direction
from vectorbt.portfolio.nb import order_nb, sort_call_seq_nb

from yahoo_data import DATA_DIR, ASSET_PRICES_FILE, get_tickers

budget = [0.8, 0.2, 0.2 ]


def pre_sim_func_nb(sc, n):
    sc.segment_mask[:, :] = False
    sc.segment_mask[n::n, :] = True
    return ()


def pre_segment_func_nb(sc, find_weights_nb, history_len):
    if history_len == -1:
        # Look back at the entire time period
        close = sc.close[:sc.i, sc.from_col:sc.to_col]
    else:
        # Look back at a fixed time period
        if sc.i - history_len <= 0:
            return (np.full(sc.group_len, np.nan),)  # insufficient data
        close = sc.close[sc.i - history_len:sc.i, sc.from_col:sc.to_col]

        # Find optimal weights
    weights = find_weights_nb(sc, close)

    # Update valuation price and reorder orders
    size_type = np.full(sc.group_len, SizeType.TargetPercent)
    direction = np.full(sc.group_len, Direction.LongOnly)
    temp_float_arr = np.empty(sc.group_len, dtype=np.float_)
    for k in range(sc.group_len):
        col = sc.from_col + k
        sc.last_val_price[col] = sc.close[sc.i, col]
    sort_call_seq_nb(sc, weights, size_type, direction, temp_float_arr)
    return (weights,)


def order_func_nb(oc, weights):
    col_i = oc.call_seq_now[oc.call_idx]
    return order_nb(
        weights[col_i],
        oc.close[oc.i, oc.col],
        size_type=SizeType.TargetPercent,
    )


def opt_weights(sc, close):
    close = pd.DataFrame(close, columns=get_tickers())
    returns = close.pct_change().dropna()
    cov = returns.cov()
    rb = RiskBudgeting(cov, budget)
    rb.solve()
    return rb.x


def plot_allocation(rb_pf):
    # Plot weights development of the portfolio
    rb_asset_value = rb_pf.asset_value(group_by=False)
    rb_value = rb_pf.value()
    rb_idxs = np.flatnonzero((rb_pf.asset_flow() != 0).any(axis=1))
    rb_dates = rb_pf.wrapper.index[rb_idxs]
    fig = (rb_asset_value.vbt / rb_value).vbt.plot(
        trace_names=get_tickers(),
        trace_kwargs=dict(
            stackgroup='one'
        )
    )
    for rb_date in rb_dates:
        fig.add_shape(
            dict(
                xref='x',
                yref='paper',
                x0=rb_date,
                x1=rb_date,
                y0=0,
                y1=1,
                line_color=fig.layout.template.layout.plot_bgcolor
            )
        )
    fig.write_image("portfolio.png")


def run_back_test():
    data = pd.read_csv(ASSET_PRICES_FILE, index_col="Date").dropna()
    portfolio = vbt.Portfolio.from_order_func(
        data,
        order_func_nb,
        pre_sim_func_nb=pre_sim_func_nb,
        pre_sim_args=(30,),
        pre_segment_func_nb=pre_segment_func_nb,
        pre_segment_args=(opt_weights, 90),
        cash_sharing=True,
        group_by=True,
        use_numba=False,
    )
    plot_allocation(portfolio)
    dsr = portfolio.returns().mean() / portfolio.returns().std()
    print(f"Sharpe Ratio {dsr * 255 ** .5}")

    return portfolio
