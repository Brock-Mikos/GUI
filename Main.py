import ttkbootstrap as tb
from ttkbootstrap.constants import *
import tkinter as tk
from tkinter import ttk
from scipy.stats import norm
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd
from matplotlib.figure import Figure
from PIL import Image, ImageTk
import os
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.figure import Figure



# --- Black-Scholes pricing function ---
def black_scholes_price(S, K, T, r, sigma, option_type='call'):
    try:
        if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
            return 0.0
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        if option_type == 'call':
            return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    except Exception:
        return 0.0

def calculate_greeks(S, K, T, r, sigma, option_type='call'):
    try:
        if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
            return (0, 0, 0, 0, 0)

        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        if option_type == 'call':
            delta = norm.cdf(d1)
            theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
                     - r * K * np.exp(-r * T) * norm.cdf(d2))
            rho = K * T * np.exp(-r * T) * norm.cdf(d2)
        else:
            delta = -norm.cdf(-d1)
            theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
                     + r * K * np.exp(-r * T) * norm.cdf(-d2))
            rho = -K * T * np.exp(-r * T) * norm.cdf(-d2)

        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        vega = S * norm.pdf(d1) * np.sqrt(T)

        return delta, gamma, theta / 365, vega / 100, rho / 100
    except Exception:
        return (0, 0, 0, 0, 0)


# --- Mispricing logic ---
def calculate_mispricing():
    try:
        S = float(entry_S.get())
        K = float(entry_K.get())
        T = float(entry_T.get()) / 365
        r = float(entry_r.get()) / 100
        sigma = float(entry_sigma.get()) / 100
        market_price = float(entry_market.get())
        option_type = option_type_var.get()

        model_price = black_scholes_price(S, K, T, r, sigma, option_type)
        mispricing = (market_price - model_price) / model_price * 100 if model_price != 0 else float('inf')
        delta, gamma, theta, vega, rho = calculate_greeks(S, K, T, r, sigma, option_type)

        result_var.set( f"Model Price: ${model_price:.2f}\n"
            f"Mispricing: {mispricing:.2f}%\n\n"
            f"Δ Delta: {delta:.4f}\n"
            f"Γ Gamma: {gamma:.4f}\n"
            f"Θ Theta: {theta:.4f}\n"
            f"V Vega: {vega:.4f}\n"
            f"ρ Rho: {rho:.4f}"
        )
    except ValueError:
        result_var.set("Invalid input")

def add_labeled_entry(parent, label_text):
    tb.Label(parent, text=label_text).pack(anchor=W, pady=5)
    entry = tb.Entry(parent)
    entry.pack(fill=X)
    return entry

# Load and update image
def update_image(strategy):
    image_path = f"spreads/{strategy.replace(' ', '_').lower()}.png"
    if os.path.exists(image_path):
        img = Image.open(image_path).resize((400, 250))
        photo = ImageTk.PhotoImage(img)
        image_label.config(image=photo)
        image_label.image = photo
    else:
        image_label.config(image="", text="No image available.")

# Description and image update
def update_description(*args):
    selected = spread_var.get()
    desc = spread_descriptions.get(selected, "Description coming soon...")
    description_text.delete("1.0", tk.END)
    description_text.insert(tk.END, desc)
    update_image(selected)


def labeled_entry(parent, label):
    tb.Label(parent, text=label).pack(anchor=W, pady=(10, 2))
    e = tb.Entry(parent)
    e.pack(fill=X)
    return e


# ---- Payoff Plotting Function ----
def plot_spread_payoff():
    try:
        K1 = float(entry_K1.get())
        K2 = float(entry_K2.get())
        P1 = float(entry_P1.get())
        P2 = float(entry_P2.get())
        spread_type = spread_type_var.get()

        S = np.linspace(min(K1, K2) - 20, max(K1, K2) + 20, 200)
        if spread_type == "Vertical Call Spread":
            payoff = np.maximum(S - K1, 0) - P1 - (np.maximum(S - K2, 0) - P2)
        elif spread_type == "Vertical Put Spread":
            payoff = (np.maximum(K2 - S, 0) - P2) - (np.maximum(K1 - S, 0) - P1)
        else:
            payoff = np.zeros_like(S)

        ax.clear()
        ax.plot(S, payoff, label="Payoff", color="cyan")
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        ax.set_title(f"{spread_type} Payoff")
        ax.set_xlabel("Stock Price at Expiration")
        ax.set_ylabel("Profit / Loss")
        ax.grid(True)
        ax.legend()
        canvas.draw()
    except ValueError:
        pass


def load_expirations():
    ticker = symbol_var.get().strip().upper()
    if not ticker:
        return
    try:
        stock = yf.Ticker(ticker)
        expirations = stock.options
        expiration_menu["values"] = expirations
        if expirations:
            expiration_var.set(expirations[0])
            load_chain()
    except Exception as e:
        print("Error loading expirations:", e)
        expiration_menu["values"] = []
        expiration_var.set("")



def load_chain():
    call_tree.delete(*call_tree.get_children())
    put_tree.delete(*put_tree.get_children())

    ticker = symbol_var.get().strip().upper()
    expiration = expiration_var.get()
    if not ticker or not expiration:
        return

    try:
        stock = yf.Ticker(ticker)
        chain = stock.option_chain(expiration)
        calls = chain.calls[["strike", "bid", "ask", "impliedVolatility", "volume"]]
        puts = chain.puts[["strike", "bid", "ask", "impliedVolatility", "volume"]]

        # Get current stock price and risk-free rate
        hist = stock.history(period="1d")
        if hist.empty:
            return
        current_price = hist["Close"][-1]
        r = 0.05  # Hardcoded risk-free rate
        T = (pd.to_datetime(expiration) - pd.Timestamp.today()).days / 365.0

        # Tag configuration for ATM rows
        call_tree.tag_configure("atm", background="#1f77b4")  # Blue
        put_tree.tag_configure("atm", background="#ff7f0e")   # Orange

        # Find ATM strike
        all_strikes = sorted(set(calls["strike"]).union(set(puts["strike"])))
        atm_strike = min(all_strikes, key=lambda x: abs(x - current_price))

        # Load Calls
        for _, row in calls.iterrows():
            K = row["strike"]
            sigma = row["impliedVolatility"]
            delta, gamma, theta, vega, rho = calculate_greeks(current_price, K, T, r, sigma, "call")
            tag = "atm" if abs(K - atm_strike) < 0.01 else ""
            call_tree.insert("", tk.END, values=(
                K, row["bid"], row["ask"],
                f"{sigma*100:.2f}%", row["volume"],
                f"{delta:.2f}", f"{gamma:.2f}", f"{theta:.2f}", f"{vega:.2f}", f"{rho:.2f}"
            ), tags=(tag,))

        # Load Puts
        for _, row in puts.iterrows():
            K = row["strike"]
            sigma = row["impliedVolatility"]
            delta, gamma, theta, vega, rho = calculate_greeks(current_price, K, T, r, sigma, "put")
            tag = "atm" if abs(K - atm_strike) < 0.01 else ""
            put_tree.insert("", tk.END, values=(
                K, row["bid"], row["ask"],
                f"{sigma*100:.2f}%", row["volume"],
                f"{delta:.2f}", f"{gamma:.2f}", f"{theta:.2f}", f"{vega:.2f}", f"{rho:.2f}"
            ), tags=(tag,))
    except Exception as e:
        print("Error loading chain:", e)

def update_expiration_checkboxes():
    for widget in expiration_filter_frame.winfo_children():
        widget.destroy()
    for exp in sorted(temp_loaded_option_data.keys()):
        var = tk.BooleanVar(value=True)
        cb = tb.Checkbutton(expiration_filter_frame, text=exp, variable=var)
        cb.pack(side=LEFT, padx=5)
        expiration_vars[exp] = var

# Plotting Function: Skew

def plot_volatility_skew():
    ax_skew.clear()
    df = get_chain_dataframe(include_expiration=True)
    if df is None or df.empty:
        return

    selected_type = option_type_skew.get()
    if selected_type != "both":
        df = df[df["type"] == selected_type]

    active_exps = [exp for exp, var in expiration_vars.items() if var.get()]
    df = df[df['expiration'].isin(active_exps)]

    if df.empty:
        return

    palette = sns.color_palette("hls", len(active_exps))
    for i, exp in enumerate(active_exps):
        exp_df = df[df['expiration'] == exp]
        ax_skew.plot(exp_df['strike'], exp_df['impliedVolatility'] * 100, label=exp, marker='o', linestyle='-')

    ax_skew.set_title(f"Volatility Skew ({selected_type.capitalize()})")
    ax_skew.set_xlabel("Strike")
    ax_skew.set_ylabel("IV (%)")
    ax_skew.grid(True)
    ax_skew.legend()
    skew_canvas.draw()

# Plotting Function: Term Structure

def plot_term_structure():
    ax_skew.clear()
    df = get_chain_dataframe(include_expiration=True)
    if df is None or df.empty:
        return

    selected_type = option_type_skew.get()
    if selected_type != "both":
        df = df[df["type"] == selected_type]

    active_exps = [exp for exp, var in expiration_vars.items() if var.get()]
    df = df[df['expiration'].isin(active_exps)]

    if df.empty:
        return

    # Compute average IV per expiration (filter to near ATM if needed)
    term_df = df.groupby('expiration')['impliedVolatility'].mean().reset_index()
    term_df['expiration_date'] = pd.to_datetime(term_df['expiration'])
    term_df = term_df.sort_values('expiration_date')

    ax_skew.plot(term_df['expiration_date'], term_df['impliedVolatility'] * 100, marker='o', linestyle='-')
    ax_skew.set_title("Term Structure of Implied Volatility")
    ax_skew.set_xlabel("Expiration Date")
    ax_skew.set_ylabel("Average IV (%)")
    ax_skew.grid(True)
    skew_canvas.draw()

# Helper to convert Treeview to DataFrame

def get_chain_dataframe(include_expiration=False):
    try:
        dfs = []
        for exp, df in temp_loaded_option_data.items():
            df = df.copy()
            if include_expiration:
                df['expiration'] = exp
            dfs.append(df)

        if not dfs:
            return None

        full_df = pd.concat(dfs, ignore_index=True)
        full_df["strike"] = pd.to_numeric(full_df["strike"], errors='coerce')
        full_df["iv"] = full_df["iv"].astype(str)
        full_df["impliedVolatility"] = pd.to_numeric(full_df["iv"].str.replace('%', ''), errors='coerce') / 100

        return full_df.dropna(subset=["strike", "impliedVolatility"])
    except Exception as e:
        print("Error building skew DataFrame:", e)
        return None

# REPLACEMENT load_chain

def load_chain():
    call_tree.delete(*call_tree.get_children())
    put_tree.delete(*put_tree.get_children())

    ticker = symbol_var.get().strip().upper()
    expiration = expiration_var.get()
    if not ticker or not expiration:
        return

    try:
        stock = yf.Ticker(ticker)
        chain = stock.option_chain(expiration)
        calls = chain.calls[["strike", "bid", "ask", "impliedVolatility", "volume"]]
        puts = chain.puts[["strike", "bid", "ask", "impliedVolatility", "volume"]]

        hist = stock.history(period="1d")
        if hist.empty:
            return
        current_price = hist["Close"][-1]
        r = 0.05
        T = (pd.to_datetime(expiration) - pd.Timestamp.today()).days / 365.0

        call_tree.tag_configure("atm", background="#1f77b4")
        put_tree.tag_configure("atm", background="#ff7f0e")
        all_strikes = sorted(set(calls["strike"]).union(set(puts["strike"])))
        atm_strike = min(all_strikes, key=lambda x: abs(x - current_price))

        for _, row in calls.iterrows():
            K = row["strike"]
            sigma = row["impliedVolatility"]
            delta, gamma, theta, vega, rho = calculate_greeks(current_price, K, T, r, sigma, "call")
            tag = "atm" if abs(K - atm_strike) < 0.01 else ""
            call_tree.insert("", tk.END, values=(
                K, row["bid"], row["ask"],
                f"{sigma*100:.2f}%", row["volume"],
                f"{delta:.2f}", f"{gamma:.2f}", f"{theta:.2f}", f"{vega:.2f}", f"{rho:.2f}"
            ), tags=(tag,))

        for _, row in puts.iterrows():
            K = row["strike"]
            sigma = row["impliedVolatility"]
            delta, gamma, theta, vega, rho = calculate_greeks(current_price, K, T, r, sigma, "put")
            tag = "atm" if abs(K - atm_strike) < 0.01 else ""
            put_tree.insert("", tk.END, values=(
                K, row["bid"], row["ask"],
                f"{sigma*100:.2f}%", row["volume"],
                f"{delta:.2f}", f"{gamma:.2f}", f"{theta:.2f}", f"{vega:.2f}", f"{rho:.2f}"
            ), tags=(tag,))

        # Store data for skew tab
        calls = calls.copy()
        puts = puts.copy()
        calls["type"] = "call"
        puts["type"] = "put"
        df = pd.concat([calls, puts], ignore_index=True)
        df.rename(columns={"impliedVolatility": "iv"}, inplace=True)
        temp_loaded_option_data[expiration] = df
        update_expiration_checkboxes()

    except Exception as e:
        print("Error loading chain:", e)

# Plot Function: Put/Call IV Ratio

def plot_put_call_iv_ratio():
    ax_ratio.clear()
    df = get_chain_dataframe(include_expiration=True)
    if df is None or df.empty:
        return

    active_exps = [exp for exp, var in expiration_vars.items() if var.get()]
    df = df[df['expiration'].isin(active_exps)]

    if df.empty:
        return

    # Filter extreme or invalid IVs
    df = df[(df['impliedVolatility'] > 0) & (df['impliedVolatility'] < 5)]

    # Group by strike and expiration
    calls = df[df['type'] == 'call'][['strike', 'expiration', 'impliedVolatility']].rename(columns={'impliedVolatility': 'call_iv'})
    puts = df[df['type'] == 'put'][['strike', 'expiration', 'impliedVolatility']].rename(columns={'impliedVolatility': 'put_iv'})

    merged = pd.merge(calls, puts, on=['strike', 'expiration'])
    merged = merged[(merged['call_iv'] > 0) & (merged['put_iv'] > 0)]
    merged['iv_ratio'] = merged['put_iv'] / merged['call_iv']

    # Remove outliers (e.g. ratio > 5x)
    merged = merged[np.isfinite(merged['iv_ratio']) & (merged['iv_ratio'] < 5)]

    for exp in sorted(merged['expiration'].unique()):
        exp_df = merged[merged['expiration'] == exp]
        ax_ratio.plot(exp_df['strike'], exp_df['iv_ratio'], label=exp, marker='o')

    ax_ratio.set_title("Put/Call IV Ratio by Strike")
    ax_ratio.set_xlabel("Strike")
    ax_ratio.set_ylabel("Put IV / Call IV")
    ax_ratio.grid(True)
    ax_ratio.legend()
    ratio_canvas.draw()
# Button to open expiration selector
def open_expiration_selector():
    popup = tk.Toplevel()
    popup.title("Select Expirations")
    popup.geometry("250x300")
    popup.grab_set()

    # Scrollable checkbox area
    canvas = tk.Canvas(popup)
    frame = ttk.Frame(canvas)
    scrollbar = ttk.Scrollbar(popup, orient="vertical", command=canvas.yview)
    canvas.configure(yscrollcommand=scrollbar.set)

    scrollbar.pack(side="right", fill="y")
    canvas.pack(side="left", fill="both", expand=True)
    canvas.create_window((0, 0), window=frame, anchor="nw")

    def on_frame_configure(event):
        canvas.configure(scrollregion=canvas.bbox("all"))

    frame.bind("<Configure>", on_frame_configure)

    # Build checkboxes
    for exp in sorted(temp_loaded_option_data.keys()):
        var = tk.BooleanVar(value=(exp in selected_iv_expirations))
        cb = ttk.Checkbutton(frame, text=exp, variable=var)
        cb.pack(anchor="w", padx=10)
        iv_checkbox_vars[exp] = var

    def apply_selection():
        selected_iv_expirations.clear()
        for exp, var in iv_checkbox_vars.items():
            if var.get():
                selected_iv_expirations.add(exp)
        popup.destroy()

    ttk.Button(popup, text="Apply", command=apply_selection).pack(pady=10)

# Load expirations
def load_builder_expirations():
    try:
        symbol = builder_symbol.get().strip().upper()
        stock = yf.Ticker(symbol)
        exps = stock.options
        builder_exp_menu["values"] = exps
        builder_leg_exp_menu["values"] = exps
        edit_exp_menu["values"] = exps
        if exps:
            builder_leg_expiration_var.set(exps[0])
            builder_expiration_var.set(exps[0])
            load_builder_strikes()
    except Exception as e:
        print("Error loading expirations:", e)

def load_builder_strikes():
    symbol = builder_symbol.get().strip().upper()
    expiration = builder_expiration_var.get()
    if not symbol or not expiration:
        return
    try:
        stock = yf.Ticker(symbol)
        chain = stock.option_chain(expiration)
        strikes = sorted(set(chain.calls["strike"]).union(set(chain.puts["strike"])))
        builder_strike_list.clear()
        builder_strike_list.extend(strikes)
        builder_strike_dropdown["values"] = strikes
        if strikes:
            builder_strike_var.set(strikes[len(strikes) // 2])
    except Exception as e:
        print("Error loading strikes:", e)

# Add Leg
def add_leg():
    symbol = builder_symbol.get().strip().upper()
    expiration = builder_leg_expiration_var.get()
    if not symbol or not expiration:
        return

    strike = float(builder_strike_var.get())
    option_type = option_type_var.get()
    side = side_var.get()
    qty = int(qty_var.get())

    try:
        stock = yf.Ticker(symbol)
        chain = stock.option_chain(expiration)
        opt_df = chain.calls if option_type == "call" else chain.puts
        row = opt_df[opt_df["strike"] == strike]
        if row.empty:
            return
        bid = row["bid"].values[0]
        ask = row["ask"].values[0]
        price = (bid + ask) / 2

        leg_tree.insert("", tk.END, values=(option_type, side, strike, qty, f"{price:.2f}", expiration))
        strategy_data.append((option_type, side, strike, qty, price, expiration))
        plot_payoff()

    except Exception as e:
        print("Error adding leg:", e)


# Remove leg
def remove_leg():
    selected = leg_tree.selection()
    for item in selected:
        index = leg_tree.index(item)
        leg_tree.delete(item)
        del strategy_data[index]
    plot_payoff()

# Reset strategy
def reset_strategy():
    for item in leg_tree.get_children():
        leg_tree.delete(item)
    strategy_data.clear()
    plot_payoff()

last_sim_marker = None

def plot_payoff(mark_price=None, hud_data=None):
    from datetime import datetime
    ax_builder.clear()
    fig_builder.set_facecolor("#121212")
    ax_builder.set_facecolor("#1a1a1a")

    if not strategy_data:
        builder_canvas.draw()
        metrics_var.set("")
        return

    try:
        current_price = yf.Ticker(builder_symbol.get().strip().upper()).history(period="1d")["Close"][-1]
    except:
        current_price = 100

    prices = np.linspace(current_price * 0.5, current_price * 1.5, 300)
    payoff = np.zeros_like(prices)

    # Get all expirations involved
    expirations = sorted(set(exp for *_, exp in strategy_data), key=lambda d: pd.to_datetime(d))
    if len(expirations) == 1:
        short_exp, long_exp = expirations[0], expirations[0]
    else:
        short_exp, long_exp = expirations[0], expirations[-1]

    T1 = (pd.to_datetime(short_exp) - pd.Timestamp.today()).days / 365
    T2 = (pd.to_datetime(long_exp) - pd.Timestamp.today()).days / 365
    T_remain = max(T2 - T1, 1 / 365)

    r = 0.05  # Risk-free rate
    iv = 0.3  # Use a fixed 30% vol or replace with real IV

    total_cost = 0

    for price_index, S in enumerate(prices):
        pnl = 0
        phase1 = 0
        phase2 = 0

        for leg in strategy_data:
            opt_type, side, strike, qty, cost, exp = leg
            qty = int(qty)
            sign = 1 if side == "buy" else -1

            if exp == short_exp:
                # Phase 1 payoff (expires at T1)
                intrinsic = max(S - strike, 0) if opt_type == "call" else max(strike - S, 0)
                phase1 += sign * qty * (intrinsic - cost) * 100
            else:
                # Survives to Phase 2: we value it at T2 using Black-Scholes
                opt_val = black_scholes_price(S, strike, T_remain, r, iv, opt_type)
                phase2 += sign * qty * (opt_val - cost) * 100

        payoff[price_index] = phase1 + phase2

    # Plot line and fill
    ax_builder.plot(prices, payoff, color="#00ffff", linewidth=2.2, label="Strategy Payoff", zorder=3)
    ax_builder.fill_between(prices, payoff, where=payoff > 0, color="#00ff00", alpha=0.25, zorder=1)
    ax_builder.fill_between(prices, payoff, where=payoff < 0, color="#ff4444", alpha=0.25, zorder=1)

    # Breakeven detection
    breakevens = []
    for i in range(1, len(payoff)):
        if (payoff[i - 1] < 0 and payoff[i] >= 0) or (payoff[i - 1] > 0 and payoff[i] <= 0):
            breakevens.append(prices[i])
            ax_builder.axvline(prices[i], color="#ffff00", linestyle="--", alpha=0.8, linewidth=1)

    max_profit = np.max(payoff)
    max_loss = np.min(payoff)
    risk_reward = abs(max_profit / max_loss) if max_loss != 0 else float("inf")

    # Chart style
    ax_builder.spines['top'].set_visible(False)
    ax_builder.spines['right'].set_visible(False)
    ax_builder.tick_params(colors='white')
    ax_builder.xaxis.label.set_color('white')
    ax_builder.yaxis.label.set_color('white')
    ax_builder.title.set_color('white')

    ax_builder.axhline(0, color="white", linewidth=1, linestyle="--", alpha=0.5)
    ax_builder.set_title("Payoff at Expiration", fontsize=13, weight="bold")
    ax_builder.set_xlabel("Underlying Price")
    ax_builder.set_ylabel("Profit / Loss ($)")
    ax_builder.grid(True, color="#333333", linestyle="--", linewidth=0.5)
    ax_builder.set_axisbelow(True)

    if sim_slider["from"] != prices[0] or sim_slider["to"] != prices[-1]:
        sim_slider.config(from_=prices[0], to=prices[-1])

    # Simulated price marker
    if mark_price:
        ax_builder.axvline(mark_price, color="#ff66ff", linestyle=":", linewidth=2, label="Sim Price", zorder=2)

    if hud_data:
        box_text = "\n".join([
            f"📍 {hud_data['Sim Price']}",
            f"💰 {hud_data['PnL']}",
            f"📈 {hud_data['Return']}",
            f"🎯 {hud_data['Type']}"
        ])
        ax_builder.text(
            0.02, 0.98, box_text,
            transform=ax_builder.transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=dict(facecolor="#222222", edgecolor="white", boxstyle="round,pad=0.5", alpha=0.9),
            color="white",
            zorder=10
        )

    ax_builder.legend(facecolor="#1a1a1a", edgecolor="gray", labelcolor="white")
    builder_canvas.draw()

    metrics_var.set(
        f"💵 Net Debit/Credit based on leg prices   "
        f"📈 Max Profit: ${max_profit:,.2f}   "
        f"⚠️ Max Loss: ${max_loss:,.2f}   "
        f"📍 Breakeven(s): {', '.join(f'{b:.2f}' for b in breakevens)}   "
        f"R:R ≈ {risk_reward:.2f}" if risk_reward != float("inf") else ""
    )


def update_sim_price():
    if not strategy_data or sim_price_var.get() == 0:
        return

    price = sim_price_var.get()
    pnl = 0
    total_cost = 0

    for leg in strategy_data:
        opt_type, side, strike, qty, cost = leg
        qty = int(qty)
        sign = 1 if side == "buy" else -1
        total_cost += sign * qty * cost * 100

        if opt_type == "call":
            payoff = max(price - strike, 0) - cost
        else:
            payoff = max(strike - price, 0) - cost

        pnl += payoff * qty * sign * 100

    # % return
    if total_cost != 0:
        ret = (pnl / abs(total_cost)) * 100
    else:
        ret = 0

    sim_price_label.set(f"PnL: ${pnl:,.2f}")

    plot_payoff(mark_price=price, hud_data={
        "Sim Price": f"${price:,.2f}",
        "PnL": f"{'+' if pnl >= 0 else ''}${pnl:,.2f}",
        "Return": f"{ret:+.1f}%",
        "Type": f"{'Net Debit' if total_cost > 0 else 'Net Credit'}"
    })

def build_predefined_strategy(strategy_name):
    symbol = builder_symbol.get().strip().upper()
    if not symbol:
        return

    try:
        stock = yf.Ticker(symbol)
        exps = stock.options
        if len(exps) < 1:
            return

        expiration_primary = exps[0]
        expiration_secondary = exps[1] if len(exps) > 1 else exps[0]

        hist = stock.history(period="1d")
        if hist.empty:
            return
        spot = hist["Close"][-1]

        chain = stock.option_chain(expiration_primary)
        strikes = sorted(set(chain.calls["strike"]).union(set(chain.puts["strike"])))
        strikes = [k for k in strikes if k > 0]

        atm = min(strikes, key=lambda x: abs(x - spot))
        step = min(np.diff(sorted(strikes))) if len(strikes) > 1 else 5
        K = atm
        Kp = round(K + step, 2)
        Kpp = round(K + 2 * step, 2)
        Km = round(K - step, 2)
        Kmm = round(K - 2 * step, 2)

        reset_strategy()

        def add_leg_auto(opt_type, side, strike, expiration, qty=1):
            try:
                chain_exp = stock.option_chain(expiration)
                df = chain_exp.calls if opt_type == "call" else chain_exp.puts
                row = df[df["strike"] == strike]
                if row.empty:
                    return
                bid = row["bid"].values[0]
                ask = row["ask"].values[0]
                price = (bid + ask) / 2
                leg_tree.insert("", tk.END, values=(opt_type, side, strike, qty, f"{price:.2f}", expiration))
                strategy_data.append((opt_type, side, strike, qty, price, expiration))
            except Exception as e:
                print(f"Error adding leg: {e}")

        # Define all supported strategies
        strategy_map = {
            "Vertical Call Spread": lambda: [
                ("call", "buy", K, expiration_primary),
                ("call", "sell", Kp, expiration_primary)
            ],
            "Vertical Put Spread": lambda: [
                ("put", "buy", Kp, expiration_primary),
                ("put", "sell", K, expiration_primary)
            ],
            "Calendar Spread": lambda: [
                ("call", "buy", K, expiration_secondary),
                ("call", "sell", K, expiration_primary)
            ],
            "Straddle": lambda: [
                ("call", "buy", K, expiration_primary),
                ("put", "buy", K, expiration_primary)
            ],
            "Strangle": lambda: [
                ("call", "buy", Kp, expiration_primary),
                ("put", "buy", Km, expiration_primary)
            ],
            "Iron Condor": lambda: [
                ("call", "sell", Kp, expiration_primary),
                ("call", "buy", Kpp, expiration_primary),
                ("put", "sell", Km, expiration_primary),
                ("put", "buy", Kmm, expiration_primary)
            ],
            "Iron Butterfly": lambda: [
                ("call", "sell", K, expiration_primary),
                ("put", "sell", K, expiration_primary),
                ("call", "buy", Kp, expiration_primary),
                ("put", "buy", Km, expiration_primary)
            ],
            "Ratio Spread": lambda: [
                ("call", "buy", K, expiration_primary),
                ("call", "sell", Kp, expiration_primary, 2)
            ],
            "Covered Call": lambda: [
                ("call", "sell", Kp, expiration_primary)
            ],
            "Protective Put": lambda: [
                ("put", "buy", Km, expiration_primary)
            ],
            "Diagonal Spread": lambda: [
                ("call", "buy", K, expiration_secondary),
                ("call", "sell", Kp, expiration_primary)
            ],
            "Butterfly Spread": lambda: [
                ("call", "buy", Km, expiration_primary),
                ("call", "sell", K, expiration_primary, 2),
                ("call", "buy", Kp, expiration_primary)
            ],
            "Box Spread": lambda: [
                ("call", "buy", K, expiration_primary),
                ("call", "sell", Kp, expiration_primary),
                ("put", "sell", K, expiration_primary),
                ("put", "buy", Kp, expiration_primary)
            ],
            "Collar": lambda: [
                ("put", "buy", Km, expiration_primary),
                ("call", "sell", Kp, expiration_primary)
            ],
            "Synthetic Long Stock": lambda: [
                ("call", "buy", K, expiration_primary),
                ("put", "sell", K, expiration_primary)
            ],
            "Reverse Iron Condor": lambda: [
                ("call", "buy", Kp, expiration_primary),
                ("call", "sell", Kpp, expiration_primary),
                ("put", "buy", Km, expiration_primary),
                ("put", "sell", Kmm, expiration_primary)
            ],
            "Jade Lizard": lambda: [
                ("call", "sell", Kp, expiration_primary),
                ("call", "buy", Kpp, expiration_primary),
                ("put", "sell", Km, expiration_primary)
            ],
            "Reverse Jade Lizard": lambda: [
                ("put", "sell", Km, expiration_primary),
                ("put", "buy", Kmm, expiration_primary),
                ("call", "sell", Kp, expiration_primary)
            ],
            "Call Backspread": lambda: [
                ("call", "sell", K, expiration_primary),
                ("call", "buy", Kp, expiration_primary, 2)
            ],
            "Put Backspread": lambda: [
                ("put", "sell", K, expiration_primary),
                ("put", "buy", Km, expiration_primary, 2)
            ],
            "Broken Wing Butterfly": lambda: [
                ("call", "buy", Km, expiration_primary),
                ("call", "sell", K, expiration_primary, 2),
                ("call", "buy", Kpp, expiration_primary)
            ],
            "Unbalanced Condor": lambda: [
                ("call", "sell", Kp, expiration_primary, 2),
                ("call", "buy", Kpp, expiration_primary),
                ("put", "sell", Km, expiration_primary),
                ("put", "buy", Kmm, expiration_primary, 2)
            ],
            "Double Diagonal": lambda: [
                ("call", "buy", K, expiration_secondary),
                ("call", "sell", Kp, expiration_primary),
                ("put", "buy", K, expiration_secondary),
                ("put", "sell", Km, expiration_primary)
            ],
            "Christmas Tree Spread": lambda: [
                ("call", "buy", Km, expiration_primary),
                ("call", "sell", K, expiration_primary, 2),
                ("call", "buy", Kp, expiration_primary)
            ],
            "Lizard Spread": lambda: [
                ("call", "sell", Kp, expiration_primary),
                ("call", "buy", Kpp, expiration_primary),
                ("put", "sell", Km, expiration_primary)
            ]
        }

        if strategy_name in strategy_map:
            for leg in strategy_map[strategy_name]():
                add_leg_auto(*leg)

        plot_payoff()

    except Exception as e:
        print(f"Error generating strategy: {e}")

def load_leg_for_edit(event=None):
    refresh_edit_strike_choices(edit_exp_var.get())
    selected = leg_tree.selection()
    if not selected:
        return
    idx = leg_tree.index(selected[0])
    leg = strategy_data[idx]
    edit_strike_var.set(leg[2])  # strike
    edit_exp_var.set(leg[5])     # expiration
    selected_leg_index.set(idx)

def update_leg():
    idx = selected_leg_index.get()
    if idx == -1 or idx >= len(strategy_data):
        return

    symbol = builder_symbol.get().strip().upper()
    new_strike = float(edit_strike_var.get())
    new_exp = edit_exp_var.get()

    opt_type, side, _, qty, _, _ = strategy_data[idx]
    try:
        stock = yf.Ticker(symbol)
        chain = stock.option_chain(new_exp)
        df = chain.calls if opt_type == "call" else chain.puts
        row = df[df["strike"] == new_strike]
        if row.empty:
            return
        bid = row["bid"].values[0]
        ask = row["ask"].values[0]
        price = (bid + ask) / 2

        # Update strategy_data
        strategy_data[idx] = (opt_type, side, new_strike, qty, price, new_exp)

        # Update treeview
        leg_tree.item(leg_tree.get_children()[idx], values=(
            opt_type, side, new_strike, qty, f"{price:.2f}", new_exp
        ))

        plot_payoff()
    except Exception as e:
        print("Update error:", e)

def refresh_edit_strike_choices(exp):
    symbol = builder_symbol.get().strip().upper()
    if not symbol or not exp:
        return
    try:
        chain = yf.Ticker(symbol).option_chain(exp)
        strikes = sorted(set(chain.calls["strike"]).union(set(chain.puts["strike"])))
        edit_strike_menu["values"] = strikes
    except Exception as e:
        print("Error loading strikes for edit:", e)


def init_app():
    return

# --- GUI Setup ---
app = tb.Window(themename="darkly")
app.title("Options Tools")
app.geometry("1080x1920")

notebook = ttk.Notebook(app)
notebook.pack(fill=BOTH, expand=True)

# ------------------ TAB 1: Mispricing Calculator ------------------
tab1 = tb.Frame(notebook, padding=20)
notebook.add(tab1, text="Mispricing Calculator")

entry_S = add_labeled_entry(tab1, "Underlying Price (S):")
entry_K = add_labeled_entry(tab1, "Strike Price (K):")
entry_T = add_labeled_entry(tab1, "Time to Expiry (in days, T):")
entry_r = add_labeled_entry(tab1, "Risk-Free Rate (%) (r):")
entry_sigma = add_labeled_entry(tab1, "Volatility (%) (σ):")
entry_market = add_labeled_entry(tab1, "Market Price:")

tb.Label(tab1, text="Option Type:").pack(anchor=W, pady=5)
option_type_var = tk.StringVar(value="call")
tb.Combobox(tab1, textvariable=option_type_var, values=["call", "put"]).pack(fill=X)

tb.Button(tab1, bootstyle="success",text="Calculate Mispricing", command=calculate_mispricing).pack(pady=15)
result_var = tk.StringVar()
tb.Label(tab1, textvariable=result_var, font=("Segoe UI", 12, "bold")).pack()

# ------------------ TAB 2: Option Screener ------------------
tab2 = tb.Frame(notebook, padding=20)
notebook.add(tab2, text="Option Screener")

tb.Label(tab2, text="Option screener functionality coming soon...", font=("Segoe UI", 11)).pack(pady=50)


# --- TAB 3: Spread Optimizer ---
tab3 = tb.Frame(notebook, padding=20)
notebook.add(tab3, text="Spread Optimizer")

# Dropdown selector
tb.Label(tab3, text="Select a Spread Strategy:", font=("Segoe UI", 11)).pack(anchor=W, pady=(0, 5))
spread_var = tk.StringVar(value="Vertical Call Spread")
spread_options = [
    "Vertical Call Spread",
    "Vertical Put Spread",
    "Calendar Spread",
    "Straddle",
    "Strangle",
    "Iron Condor",
    "Iron Butterfly",
    "Ratio Spread",
    "Covered Call",
    "Protective Put",
    "Diagonal Spread",
    "Butterfly Spread",
    "Box Spread",
    "Collar",
    "Synthetic Long Stock",
    "Reverse Iron Condor",
    "Jade Lizard",
    "Reverse Jade Lizard",
    "Call Backspread",
    "Put Backspread",
    "Broken Wing Butterfly",
    "Unbalanced Condor",
    "Double Diagonal",
    "Christmas Tree Spread",
    "Lizard Spread"
]

tb.Combobox(tab3, textvariable=spread_var, values=spread_options).pack(fill=X)

# Multiline text output
description_text = tk.Text(tab3, wrap="word", font=("Segoe UI", 10), bg="#2b2b2b", fg="white", height=12)
description_text.pack(fill="both", expand=False, pady=(10, 10))

# Optional image label
image_label = tk.Label(tab3, bg="#2b2b2b")
image_label.pack(pady=(0, 10))

# Full spread descriptions (replace this dictionary with the full one from earlier)
spread_descriptions = {
    "Vertical Call Spread": (
        "🔹 **Vertical Call Spread**\n"
        "• Structure: Buy 1 lower strike call, sell 1 higher strike call (same expiry).\n"
        "• View: Moderately bullish.\n"
        "• Max Profit: Difference between strikes - net debit.\n"
        "• Max Loss: Net debit paid.\n"
        "• Breakeven: Lower strike + net debit.\n"
        "• Ideal when you expect the stock to rise but want limited risk."
    ),
    "Vertical Put Spread": (
        "🔹 **Vertical Put Spread**\n"
        "• Structure: Buy 1 higher strike put, sell 1 lower strike put (same expiry).\n"
        "• View: Moderately bearish.\n"
        "• Max Profit: Difference between strikes - net debit.\n"
        "• Max Loss: Net debit paid.\n"
        "• Breakeven: Higher strike - net debit.\n"
        "• Limits downside risk while betting on a drop."
    ),
    "Calendar Spread": (
        "🔹 **Calendar Spread**\n"
        "• Structure: Sell short-dated option, buy long-dated option (same strike).\n"
        "• View: Neutral short-term, directional long-term.\n"
        "• Max Profit: Occurs when underlying is near strike at short expiry.\n"
        "• Max Loss: Net debit paid.\n"
        "• Benefit: Takes advantage of time decay in short option.\n"
        "• Ideal for when you expect low volatility near-term but a bigger move later."
    ),
    "Straddle": (
        "🔹 **Straddle**\n"
        "• Structure: Buy 1 call and 1 put at the same strike and expiry.\n"
        "• View: Expect large movement in either direction.\n"
        "• Max Profit: Unlimited on upside, significant on downside.\n"
        "• Max Loss: Net debit (combined cost of both options).\n"
        "• Breakevens: Strike ± total premium.\n"
        "• Profits from volatility, commonly used around earnings."
    ),
    "Strangle": (
        "🔹 **Strangle**\n"
        "• Structure: Buy 1 OTM call and 1 OTM put (different strikes, same expiry).\n"
        "• View: Expect very large move in either direction.\n"
        "• Max Profit: Unlimited upside, significant downside.\n"
        "• Max Loss: Net debit (lower than straddle).\n"
        "• Breakevens: Call strike + debit and put strike - debit.\n"
        "• Cheaper than a straddle but requires a bigger move."
    ),
    "Iron Condor": (
        "🔹 **Iron Condor**\n"
        "• Structure: Sell OTM call and put, buy further OTM call and put (4 legs).\n"
        "• View: Expect sideways movement, low volatility.\n"
        "• Max Profit: Net credit received.\n"
        "• Max Loss: Width of either spread - net credit.\n"
        "• Breakevens: Short call + credit, short put - credit.\n"
        "• Great for range-bound markets with low implied volatility."
    ),
    "Iron Butterfly": (
        "🔹 **Iron Butterfly**\n"
        "• Structure: Sell ATM straddle (call + put), buy OTM wings.\n"
        "• View: Expect very little movement.\n"
        "• Max Profit: Net credit (when stock closes at middle strike).\n"
        "• Max Loss: Width of wing - credit.\n"
        "• Breakevens: Strike ± (wings - credit).\n"
        "• High reward-to-risk for neutral setups with high IV."
    ),
    "Ratio Spread": (
        "🔹 **Ratio Spread**\n"
        "• Structure: Buy 1 option, sell 2 (or more) of same type (same expiry).\n"
        "• View: Directional with volatility edge.\n"
        "• Max Profit: Complex; often uncapped.\n"
        "• Max Loss: Can be unlimited on the uncovered side.\n"
        "• Use with caution—potential for large directional loss.\n"
        "• Often combined with hedges to reduce risk."
    ),
    "Covered Call": (
        "🔹 **Covered Call**\n"
        "• Structure: Own 100 shares, sell 1 call option.\n"
        "• View: Neutral to slightly bullish.\n"
        "• Max Profit: Premium + upside to strike.\n"
        "• Max Loss: Stock falls (offset by premium).\n"
        "• Breakeven: Stock price - premium.\n"
        "• Generates passive income, but caps upside."
    ),
    "Protective Put": (
        "🔹 **Protective Put**\n"
        "• Structure: Own 100 shares, buy 1 put.\n"
        "• View: Bullish with downside protection.\n"
        "• Max Profit: Unlimited (stock appreciation).\n"
        "• Max Loss: Limited to (stock price - strike + premium).\n"
        "• Breakeven: Stock price + put premium.\n"
        "• Like insurance: caps losses, keeps upside."
    ),
    "Diagonal Spread": (
        "🔹 **Diagonal Spread**\n"
        "• Structure: Buy longer-dated option, sell shorter-dated option (different strikes).\n"
        "• View: Directional with a time-decay advantage.\n"
        "• Max Profit: When stock is at short strike near short expiry.\n"
        "• Max Loss: Net debit paid.\n"
        "• Benefit: Combines vertical and calendar spread features.\n"
        "• Great when expecting a short-term move followed by consolidation."
    ),
    "Butterfly Spread": (
        "🔹 **Butterfly Spread**\n"
        "• Structure: Buy 1 lower strike, sell 2 middle strikes, buy 1 higher strike (same expiry).\n"
        "• View: Neutral — expect low movement.\n"
        "• Max Profit: At middle strike at expiration.\n"
        "• Max Loss: Net debit paid.\n"
        "• Breakevens: Lower strike + net debit, higher strike - net debit.\n"
        "• Low cost strategy to profit from low volatility."
    ),
    "Box Spread": (
        "🔹 **Box Spread**\n"
        "• Structure: Bull call spread + bear put spread (same strikes and expiry).\n"
        "• View: Arbitrage (theoretically riskless).\n"
        "• Max Profit: Fixed—difference between strikes.\n"
        "• Max Loss: Only if mispriced or commissions impact trade.\n"
        "• Used by pros for synthetic lending or arbitrage.\n"
        "• Not directional—pure pricing play."
    ),
    "Collar": (
        "🔹 **Collar**\n"
        "• Structure: Own stock, buy protective put, sell covered call.\n"
        "• View: Conservative bullish — capped gain, limited loss.\n"
        "• Max Profit: Strike of call - purchase price + net premium.\n"
        "• Max Loss: Purchase price - put strike - net premium.\n"
        "• Breakeven: Stock price + net premium paid/received.\n"
        "• Ideal for hedging while generating income."
    ),
    "Synthetic Long Stock": (
        "🔹 **Synthetic Long Stock**\n"
        "• Structure: Buy 1 call, sell 1 put (same strike and expiry).\n"
        "• View: Bullish, same as owning the stock.\n"
        "• Max Profit: Unlimited.\n"
        "• Max Loss: Large if stock drops significantly.\n"
        "• Breakeven: Strike ± net premium.\n"
        "• Acts like long stock with less capital outlay, but same risk."
    ),
        "Reverse Iron Condor": (
        "🔹 **Reverse Iron Condor**\n"
        "• Structure: Buy OTM call and put, sell further OTM call and put (4 legs).\n"
        "• View: Expect large move in either direction.\n"
        "• Max Profit: Width of spreads - net debit.\n"
        "• Max Loss: Net debit paid.\n"
        "• Breakevens: Lower call - debit, higher put + debit.\n"
        "• Opposite of iron condor—used when expecting volatility breakout."
    ),
    "Jade Lizard": (
        "🔹 **Jade Lizard**\n"
        "• Structure: Sell OTM call spread and OTM put (3 legs).\n"
        "• View: Neutral to bullish.\n"
        "• Max Profit: Net credit received.\n"
        "• Max Loss: Unlimited on downside if put is breached.\n"
        "• Breakeven: Put strike - net credit.\n"
        "• Designed to avoid upside risk—no risk if stock rallies hard."
    ),
    "Reverse Jade Lizard": (
        "🔹 **Reverse Jade Lizard**\n"
        "• Structure: Sell OTM put spread and OTM call.\n"
        "• View: Neutral to bearish.\n"
        "• Max Profit: Net credit received.\n"
        "• Max Loss: Unlimited on upside if call is breached.\n"
        "• Breakeven: Call strike + credit.\n"
        "• Bearish version of Jade Lizard—avoids downside risk."
    ),
    "Call Backspread": (
        "🔹 **Call Backspread**\n"
        "• Structure: Sell 1 lower strike call, buy 2 higher strike calls (same expiry).\n"
        "• View: Very bullish with volatility edge.\n"
        "• Max Profit: Unlimited if stock surges.\n"
        "• Max Loss: If stock hovers near short call.\n"
        "• Breakevens: Depends on ratio; typically lower strike + debit.\n"
        "• Best when expecting strong upside breakout."
    ),
    "Put Backspread": (
        "🔹 **Put Backspread**\n"
        "• Structure: Sell 1 higher strike put, buy 2 lower strike puts (same expiry).\n"
        "• View: Very bearish with volatility edge.\n"
        "• Max Profit: Significant if stock crashes.\n"
        "• Max Loss: If stock stays near short put.\n"
        "• Breakevens: Depends on ratio; typically higher strike - debit.\n"
        "• Profits most when volatility spikes and price collapses."
    ),
        "Broken Wing Butterfly": (
        "🔹 **Broken Wing Butterfly**\n"
        "• Structure: Like a butterfly, but one wing is wider (uneven strikes).\n"
        "• View: Neutral to slightly directional (depends on skew).\n"
        "• Max Profit: When stock closes near middle strike.\n"
        "• Max Loss: Reduced on one side (compared to regular butterfly).\n"
        "• Breakevens: Adjusted based on wing width and credit/debit.\n"
        "• Can be placed for a credit to reduce or eliminate max loss."
    ),
    "Unbalanced Condor": (
        "🔹 **Unbalanced Condor**\n"
        "• Structure: 4-leg condor with different quantities on each side.\n"
        "• View: Range-bound with skewed probability bias.\n"
        "• Max Profit: Net credit received.\n"
        "• Max Loss: Defined by wider side of the spread.\n"
        "• Breakevens: Based on skew and net credit.\n"
        "• Used when IV skew favors one side more than the other."
    ),
    "Double Diagonal": (
        "🔹 **Double Diagonal**\n"
        "• Structure: Calendar + vertical on both call and put sides (4 legs).\n"
        "• View: Neutral with long volatility bias.\n"
        "• Max Profit: When underlying closes between short strikes.\n"
        "• Max Loss: Net debit paid.\n"
        "• Breakevens: Wide range around short strikes.\n"
        "• Great for range-bound stocks expected to gain IV."
    ),
    "Christmas Tree Spread": (
        "🔹 **Christmas Tree Spread**\n"
        "• Structure: Variation of butterfly/backspread with more contracts on one wing.\n"
        "• View: Directional but controlled risk.\n"
        "• Max Profit: Typically near clustered strikes.\n"
        "• Max Loss: Net debit or defined by outer strikes.\n"
        "• Breakevens: Varies—dependent on structure.\n"
        "• Creative structure to express a skewed directional view."
    ),
    "Lizard Spread": (
        "🔹 **Lizard Spread**\n"
        "• Structure: Variation of a ratio spread with no risk on one side.\n"
        "• View: Neutral to bullish.\n"
        "• Max Profit: Net credit received.\n"
        "• Max Loss: Only if price moves strongly in one direction.\n"
        "• Breakevens: Based on strike distance and credit.\n"
        "• Great for earnings plays with skewed IV."
    )



}

# Initialize
spread_var.trace_add("write", update_description)
update_description()


# ------------------- TAB 4: Visualizations ------------------
tab4 = tb.Frame(notebook, padding=20)
notebook.add(tab4, text="Payoff Visualization")

# ---- UI Inputs for Payoff ----
tb.Label(tab4, text="Select Spread Type:", font=("Segoe UI", 10)).pack(anchor=W, pady=(0, 5))
spread_type_var = tk.StringVar(value="Vertical Call Spread")
tb.Combobox(tab4, textvariable=spread_type_var, values=["Vertical Call Spread", "Vertical Put Spread"]).pack(fill=X)


entry_K1 = labeled_entry(tab4, "Buy Strike (K1):")
entry_K2 = labeled_entry(tab4, "Sell Strike (K2):")
entry_P1 = labeled_entry(tab4, "Buy Premium (P1):")
entry_P2 = labeled_entry(tab4, "Sell Premium (P2):")


# ---- Button ----
tb.Button(tab4, text="Plot Payoff", command=plot_spread_payoff).pack(pady=10)

# ---- Matplotlib Canvas ----
fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
canvas = FigureCanvasTkAgg(fig, master=tab4)
canvas.get_tk_widget().pack(fill=BOTH, expand=True, pady=10)

# ------------------ TAB 5: Option Chain Viewer ------------------
tab5 = tb.Frame(notebook, padding=20)
notebook.add(tab5, text="Option Chain Viewer")

# --- UI Elements ---
symbol_var = tk.StringVar(value="AAPL")
expiration_var = tk.StringVar()

tb.Label(tab5, text="Ticker Symbol:").pack(anchor=W, pady=(0, 2))
symbol_entry = tb.Entry(tab5, textvariable=symbol_var)
symbol_entry.pack(fill=X)
symbol_entry.bind("<Return>", lambda e: load_expirations())  # Enter to load expirations

expiration_menu = ttk.Combobox(tab5, textvariable=expiration_var)
expiration_menu.pack(fill=X, pady=(10, 5))
expiration_menu.bind("<<ComboboxSelected>>", lambda e: load_chain())  # Change expiration triggers reload

# --- Treeview Column Setup ---
columns = ("Strike", "Bid", "Ask", "IV", "Volume", "Delta", "Gamma", "Theta", "Vega", "Rho")

frame = tb.Frame(tab5)
frame.pack(fill=BOTH, expand=True, pady=10)

frame.grid_rowconfigure(1, weight=1)
frame.grid_columnconfigure(0, weight=1)
frame.grid_columnconfigure(1, weight=1)

# ----- CALL TREE -----
call_label = tb.Label(frame, text="Calls", anchor="center", font=("Segoe UI", 10, "bold"))
call_label.grid(row=0, column=0, sticky="n", padx=10)

call_tree = ttk.Treeview(frame, columns=columns, show="headings")
call_tree.tag_configure("atm", background="#1f77b4")  # Blue highlight for ATM call
for col in columns:
    call_tree.heading(col, text=col)
    call_tree.column(col, anchor="center", width=80)
call_tree.grid(row=1, column=0, sticky="nsew", padx=10)

# ----- PUT TREE -----
put_label = tb.Label(frame, text="Puts", anchor="center", font=("Segoe UI", 10, "bold"))
put_label.grid(row=0, column=1, sticky="n", padx=10)

put_tree = ttk.Treeview(frame, columns=columns, show="headings")
put_tree.tag_configure("atm", background="#ff7f0e")  # Orange highlight for ATM put
for col in columns:
    put_tree.heading(col, text=col)
    put_tree.column(col, anchor="center", width=80)
put_tree.grid(row=1, column=1, sticky="nsew", padx=10)
#------------- TAB 6: Vol Skew / Term structure ------------------
# Create new tab
tab6 = tb.Frame(notebook, padding=20)
notebook.add(tab6, text="Volatility Skew")

# Controls Frame
controls = tb.Frame(tab6)
controls.pack(fill=X, pady=(0, 10))

# Dropdown for option type
option_type_skew = tk.StringVar(value="call")
tb.Label(controls, text="Option Type:").pack(side=LEFT, padx=(0, 5))
tb.Combobox(controls, textvariable=option_type_skew, values=["call", "put", "both"], width=10).pack(side=LEFT)

# Expiration filtering (multi-select)
expiration_filter_frame = tb.Frame(tab6)
expiration_filter_frame.pack(fill=X)
expiration_vars = {}

# Store loaded chains for each expiration
temp_loaded_option_data = {}

# Button to plot skew
plot_button = tb.Button(controls, text="Plot Skew", command=lambda: plot_volatility_skew())
plot_button.pack(side=LEFT, padx=10)

# Button to plot term structure
term_button = tb.Button(controls, text="Term Structure", command=lambda: plot_term_structure())
term_button.pack(side=LEFT, padx=10)

# Matplotlib area
fig_skew = Figure(figsize=(6, 4), dpi=100)
ax_skew = fig_skew.add_subplot(111)
skew_canvas = FigureCanvasTkAgg(fig_skew, master=tab6)
skew_canvas.get_tk_widget().pack(fill=BOTH, expand=True)

# --- TAB 7: Put/Call IV Ratio ---
tab7 = tb.Frame(notebook, padding=20)
notebook.add(tab7, text="Put/Call IV Ratio")

# Controls
iv_ratio_controls = tb.Frame(tab7)
iv_ratio_controls.pack(fill=X, pady=(0, 10))

# Shared expiration state
selected_iv_expirations = set()
iv_checkbox_vars = {}

# Button to launch selector
ttk.Button(iv_ratio_controls, text="Select Expirations", command=open_expiration_selector).pack(side=LEFT, padx=5)

# Button to plot
iv_ratio_button = tb.Button(iv_ratio_controls, text="Plot IV Ratio", command=lambda: plot_put_call_iv_ratio())
iv_ratio_button.pack(side=LEFT, padx=10)

# Plot Area
fig_ratio = Figure(figsize=(6, 4), dpi=100)
ax_ratio = fig_ratio.add_subplot(111)
ratio_canvas = FigureCanvasTkAgg(fig_ratio, master=tab7)
ratio_canvas.get_tk_widget().pack(fill=BOTH, expand=True)

# ------------------ TAB 8: Strategy Builder ------------------

tab8 = tb.Frame(notebook, padding=20)
notebook.add(tab8, text="Strategy Builder")


# --- Widgets ---

builder_controls = tb.Frame(tab8)
builder_controls.pack(fill=X, pady=10)

builder_legs = []
strategy_data = []

# Symbol input
tb.Label(builder_controls, text="Ticker Symbol:").grid(row=0, column=0, sticky="w")
builder_symbol = tb.Entry(builder_controls)
builder_symbol.grid(row=0, column=1, padx=5)
builder_symbol.insert(0, "AAPL")

# Expiration dropdown
tb.Label(builder_controls, text="Expiration:").grid(row=0, column=2, sticky="w")
builder_expiration_var = tk.StringVar()
builder_exp_menu = ttk.Combobox(builder_controls, textvariable=builder_expiration_var)
builder_exp_menu.grid(row=0, column=3, padx=5)

builder_symbol.bind("<Return>", lambda e: load_builder_expirations())
builder_exp_menu.bind("<<ComboboxSelected>>", lambda e: load_builder_strikes())

# Strategy Selector
strategy_frame = tb.Frame(tab8)
strategy_frame.pack(fill=X, pady=10)

tb.Label(strategy_frame, text="Select Strategy:").pack(side=LEFT, padx=5)

strategy_type_var = tk.StringVar()
strategy_menu = ttk.Combobox(strategy_frame, textvariable=strategy_type_var, width=30)
strategy_menu["values"] = [
    "Vertical Call Spread", "Vertical Put Spread", "Calendar Spread", "Straddle",
    "Strangle", "Iron Condor", "Iron Butterfly", "Ratio Spread", "Covered Call",
    "Protective Put", "Diagonal Spread", "Butterfly Spread", "Box Spread", "Collar",
    "Synthetic Long Stock", "Reverse Iron Condor", "Jade Lizard", "Reverse Jade Lizard",
    "Call Backspread", "Put Backspread", "Broken Wing Butterfly", "Unbalanced Condor",
    "Double Diagonal", "Christmas Tree Spread", "Lizard Spread"
]
strategy_menu.current(0)
strategy_menu.pack(side=LEFT)

tb.Button(strategy_frame, text="Generate Strategy", command=lambda: build_predefined_strategy(strategy_type_var.get())).pack(side=LEFT, padx=10)


# Load strike prices
builder_strike_list = []
builder_strike_var = tk.DoubleVar()

# Leg Form
leg_frame = tb.Frame(tab8)
leg_frame.pack(fill=X, pady=5)

option_type_var = tk.StringVar(value="call")
side_var = tk.StringVar(value="buy")
qty_var = tk.IntVar(value=1)

tb.Label(leg_frame, text="Option Type:").grid(row=0, column=0)
tb.Combobox(leg_frame, textvariable=option_type_var, values=["call", "put"], width=10).grid(row=0, column=1)

tb.Label(leg_frame, text="Buy/Sell:").grid(row=0, column=2)
tb.Combobox(leg_frame, textvariable=side_var, values=["buy", "sell"], width=10).grid(row=0, column=3)

tb.Label(leg_frame, text="Strike:").grid(row=0, column=4)
builder_strike_dropdown = ttk.Combobox(leg_frame, textvariable=builder_strike_var, values=builder_strike_list, width=10)
builder_strike_dropdown.grid(row=0, column=5)

builder_leg_expiration_var = tk.StringVar()
tb.Label(leg_frame, text="Expiration:").grid(row=0, column=8)
builder_leg_exp_menu = ttk.Combobox(leg_frame, textvariable=builder_leg_expiration_var, width=15)
builder_leg_exp_menu.grid(row=0, column=9)

tb.Label(leg_frame, text="Qty:").grid(row=0, column=6)
tb.Entry(leg_frame, textvariable=qty_var, width=5).grid(row=0, column=7)

sim_price_var = tk.DoubleVar()
sim_price_label = tk.StringVar()


# Leg Table
table_frame = tb.Frame(tab8)
table_frame.pack(fill=X, pady=10)

columns = ("Type", "Side", "Strike", "Qty", "Price","Expiration")
leg_tree = ttk.Treeview(table_frame, columns=columns, show="headings", height=6)
for col in columns:
    leg_tree.heading(col, text=col)
    leg_tree.column(col, width=80, anchor="center")
leg_tree.pack(side=LEFT, fill="x", expand=True)

# Editable Leg Panel
editor_frame = tb.Frame(tab8)
editor_frame.pack(fill=X, pady=(5, 10))

edit_strike_var = tk.DoubleVar()
edit_exp_var = tk.StringVar()
selected_leg_index = tk.IntVar(value=-1)

tb.Label(editor_frame, text="Edit Strike:").pack(side=LEFT)
edit_strike_menu = ttk.Combobox(editor_frame, textvariable=edit_strike_var, width=10)
edit_strike_menu.pack(side=LEFT, padx=5)

tb.Label(editor_frame, text="Edit Expiration:").pack(side=LEFT)
edit_exp_menu = ttk.Combobox(editor_frame, textvariable=edit_exp_var, width=15)
edit_exp_menu.pack(side=LEFT, padx=5)
edit_exp_var.trace_add("write", lambda *_, e=None: refresh_edit_strike_choices(edit_exp_var.get()))


leg_tree.bind("<<TreeviewSelect>>", load_leg_for_edit)

tb.Button(editor_frame, text="Update Leg", command=update_leg).pack(side=LEFT, padx=10)

tb.Button(leg_frame, text="Add Leg", command=add_leg).grid(row=0, column=8, padx=10)
tb.Button(leg_frame, text="Remove Selected", command=remove_leg).grid(row=0, column=9, padx=5)
tb.Button(leg_frame, text="Reset All", command=reset_strategy).grid(row=0, column=10, padx=5)

# Plot area
fig_builder = Figure(figsize=(6, 4), dpi=100)
ax_builder = fig_builder.add_subplot(111)
builder_canvas = FigureCanvasTkAgg(fig_builder, master=tab8)
builder_canvas.get_tk_widget().pack(fill=BOTH, expand=True)

metrics_var = tk.StringVar()
tb.Label(tab8, textvariable=metrics_var, font=("Segoe UI", 10, "bold"), foreground="lightgreen").pack(pady=5)

# Simulated price slider
sim_frame = tb.Frame(tab8)
sim_frame.pack(fill=X, pady=(5, 10))

tb.Label(sim_frame, text="Simulated Price:").pack(side=LEFT, padx=(0, 10))
sim_slider = ttk.Scale(sim_frame, from_=0, to=1000, variable=sim_price_var, command=lambda v: update_sim_price())
sim_slider.pack(side=LEFT, fill=X, expand=True)
tb.Label(sim_frame, textvariable=sim_price_label).pack(side=LEFT, padx=10)



# Start app
app.mainloop()