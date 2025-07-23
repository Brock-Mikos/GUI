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

from PIL import Image, ImageTk
import os

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


# Start app
app.mainloop()