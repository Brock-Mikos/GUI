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

# ------------------- TAB 3: Spread Creator ------------------
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

        result_var.set(f"Model Price: ${model_price:.2f}\nMispricing: {mispricing:.2f}%")
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


# --- GUI Setup ---
app = tb.Window(themename="darkly")
app.title("Options Tools")
app.geometry("600x450")

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

tb.Button(tab1, text="Calculate Mispricing", bootstyle=SUCCESS, command=calculate_mispricing).pack(pady=15)
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
    "Protective Put"
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

def labeled_entry(parent, label):
    tb.Label(parent, text=label).pack(anchor=W, pady=(10, 2))
    e = tb.Entry(parent)
    e.pack(fill=X)
    return e

entry_K1 = labeled_entry(tab4, "Buy Strike (K1):")
entry_K2 = labeled_entry(tab4, "Sell Strike (K2):")
entry_P1 = labeled_entry(tab4, "Buy Premium (P1):")
entry_P2 = labeled_entry(tab4, "Sell Premium (P2):")

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

# ---- Button ----
tb.Button(tab4, text="Plot Payoff", bootstyle=SUCCESS, command=plot_spread_payoff).pack(pady=10)

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

expiration_menu = ttk.Combobox(tab5, textvariable=expiration_var)
expiration_menu.pack(fill=X, pady=(10, 5))

columns = ("Strike", "Bid", "Ask", "IV", "Volume")
tree = ttk.Treeview(tab5, columns=columns, show="headings", height=15)
for col in columns:
    tree.heading(col, text=col)
    tree.column(col, anchor="center")
tree.pack(fill="both", expand=True)

def load_expirations():
    ticker = symbol_var.get().strip().upper()
    print(f"Fetching expirations for {ticker}")  # 👈 DEBUG
    if not ticker:
        return
    try:
        stock = yf.Ticker(ticker)
        expirations = stock.options
        print("Expirations:", expirations)  # 👈 DEBUG
        expiration_menu["values"] = expirations
        if expirations:
            expiration_var.set(expirations[0])
    except Exception as e:
        print("Error loading expirations:", e)
        expiration_menu["values"] = []
        expiration_var.set("")
        
def load_chain():
    tree.delete(*tree.get_children())
    ticker = symbol_var.get().strip().upper()
    expiration = expiration_var.get()
    if not ticker or not expiration:
        return
    try:
        stock = yf.Ticker(ticker)
        chain = stock.option_chain(expiration)
        calls = chain.calls[["strike", "bid", "ask", "impliedVolatility", "volume"]]
        for _, row in calls.iterrows():
            tree.insert("", tk.END, values=(
                row["strike"], row["bid"], row["ask"],
                f"{row['impliedVolatility']*100:.2f}%", row["volume"]
            ))
    except Exception as e:
        print("Error loading chain:", e)

tb.Button(tab5, text="Load Expirations", bootstyle=INFO, command=load_expirations).pack(pady=5)
tb.Button(tab5, text="Load Option Chain", bootstyle=PRIMARY, command=load_chain).pack(pady=(0, 10))

# Start app
app.mainloop()
