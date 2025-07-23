import ttkbootstrap as ttk
from ttkbootstrap.constants import *
import packer

root = ttk.Window(title="Options Tools", themename="superhero")

notebook = ttk.Notebook(root)

mispricing_calculator = ttk.Frame(notebook)
option_screener = ttk.Frame(notebook)
spread_optimizer = ttk.Frame(notebook)
visualizations = ttk.Frame(notebook)
option_chain_viewer = ttk.Frame(notebook)

notebook.add(mispricing_calculator, text="Mispricing Calculator", padding=20)
notebook.add(option_screener, text="Option Screener")
notebook.add(spread_optimizer, text="Spread Optimizer")
notebook.add(visualizations, text="Visualizations")
notebook.add(option_chain_viewer, text="Option Chain Viewer")

underlying_price = packer.make_entry_widget(mispricing_calculator, "Underlying Price(S):")
strike_price = packer.make_entry_widget(mispricing_calculator, "Strike Price (K):")
time_to_expire = packer.make_entry_widget(mispricing_calculator, "Time to Expiry (in days, T):")
risk_free_rate = packer.make_entry_widget(mispricing_calculator, "Risk-Free Rate (%) (r):")
volitility = packer.make_entry_widget(mispricing_calculator,  "Volatility (%) (σ):")
market_price = packer.make_entry_widget(mispricing_calculator, "Market Price:")

notebook.pack(expand=True, fill='both')

root.mainloop()
