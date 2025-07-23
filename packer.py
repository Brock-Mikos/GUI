import ttkbootstrap as ttk
from ttkbootstrap.constants import *

def make_entry_widget(parent_widget, entry_label):
    widget_label = ttk.Label(parent_widget, text=entry_label)
    widget_label.pack(anchor = 'w', pady = 5)
    entry = ttk.Entry(parent_widget)
    entry.pack(fill=X)
    return entry
