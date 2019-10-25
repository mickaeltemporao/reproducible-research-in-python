"""Data Acquisition Module."""

import wikipedia as wp
import pandas as pd

# Identify Wikipedia Page and acquire the date
PAGE_TITLE = "Opinion polling for the 2019 Canadian federal election"

def get_data(page_title=PAGE_TITLE):
    """Get the html source, and Extract tables into a pd.DataFrame()."""
    html = wp.page(page_title).html().encode("UTF-8")
    return pd.read_html(html)[0].iloc[2:,:]

if __name__ == '__main__':
    df = get_data()
