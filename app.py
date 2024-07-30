from flask import Flask, render_template
import fcns as f

import matplotlib.pyplot as plt
import numpy as np
import os
import io
import random


app = Flask(__name__, static_url_path='/static', static_folder='static')

instance = f.portfolio("Activities_for_01Dec2019_to_13Jul2024.csv")
port_dict = instance.df_portfolio.to_dict(orient='index')
tickers = list(port_dict.keys())


@app.template_filter('round')
def round_filter(value):
    try:
        return round(value, 2)
    except (ValueError, TypeError):
        return value


app.jinja_env.filters['round'] = round_filter

directory_path = ''
time_series_plots_path = [os.path.join(
    'img', x) for x in os.listdir(directory_path)]
print(time_series_plots_path)


@app.route('/')
def index():
    enumerated_tickers = [(i + 1, ticker) for i, ticker in enumerate(tickers)]
    return render_template('index.html',
                            enumerated_tickers=enumerated_tickers,
                            portfolio=port_dict,
                            last_date=instance.last_updated_date,
                            
                            ## TODO this should all be in the same dictionary instead of passing n lists
                            daily_close = instance.daily_close,
                            prev_close = instance.prev_close,
                            daily_pl=instance.daily_pl,
                            
                            xr=instance.todays_exchange,
                            open_pl=instance.market_pl,
                            total_div=instance.total_div,
                            total_pl=instance.total_pl,
                            imgs=time_series_plots_path
                            )


if __name__ == '__main__':
    app.run(debug=True)
