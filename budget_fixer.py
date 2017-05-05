from bs4 import BeautifulSoup
import urllib.request
import pandas as pd
import numpy as np
from datetime import date



def historical_currency_converter(amount, year, currency, hist_currency_dict = None):
    """:returns amount * converted_ratio_of_currency_versus_USD for given year"""

    if currency == 'USD':
        return amount

    key = str(currency)+'_'+str(year)

    if hist_currency_dict is None:
        # memoization
        try:
            hist_currency_dict = pd.read_pickle('hist_currency_dict.pickle')
        except:
            hist_currency_dict = dict()


    if key in hist_currency_dict:
        return amount * hist_currency_dict[key]


    if year < 1996:
        print("year fixed 1996 cuz of lack of data")
        year = 1996

    day = 1
    while True:     # while True is here because first day of every year may not be business day
        url = "http://www.xe.com/currencytables/?from=USD&date={}-01-0{}".format(int(year),int(day))
        print("url: {}", url)
        df = pd.read_html(url)[0]

        # if market is open!
        if len(df.columns) == 4: # table of return of read_html has 4 column
            break
        day += 1

    df.columns = ['code','name', 'units_per_usd', 'usd_per_unit']


    # get usd_per_unit for given currency
    if np.sum(df.code.str.contains(currency)) > 0: # if we have currency in dataframe
        val = (df.loc[df.code == currency])['usd_per_unit'].tolist()[0]
    else:
        print("Currency could not found!")
        val = np.nan

    hist_currency_dict[key] = val
    pd.to_pickle(hist_currency_dict, 'hist_currency_dict.pickle')

    return amount * hist_currency_dict[key]

def inflation_ratio(old_year,new_year):
    """only support usd, return new_date's value"""
    # todo memoization
    amount = 1
    url_address = "https://data.bls.gov/cgi-bin/cpicalc.pl?cost1={}&year1={}&year2={}".format(int(amount),int(old_year),int(new_year))
    print(url_address)
    with urllib.request.urlopen(url_address) as url:

        html_doc = url.read()

    soup = BeautifulSoup(html_doc,"lxml")


    return float(soup.find("span", {"id": "answer"}).contents[0][1:])



def get_country_currency_table():
    """:returns DataFrame['country','currency_abbv']"""
    raw = pd.read_csv("country_currency_codes.csv")

    df = pd.DataFrame()
    df['country'] = raw['name'].iloc[2:]
    df['currency'] = raw['ISO4217-currency_alphabetic_code'].iloc[2:]

    df.country = df.country.str.replace('US','USA')

    return df.reset_index().drop('index',axis=1)

def handler(movie_IDs, movie_countries, movie_budgets, movie_years, hist_currency_dict = None):
    """Calculates inflation rate and use historical currency converter
    :param pd.Series movie_countries
    :param pd.Series movie_budgets
    :param pd.Series movie_years
    :returns DataFrame['usd_old','usd_today', 'movie_title', 'title_year']"""
    country_currency_table = get_country_currency_table()

    movie_currencies = pd.Series(np.zeros(len(movie_countries)))

    ret_df = pd.DataFrame(columns=['usd_old', 'usd_today'])

    for i, country in enumerate(country_currency_table.country.values):
        # select rows which has same country names and then update curreny column
        movie_currencies.loc[movie_countries == country] = country_currency_table.currency.iloc[i]

    for i, (id, budget, year, currency) in enumerate(zip(movie_IDs.values, movie_budgets.values, movie_years.values, movie_currencies.values)):
        print("id: {} || year: {} || currency: {}".format(id, year, currency))
        if isinstance(currency, str):     # valid currency
            usd_old = historical_currency_converter(budget, year, currency, hist_currency_dict)
            usd_today = usd_old * get_cpi_rate(year, 2017)
        else:
            print("non-valid currency detected!")
            usd_old = np.nan
            usd_today = np.nan

        ret_df.loc[i, 'usd_old'] = usd_old
        ret_df.loc[i, 'usd_today'] = usd_today
        ret_df.loc[i, 'ID'] = id

        print("usd_old: {} || usd_today: {}".format(usd_old,usd_today))

    return ret_df


def get_cpi_rates():
    """
    this function read cpi rate csv then returns dataframe
    usage : print(get_cpi_rates().loc[1940,'cpi_rate'])
    """

    df = pd.read_csv('cpi_usa.csv', index_col=0)
    df.index = pd.to_datetime(df.index)

    df = df.resample('BAS').mean()  # change sampling to business year start
    df.index = df.index.year    # datetime to year
    df.columns = ['cpi_rate']

    return df

def get_cpi_rate(old_year, new_year):
    """:returns rate_of_new_year / rate_of_old_year"""
    if new_year < old_year:
        raise Exception("old_year should be older than new_year")

    if old_year < 1913:
        print("old_year fixed 1913 cuz of lack of data")
        old_year = 1913

    df = get_cpi_rates()

    return df.loc[new_year, 'cpi_rate'] / df.loc[old_year, 'cpi_rate']



# print(historical_currency_converter(100,2001,))


# #
# pd.set_option('display.width', 480)
#
#
# movie_year = 2006
# movie_budget = 100
# usd_old = historical_currency_converter(movie_budget, movie_year, 'EUR')
# usd_today = usd_old*inflation_ratio( movie_year, 2017)
# print(usd_old)
# print(usd_today)





