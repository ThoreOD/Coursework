import json
import math
import numpy as np
import pandas as pd
import requests

from typing import Union, List

def wb(country_code: Union[str, List[str]], indicator: Union[str, List[str]], year: str) -> pd.DataFrame:
    """Retrieval of data from the World Bank API.

    Keyword arguments:
    country_code -- alpha-2 or alpha-3 country code (string or list of string/s) or all for each country code (string or list of string)
    indicator (version 2) -- world bank indicator (string or list of string/s)
    year (1960 to 2021) -- digit specifying year/s of interest (string)
                       --> if two years, single string with years separated by "-" (e.g. "2010-2012")

    Returns:
    pandas.DataFrame -- DataFrame conatining the data for the specified country code/s, indicator/s, and year/s based on the world development indicators.
    --> provides alpha_2_code, alpha_3_code, country, year, and the indicators as features

    Raises:
    Data Error (Indicator) -- If the data for entered indicator/s is not available.
    Data Error (Year) -- If the data for entered year/s is not available.
    Input Error -- If the entered country code/s, indicator/s, and/or year/s are not recognized.
    Sequence Error -- If the entered years are in reverse order.
    Type Error -- If the type of entered country code/s, indicator/s, and/or year/s is not valid.
    --> returns empty list

    Example:
    >> wb("all", ["SP.POP.TOTL", "SH.MED.PHYS.ZS"], "2005-2010")
    """

    df = []
    
    try:
        year = year.split("-")
        if len(year) >= 2:
            year_range = int(year[1]) - int(year[0])
            if int(year[1]) != 2022:
                year_range += 1
        else:
            year_range = 1
        if country_code == "all":
            country_range = 266
        else:
            country_range = len(country_code)
        indicator_range = len(indicator)
        
        max_load = 32767
        total_pages = math.ceil((year_range * country_range * indicator_range) / max_load)

        def arg_prep(country_code, indicator, year):
            arg = [country_code, indicator, year]
            for i, j in enumerate(arg):
                if isinstance(j, list):
                    char = ":" if i == 2 else ";"
                    arg[i] = char.join(j)
            return arg[0], arg[1], arg[2]
        country_code, indicator, year = arg_prep(country_code, indicator, year)
            
        try:
            for i in range(1, total_pages + 1):
                params = {"format": "json", 
                          "per_page": max_load,
                          "page": i,
                          "date": year,
                          "source": 2}
                endpoint = f"https://api.worldbank.org/v2/en/country/{country_code}/indicator/{indicator}"
                response = requests.get(endpoint, params).json()[1]
                subset = pd.json_normalize(data = response,
                                           sep = "_")
                df.append(subset)
                
            non_country_id = ["ZH", "ZI", "1A", "S3", "B8", "V2", "Z4", "4E", "T4", "XC", "Z7", "7E", "T7", 
                              "EU", "F1", "XE", "XD", "XF", "ZT","XH", "XI", "XG", "V3", "ZJ", "XJ", "T2", 
                              "XL", "XO", "XM", "XN", "ZQ", "XQ", "T3", "XP", "XU", "XY", "OE", "S4", "S2", 
                              "V4", "V1", "S1", "8S", "T5", "ZG", "ZF", "T6", "XT", "1W"]
            df = pd.concat(objs = df)
            df = df[~df.country_id.isin(non_country_id)]
            df.replace(to_replace = "", 
                       value = np.nan, 
                       inplace = True)
            df.dropna(axis = 1,
                      how = "all",
                      inplace = True)
            try:
                df['value'] = df.apply(lambda row: round(number = row["value"], 
                                                         ndigits = row["decimal"]), 
                                                         axis = 1 )
                df.drop(labels = ["decimal", 
                                  "indicator_value"],
                        axis = 1,
                        inplace = True)
                df = df.pivot(values = "value",
                              index = ["country_id",
                                       "countryiso3code",
                                       "country_value",
                                       "date"],
                              columns = "indicator_id").sort_index(level = ["country_value", 
                                                                            "date"]).reset_index().rename_axis(mapper = None,
                                                                                                               axis = 1)
                df.rename(columns = {"country_id": "alpha_2_code",
                                     "countryiso3code": "alpha_3_code",
                                     "country_value": "country",
                                     "date": "year"},
                          inplace = True)
                df.columns = df.columns.str.lower()

            except KeyError:
                print(f"\033[1mData Error (Indicator): \033[0mThe data for the requested indicator is not available under the specific country code year combination entered. Please try enter a different country code year combination. If you're unsure refer to the function's documentation for guidance.")
                df = []

        except IndexError:
            print(f"\033[1mInput Error: \033[0mThe entered input (e.g. country code, indicator, year) has not been recognized. Please review the spelling and try again. If you're unsure refer to the function's documentation for guidance.")

        except ValueError:
            print(f"\033[1mSequence Error: \033[0m The sequence of years was entered in reverse order. Please review the chronology and try again. If you're unsure refer to the function's documentation for guidance.")

        except NotImplementedError:
            print(f"\033[1mData Error (Year): \033[0mThe data for the requested year is not yet available. Please try again later or enter a different year. If you're unsure refer to the function's documentation for guidance.")

    except (TypeError, AttributeError):
        print(f"\033[1mType Error: \033[0mThe type of the entered input (e.g. country code, indicator, year) is not valid. Please review the type and try again. If you're unsure refer to the function's documentation for guidance.")
    
    return df