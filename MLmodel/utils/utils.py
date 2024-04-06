# import built-in packages
from datetime import datetime, timedelta

# import third party packages
from pandas import DataFrame


def matlab_time_to_datetime(date_column: DataFrame) -> list[str]:
    '''
    convert matlab date time column to readable time

    param: data Dataframe
    '''

    start_date = datetime(1, 1, 1)
    dates = []

    for date_decimal in date_column:
        # Convert decimal to datetime object
        delta = timedelta(days=date_decimal)

        date = start_date + delta

        # Output: 2003-01-01
        dates.append(date.strftime('%Y-%m-%d %H'))

    return dates
