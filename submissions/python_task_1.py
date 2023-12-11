import pandas as pd


def generate_car_matrix(df) -> pd.DataFrame:
    """
    Creates a DataFrame for id combinations.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Matrix generated with 'car' values, 
                          where 'id_1' and 'id_2' are used as indices and columns respectively.
    """
    car_matrix = df.pivot(index='id_1', columns='id_2', values='car').fillna(0)
    car_matrix.values[[range(len(car_matrix))]*2] = 0
    return car_matrix


def get_type_count(df) -> dict:
    """
    Categorizes 'car' values into types and returns a dictionary of counts.

    Args:
        df (pandas.DataFrame)

    Returns:
        dict: A dictionary with car types as keys and their counts as values.
    """
    conditions = [
        (df['car'] <= 15),
        (df['car'] > 15) & (df['car'] <= 25),
        (df['car'] > 25)
    ]
    choices = ['low', 'medium', 'high']
    df['car_type'] = pd.Series(np.select(conditions, choices))
    type_count = df['car_type'].value_counts().to_dict()
    return dict(sorted(type_count.items()))


def get_bus_indexes(df) -> list:
    """
    Returns the indexes where the 'bus' values are greater than twice the mean.

    Args:
        df (pandas.DataFrame)

    Returns:
        list: List of indexes where 'bus' values exceed twice the mean.
    """
    bus_mean = df['bus'].mean()
    bus_indexes = df[df['bus'] > 2 * bus_mean].index.tolist()
    return sorted(bus_indexes)


def filter_routes(df) -> list:
    """
    Filters and returns routes with average 'truck' values greater than 7.

    Args:
        df (pandas.DataFrame)

    Returns:
        list: List of route names with average 'truck' values greater than 7.
    """
    routes_above_7 = df.groupby('route')['truck'].mean(
    ).loc[lambda x: x > 7].index.tolist()
    return sorted(routes_above_7)


def multiply_matrix(matrix) -> pd.DataFrame:
    """
    Multiplies matrix values with custom conditions.

    Args:
        matrix (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Modified matrix with values multiplied based on custom conditions.
    """
    modified_matrix = matrix.applymap(
        lambda x: x * 0.75 if x > 20 else x * 1.25)
    return modified_matrix.round(1)


def time_check(df) -> pd.Series:
    """
    Use shared dataset-2 to verify the completeness of the data by checking whether the timestamps for each unique (`id`, `id_2`) pair cover a full 24-hour and 7 days period

    Args:
        df (pandas.DataFrame)

    Returns:
        pd.Series: return a boolean series
    """
    
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['time_of_day'] = df['timestamp'].dt.time

    completeness_check = df.groupby(['id', 'id_2'])[['day_of_week', 'time_of_day']].agg({
        # Check if all 7 days are covered
        'day_of_week': lambda x: len(set(x)) == 7,
        # Check if 24-hour period is covered
        'time_of_day': lambda x: x.max() - x.min() == pd.Timedelta(hours=23, minutes=59, seconds=59)
    })

    return completeness_check.all(axis=1)
