import pandas as pd
import numpy as np


def calculate_distance_matrix(df) -> pd.DataFrame:
    """
    Calculate a distance matrix based on the dataframe, df.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Distance matrix
    """
   
    distance_matrix = df.pivot(
        index='id_start', columns='id_end', values='distance').fillna(0)
    distance_matrix = distance_matrix.add(distance_matrix.T, fill_value=0)
    np.fill_diagonal(distance_matrix.values, 0)
    return distance_matrix


def unroll_distance_matrix(df) -> pd.DataFrame:
    """
    Unroll a distance matrix to a DataFrame in the style of the initial dataset.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Unrolled DataFrame containing columns 'id_start', 'id_end', and 'distance'.
    """
    unrolled_df = df.stack().reset_index()
    unrolled_df.columns = ['id_start', 'id_end', 'distance']
    return unrolled_df


def find_ids_within_ten_percentage_threshold(df, reference_id) -> pd.DataFrame:
    """
    Find all IDs whose average distance lies within 10% of the average distance of the reference ID.

    Args:
        df (pandas.DataFrame)
        reference_id (int)

    Returns:
        pandas.DataFrame: DataFrame with IDs whose average distance is within the specified percentage threshold
                          of the reference ID's average distance.
    """
    avg_distance_reference = df[df['id_start']
                                == reference_id]['distance'].mean()
    threshold = 0.1 * avg_distance_reference
    result_df = df.groupby('id_start')['distance'].mean().loc[lambda x: (
        x >= avg_distance_reference - threshold) & (x <= avg_distance_reference + threshold)].reset_index()
    return result_df.sort_values(by='id_start')


def calculate_toll_rate(df) -> pd.DataFrame:
    """
    Calculate toll rates for each vehicle type based on the unrolled DataFrame.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
  
    toll_rates = {
        'moto': 0.8,
        'car': 1.2,
        'rv': 1.5,
        'bus': 2.2,
        'truck': 3.6
    }

    for vehicle_type, rate in toll_rates.items():
        df[vehicle_type] = df['distance'] * rate

    return df


def calculate_time_based_toll_rates(df) -> pd.DataFrame:
    """
    Calculate time-based toll rates for different time intervals within a day.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
    
    df['start_day'] = df['start_time'] = df['end_day'] = df['end_time'] = None

    
    df['start_day'] = pd.to_datetime(df['start_day'])
    df['end_day'] = pd.to_datetime(df['end_day'])
    df['start_time'] = pd.to_datetime(df['start_time']).dt.time
    df['end_time'] = pd.to_datetime(df['end_time']).dt.time

    
    weekday_time_ranges = [(pd.to_datetime('00:00:00').time(), pd.to_datetime('10:00:00').time()),
                           (pd.to_datetime('10:00:00').time(),
                            pd.to_datetime('18:00:00').time()),
                           (pd.to_datetime('18:00:00').time(), pd.to_datetime('23:59:59').time())]
    weekend_time_ranges = [
        (pd.to_datetime('00:00:00').time(), pd.to_datetime('23:59:59').time())]

    # Applying discount factors based on time ranges
    df['discount_factor'] = np.where(df['start_day'].dt.weekday < 5,  # Weekday
                                     np.select(
                                         [df['start_time'].between(
                                             start, end) for start, end in weekday_time_ranges],
                                         [0.8, 1.2, 0.8]),
                                     # Weekend
                                     0.7)

    # Multiplying distance by discount factor for each vehicle type
    vehicle_types = ['moto', 'car', 'rv', 'bus', 'truck']
    for vehicle_type in vehicle_types:
        df[vehicle_type] = df[vehicle_type] * df['discount_factor']

    return df.drop(columns='discount_factor')
