#
# Functions to get, load, clean, split and pickle data.
#

import os

import pandas as pd
import wget


#
# Get data
#

def get_data(dl_urls, data_dir):
    '''Download data files in csv format.

    Parameters
    ----------
    dl_urls : dict of str: str
        Dictionary of destination filenames and URLs that point directly
        to data files to be downloaded.
    data_dir : str
        Location of destination directory for downloaded data files.

    Returns
    -------
    df : pd.DataFrame
        Data Frame that displays the file name and file size of
        downloaded data files.
    '''
    report = {}
    for f, url in dl_urls.items():
        path = os.path.join(data_dir, f)
        if not os.path.isfile(path):
            if not os.path.isdir(data_dir):
                os.mkdir(data_dir)
            try:
                wget.download(url=url, out=path)
            except Exception as e:
                print(e)
                raise
        file_size_mb = round(os.path.getsize(path) / (1024 ** 2), 2)
        report[f] = file_size_mb
    return pd.DataFrame.from_dict(
        report,
        orient='index',
        columns=['File Size in MB'],
    )


#
# Load data
#

def read_data(data_title, data_dir, data_file):
    '''Read data from CSV or GEOJSON files into data frames.

    Output dtypes, non-null values and memory usage information on data
    frames. Store dataframe memory usage for comparison later.

    Parameters
    ----------
    data_title : str
        Brief title that describes the nature of the data.
    data_file : str
        File name of data file.
    data_dir : str
        Directory where data files are located.

    Returns
    -------
    df : pd.DataFrame or GeoDataFrame
        DataFrame of CSV data or GeoDataFrame of GEOJSON data.
    '''
    ext = data_file.split(".")[-1].lower()
    file_path = os.path.join(data_dir, data_file)
    if ext == 'csv':
        try:
            df = pd.read_csv(file_path, sep=',', header=0)
        except IOError as e:
            print(e)
            raise
    elif ext == 'pkl':
        try:
            df = pd.read_pickle(file_path)
        except IOError as e:
            print(e)
            raise
    else:
        raise Exception('Data file extension `{}` is the incorrect file type. '
                        'Use .csv and .pkl files only.'.format(ext))
    df.columns = (df
                  .columns
                  .str.strip()
                  .str.lower()
                  .str.replace(' ', '_'))
    print('{}'.format(data_title))
    print(df.info(memory_usage='deep'))
    print('\n')

    return df


#
# Clean data
#

def sample_size(df):
    '''Limit the number of observations to display using the sample
    method.

    If the number of samples to display is more than the number of rows
    in the data frame, an error will occur.

    Parameters
    ----------
    df : pd.DataFrame
        Data Frame that samples will be drawn from.

    Returns
    -------
    sample: int
        The number of samples to display.
    '''
    if df.shape[0] < 3:
        sample = df.shape[0]
    else:
        sample = 3
    return sample


def review_data(df):
    '''Pretty print the data frame head, tail, samples and nulls.

    Relies on the IPython.display setting in the Settings sections to
    pretty print all data frames, not just the last data frame in a
    Jupyter cell.

    Parameters
    ----------
    df : pd.DataFrame
        Data Frame that will be displayed.

    Returns
    -------
    None
    '''
    sample = sample_size(df)

    display(df.head(sample))
    display(df.tail(sample))
    display(df.sample(sample))
    df_null = df.isnull().sum().to_frame(name='is_null')
    display(df_null)


def memory_usage(df):
    '''Return memory usage of data frame in megabytes (MB).

    Parameters
    ----------
    df : pd.DataFrame
        Data Frame that will be analyzed for memory usage.

    Returns
    -------
    Float that holds the memory usage of the data frame in megabytes
    (MB).
    '''
    return round(df.memory_usage(deep=True).sum() / (1024 ** 2), 2)


def optimize_df(df, cols_convert, cols_order=None):
    '''Convert columns to optimal data type and display memory usage.

    Parameters
    ----------
    df : pd.DataFrame
        Data Frame that will be optimized.
    cols_convert : dict of str: data type
        Dictionary of columns names as keys and data types as values.
    cols_order : list of str, optional
        Ordered list of column names to reorder columns in data frame.

    Returns
    -------
    None
    '''
    print(df.info(memory_usage='deep'))
    mem_before = memory_usage(df)
    for col, dtype in cols_convert.items():
        df[col] = df[col].astype(dtype)
    if cols_order:
        df = df[cols_order]
    mem_after = memory_usage(df)

    print(df.info(memory_usage='deep'))
    print('*' * 20)
    print('Memory usage before: {} MB'.format(mem_before))
    print('Memory usage after: {} MB'.format(mem_after))

    sample = sample_size(df)
    display(df.sample(sample))


#
# Split data
#

def split_train_test(df, col, no_dupe, date_lt='2017-11-01',
                     date_gt='2017-10-01'):
    '''Split datset into training and test sets.

    Parameters
    ----------
    df : pd.DataFrame
        Data Frame of raw values that will be grouped into counts.
    col : str
        Datetime column name for either opened or closed requests.
    no_dupe : boolean
        Specifies whether or not to exclude duplicated Service Request
        Numbers.
    date_lt : str
        Cut off date for training set.
    date_gt : str
        Cut off date for test set.

    Returns
    -------
    df_train : pd.DataFrame
        Data Frame of data points for training set.
    df_test : pd.DataFrame
        Data Frame of data points for test set.
    '''
    mask = df[col].notnull()
    if no_dupe:
        mask = (mask) & ~(df['is_duplicate'])
    df_ct = (df[mask]
             .set_index(col)
             .resample('MS')
             .size()
             .to_frame('count'))
    # Drop any dates after October 2018
    date_cutoff = '2018-11-01'
    return df_ct[
               (df_ct.index < date_lt) &
               (df_ct.index < date_cutoff)
               ].copy(), df_ct[
               (df_ct.index > date_gt) &
               (df_ct.index < date_cutoff)
               ].copy()


#
# Pickle data
#

def export_pickle(filename, df, data_dir):
    '''Pickle data frames to preserve attributes.

    Parameters
    ----------
    filename : str
        Filename for output file.
    df : DataFrame or GeoDataFrame
        DataFrame or GeoDataFrame to be pickled.

    Returns
    -------
    None
    '''
    pkl_file = os.path.join(data_dir, '{}.pkl'.format(filename))
    df.to_pickle(pkl_file)
    print('{}: {}'.format(filename, df.shape))