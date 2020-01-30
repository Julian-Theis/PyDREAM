import datetime

def time_delta_seconds(ts_start, ts_end):
    if not isinstance(ts_start, datetime.datetime) or not isinstance(ts_end, datetime.datetime):
        raise ValueError('The timestamps are not of <class \'datetime.datetime\'>')
    return (ts_end - ts_start).total_seconds()