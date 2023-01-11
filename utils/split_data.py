def split_data(data, spliter_count):
    spliter_count = int(len(data) * spliter_count)
    return data[:spliter_count], data[spliter_count:]