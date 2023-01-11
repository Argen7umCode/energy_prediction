def get_sequences(data, window_lenght):
    sequences = []
    for i in range(len(data)-window_lenght):
        sequences.append([
            data[i:i+window_lenght],
            data[i+window_lenght:i+window_lenght+1]
        ])
    return sequences

