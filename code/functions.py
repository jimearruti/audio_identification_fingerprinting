import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import librosa
from skimage.feature import peak_local_max


def get_peak_coordinates(filename: str, time_frequency_representation: str = "STFT", window_length: int = 2048, 
                         hop_length: int = 512, min_distance: int = 10, threshold_rel: float = 0.05, 
                         n_fft: int = 4096, plot: bool = False) -> np.ndarray:
    '''Extracts peak coordinates from an audio file using a specified time-frequency representation and peak picking method.
    Args:
        filename (str): Path to the audio file.
        time_frequency_representation (str): Type of time-frequency representation to use ("STFT", or "Mel").
        window_length (int): Length of the window for STFT (ignored for CQT and Mel).
        hop_length (int): Hop length for the time-frequency representation.
        min_distance (int): Minimum distance between peaks.
        threshold_rel (float): Relative threshold for peak picking.
        n_fft (int): Number of FFT points.
        plot (bool): Whether to plot the spectrogram and constellation map.
    Returns:
        np.ndarray: Peak coordinates as (frequency_bin, time_bin) rows.
    '''
    # Load audio
    y, sr = librosa.load(os.path.join(filename), sr=22050)

    # Compute the time-frequency representation based on the specified type
    if time_frequency_representation == "STFT":
        # Compute STFT spectrogram
        spectrogram = np.abs(librosa.stft(y, n_fft=n_fft, window='hann', win_length=window_length, hop_length=hop_length))
    elif time_frequency_representation == "Mel":
        # Compute Mel spectrogram
        spectrogram = np.abs(librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, window='hann', win_length=window_length,
                                                            hop_length=hop_length))
    else:
        raise ValueError("Invalid time-frequency representation. Please choose 'STFT' or 'Mel'.")

    # Add a small constant to avoid log(0)
    spectrogram = spectrogram + 1e-10  

    # Detect peaks from spectrogram and plot constellation map
    coordinates = peak_local_max(np.log(spectrogram), min_distance=min_distance, threshold_rel=threshold_rel)

    if plot:
        # plot the spectrogram
        plt.figure(figsize=(10, 5))
        librosa.display.specshow(librosa.amplitude_to_db(spectrogram, ref=np.max), y_axis='linear', x_axis='time', 
                                 cmap='gray_r', sr=sr)

        # plot the constellation on top of the spectrogram
        plt.figure(figsize=(10, 5))
        librosa.display.specshow(librosa.amplitude_to_db(spectrogram, ref=np.max), y_axis='linear', x_axis='time', 
                                 cmap='gray_r', sr=sr)
        if time_frequency_representation == "STFT":
            plt.plot(librosa.frames_to_time(coordinates[:, 1], sr=sr, hop_length=hop_length), 
                     librosa.fft_frequencies(sr=sr, n_fft=n_fft)[coordinates[:, 0]], 'r.')
        elif time_frequency_representation == "Mel":
            plt.plot(librosa.frames_to_time(coordinates[:, 1], sr=sr, hop_length=hop_length), 
                     librosa.mel_frequencies(n_mels=spectrogram.shape[0])[coordinates[:, 0]], 'r.')
    return coordinates


def get_fingerprints(coordinates: np.ndarray, target_zone_time_start: int=0, target_zone_size_time: int=200,
                     target_zone_size_frequency: int=200, fan_out: int=30) -> list[tuple[int, int, int, int]]:
    ''' Generates fingerprints from peak coordinates by creating pairs of peaks within a defined target zone.
        Follows the algorithm described in the paper "An Industrial-Strength Audio Search Algorithm" by Wang.
    Args:        
        coordinates (numpy.ndarray): A 2D array of peak coordinates, where each row is (frequency_bin, time_bin).
        target_zone_time_start (int): Start time of the target zone.
        target_zone_size_time (int): Size of the target zone in time.
        target_zone_size_frequency (int): Size of the target zone in frequency.
        fan_out (int): Maximum number of fingerprints to generate per anchor.
    Returns:
        list[tuple[int, int, int, int]]: Fingerprints as
            (anchor_frequency, target_frequency, time_difference, anchor_time).
    '''
    fingerprints = []
    # Order by time
    coordinates_sorted = sorted(coordinates, key=lambda x: x[1])
    # iterate over all coordinate pairs
    for i_anchor, anchor in enumerate(coordinates_sorted):
        added_hashes = 0
        # only check the coordinates in the future wrt the anchor
        for target in coordinates_sorted[i_anchor + 1:]:
            df = abs(target[0] - anchor[0])
            dt = target[1] - anchor[1]
            if dt > target_zone_time_start + target_zone_size_time:
                # we're outside the target zone! stop checking this anchor
                break
            # check if the target is in the target zone
            if (df <= target_zone_size_frequency and dt >= target_zone_time_start and 
                dt <= target_zone_time_start + target_zone_size_time):
                # fingerprint is (anchor_frequency, target_frequency, time_difference, anchor_time)
                fingerprints.append((anchor[0], target[0], dt, anchor[1]))
                added_hashes += 1
                if added_hashes >= fan_out:
                    # if we've reached the max number of hashes per anchor, move on to the next one
                    break
    return fingerprints


def match_fingerprints(
    query_fingerprints: list[tuple[int, int, int, int]],
    db_hashes: dict[str, list[tuple[str, int]]]
) -> list[tuple[str, tuple[int, int]]]:
    '''Matches the fingerprints of a query audio file against a database of fingerprints and returns the best matches.
    Args:
        query_fingerprints (list): List of fingerprints for the query audio file, where each fingerprint is a tuple 
            of (anchor_frequency, target_frequency, time_difference, anchor_time).
        db_hashes (dict): Dictionary containing the fingerprint database, where 
            each key is a string representation of a fingerprint (anchor_frequency, target_frequency, time_difference) 
            and the value is a list of tuples (db_filename, db_anchor_time) for each matching fingerprint in the database.
    Returns:
        list[tuple[str, tuple[int, int]]]: A list of the top 3 best matches, where each match is a tuple of 
            (db_filename, (count, time_difference)), sorted by count in descending order.
    '''
    matches = {}
    for fingerprint in query_fingerprints:  
        # create hash key for the fingerprint (anchor_frequency, target_frequency, time_difference)
        fingerprint_key = str((int(fingerprint[0]), int(fingerprint[1]), int(fingerprint[2])))
        target_time = int(fingerprint[3])

        # check for coincidences in hashes and store the time differences for each one
        if fingerprint_key in db_hashes:
            for db_filename, db_anchor_time in db_hashes[fingerprint_key]:
                if db_filename not in matches:
                    matches[db_filename] = []
                matches[db_filename].append(int(db_anchor_time) - target_time)

    processed_matches = {}
    # get the most common time difference for the matches
    for db_filename, time_differences in matches.items():
        if not time_differences:
            continue
        min_diff = min(time_differences)
        max_diff = max(time_differences)
        if min_diff == max_diff:
            # the time differences are the same, let's just count the coincidences
            processed_matches[db_filename] = (len(time_differences), min_diff)
            continue
        # create a histogram of the time differences to find the most common one
        custom_bins = np.arange(min_diff, max_diff + 1, 1, dtype=int)
        h = np.histogram(time_differences, bins=custom_bins)
        # find where the peak is
        idx_max = h[0].argmax()
        # get the count at the peak
        max_count = h[0][idx_max]
        # get the time difference at the peak
        match_time = h[1][idx_max]
        # store the count and the time difference for the best match
        processed_matches[db_filename] = (max_count, match_time)
        
    # sort best matches by count and return the top 3  
    # each item of dictionary is (db_filename, (count, time_difference))  
    best_matches = sorted(processed_matches.items(), key=lambda x: x[1][0], reverse=True)[:3]
    return best_matches


def evaluate_results(results_path: str, style: str | None = None) -> tuple[float, float, float, float]:
    ''' Evaluates the identification results by calculating mean recall at ranks 1, 2, and 3, 
        as well as mean Average Precision (mAP).
    Args:
        results_path (str): Path to the text file containing the identification results.
        style (str, optional): If provided, filters the results to only include queries of a specific style before evaluation.
    Returns:
        tuple: A tuple containing mean recall at rank 1, mean recall at rank 2, mean recall at rank 3, 
            and mean Average Precision (mAP).
    ''' 

    # open the output file and read the results
    with open(results_path, 'r') as f:
        lines = f.readlines()
    lines = [line.split('\t') for line in lines]

    # create a dataframe from the results
    results = pd.DataFrame(lines, columns=['query_file', 'Match1', 'Match2', 'Match3'], )
    # remove newline character from Match3
    results['Match3'] = results['Match3'].str.replace('\n', '')
    # extract the target and style from the query file name
    results['target'] = results['query_file'].apply(lambda x: x.split('-')[0] + '.wav') 
    results['style'] = results['query_file'].apply(lambda x: x.split('.')[0])

    if style is not None:
        # check if style is valid
        if style not in results['style'].unique():
            raise ValueError(f"Style '{style}' not found in results. Available styles: {results['style'].unique()}")
        # filter per style
        results = results[results['style'] == style]

    # calculate chi at each rank (1 if the match is correct, 0 otherwise)
    results['chi_1'] = (results['Match1'] == results['target']).astype(int)
    results['chi_2'] = (results['Match2'] == results['target']).astype(int)
    results['chi_3'] = (results['Match3'] == results['target']).astype(int)

    # Calculate precision at each rank
    precision_rank_1 = results['chi_1']
    precision_rank_2 = (results['chi_1'] + results['chi_2']) / 2
    precision_rank_3 = (results['chi_1'] + results['chi_2'] + results['chi_3']) / 3

    # calculate recall at each rank 
    recall_rank_1 = results['chi_1']
    recall_rank_2 = (results['chi_1'] + results['chi_2'])
    recall_rank_3 = (results['chi_1'] + results['chi_2'] + results['chi_3'])

    # Calculate Average Precision (AP) per query 
    results['AP_query'] = ( (precision_rank_1 * results['chi_1']) + 
                            (precision_rank_2 * results['chi_2']) + 
                            (precision_rank_3 * results['chi_3']) )
    
    # Return mean recall at each rank and mean Average Precision (mAP) across all queries
    return recall_rank_1.mean(), recall_rank_2.mean(), recall_rank_3.mean(), results['AP_query'].mean()