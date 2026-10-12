import os
import json
from tqdm import tqdm
from functions import get_peak_coordinates, get_fingerprints, match_fingerprints


def fingerprintBuilder(
    database_path: str,
    fingerprints_path: str,
    params: dict[str, int] | None = None
) -> None:
    '''Builds a fingerprint database from audio files in the specified directory and saves it as a JSON file.
    Args:
        database_path (str): Path to the directory containing the audio files for the database.
        fingerprints_path (str): Path where the generated fingerprint JSON file will be saved.
        params (dict[str, int] | None): Dictionary containing the parameters for fingerprint generation.
            - window_length (int): Length of the window for peak detection (default: 2048).
            - n_fft (int): Number of FFT points for peak detection (default: 4096).
            - hop_length (int): Hop length for peak detection (default: 512).
            - target_zone_time_start (int): Start time of the target zone in milliseconds (default: 0).
            - target_zone_size_time (int): Size of the target zone in time in milliseconds (default: 200).
            - target_zone_size_frequency (int): Size of the target zone in frequency in Hz (default: 200).
            - fan_out (int): Number of target points to consider for each anchor point (default: 30).
    '''
    params = params or {}

    # create a dictionary to store the fingerprints for each audio file in the database
    db_hashes = {}

    window_length = params.get('window_length', 2048)
    n_fft = params.get('n_fft', 4096)
    hop_length = params.get('hop_length', 512)
    target_zone_time_start = params.get('target_zone_time_start', 0)
    target_zone_size_time = params.get('target_zone_size_time', 200)
    target_zone_size_frequency = params.get('target_zone_size_frequency', 200)
    fan_out = params.get('fan_out', 30)

    print("Building fingerprint database...")
    for filename in tqdm(os.listdir(database_path)):
        # get the peak coordinates for the audio file
        coordinates = get_peak_coordinates(os.path.join(database_path, filename),
                                           window_length=window_length, n_fft=n_fft, hop_length=hop_length)
        # generate fingerprints for the audio file
        fingerprints = get_fingerprints(coordinates, 
                                        target_zone_time_start=target_zone_time_start, 
                                        target_zone_size_time=target_zone_size_time, 
                                        target_zone_size_frequency=target_zone_size_frequency, 
                                        fan_out=fan_out)

        for fingerprint in fingerprints:
            # create hash key for the fingerprint (anchor_frequency, target_frequency, time_difference)
            fingerprint_key = str((int(fingerprint[0]), int(fingerprint[1]), int(fingerprint[2])))
            if fingerprint_key not in db_hashes:
                db_hashes[fingerprint_key] = []
            # store the filename and anchor time for each fingerprint
            db_hashes[fingerprint_key].append((filename, int(fingerprint[3])))

    # save in a json file
    with open(fingerprints_path, 'w') as f:
        json.dump(db_hashes, f)


def audioIdentification(
    queryset_path: str,
    fingerprints_path: str,
    output_path: str,
    params: dict[str, int] | None = None
) -> None:
    '''Identifies audio files in the queryset by comparing their fingerprints to a database of
        fingerprints and saves the results to a text file.
    Args:
        queryset_path (str): Path to the directory containing the query audio files.
        fingerprints_path (str): Path to the JSON file containing the fingerprint database.
        output_path (str): Path where the identification results will be saved as a text file.
        params (dict[str, int] | None): Dictionary containing the parameters for fingerprint generation.
            - window_length (int): Length of the window for peak detection (default: 2048).
            - n_fft (int): Number of FFT points for peak detection (default: 4096).
            - hop_length (int): Hop length for peak detection (default: 512).
            - target_zone_time_start (int): Start time of the target zone in milliseconds (default: 0).
            - target_zone_size_time (int): Size of the target zone in time in milliseconds (default: 200).
            - target_zone_size_frequency (int): Size of the target zone in frequency in Hz (default: 200).
            - fan_out (int): Number of target points to consider for each anchor point (default: 30).
    '''
    params = params or {}

    # load the fingerprint database
    with open(fingerprints_path, 'r') as f:
        db_hashes = json.load(f)

    # get parameters
    window_length = params.get('window_length', 2048)
    n_fft = params.get('n_fft', 4096)
    hop_length = params.get('hop_length', 512)
    target_zone_time_start = params.get('target_zone_time_start', 0)
    target_zone_size_time = params.get('target_zone_size_time', 200)
    target_zone_size_frequency = params.get('target_zone_size_frequency', 200)
    fan_out = params.get('fan_out', 30)

    with open(output_path, 'w') as f:
        for query_filename in tqdm(os.listdir(queryset_path)):
            # get the peak coordinates for the query file
            coordinates_query = get_peak_coordinates(os.path.join(queryset_path, query_filename),
                                                    window_length=window_length, n_fft=n_fft, hop_length=hop_length)
            # generate fingerprints for the query file
            query_fingerprints = get_fingerprints(coordinates_query, target_zone_time_start=target_zone_time_start, target_zone_size_time=target_zone_size_time, 
                                        target_zone_size_frequency=target_zone_size_frequency, fan_out=fan_out)

            # match the query fingerprints with the database fingerprints and get the best matches
            best_matches = match_fingerprints(query_fingerprints, db_hashes)
            
            # write the results to the output file
            f.write("\t".join([query_filename] + [db_filename for db_filename, _ in best_matches]) + "\n")