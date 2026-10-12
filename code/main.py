import os
import time
from audio_identification import audioIdentification, fingerprintBuilder
from functions import evaluate_results


def main(database_path: str, database_fingerprints_path: str, queryset_path: str, 
    output_path: str, params: dict[str, int] | None = None) -> tuple[tuple[float, float, float, float], float]:
    '''Build fingerprints, run identification, and return metrics with processing time.'''
    params = params or {}
    fingerprintBuilder(database_path, database_fingerprints_path, params)
    start_time = time.perf_counter()    
    audioIdentification(queryset_path, database_fingerprints_path, output_path, params)
    identification_processing_time = time.perf_counter() - start_time
    metrics = evaluate_results(output_path)
    return metrics, identification_processing_time


if __name__ == "__main__":  
    database_path = os.path.join("database_recordings")
    database_fingerprints_path = os.path.join("fingerprints_database.json")
    queryset_path = os.path.join("query_recordings")
    output_path = os.path.join("output.txt")

    metrics, identification_processing_time = main(
        database_path, database_fingerprints_path,
        queryset_path, output_path
    )
    recall_rank_1, recall_rank_2, recall_rank_3, map_score = metrics 

    print(f"Mean Recall at Rank 1: {recall_rank_1:.4f}")
    print(f"Mean Recall at Rank 2: {recall_rank_2:.4f}")
    print(f"Mean Recall at Rank 3: {recall_rank_3:.4f}")
    print(f"Mean Average Precision: {map_score:.4f}")
    print(f"Identification processing time: {identification_processing_time:.2f} seconds\n")




