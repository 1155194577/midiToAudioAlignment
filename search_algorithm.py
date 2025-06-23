import numpy as np 
import pandas as pd
import mido
from tqdm import tqdm
from midi_processing import transform_note_event_interval,convert_note_event_intervals_to_midi_function,convert_midi_function_to_midi_contour,convert_midi_contour_to_pitch_contour
from util import compute_similarity_metric
    
# Helper function to validate events
def has_negative_times(events):
    """Check if any event has negative time values"""
    return any(event['start_time'] < 0 or event['end_time'] < 0 for event in events)

def brute_force_search(reference_contour, time, midi_events, search_space, search_record):
    best_similarity = float('-inf')
    best_parameters = None
    total = len(search_space['vertical']) * len(search_space['horizontal']) * len(search_space['scaling_factor'])
    progress_bar = tqdm(total=total, desc="Brute Force Search")
    
    for vertical_displacement in search_space['vertical']:
        for horizontal_displacement in search_space['horizontal']:
            for scaling_factor in search_space['scaling_factor']: 
                progress_bar.update(1)
                progress_bar.set_postfix({
                    'vertical': vertical_displacement,
                    'horizontal': horizontal_displacement,
                    'scaling': scaling_factor,
                    'best_sim': f"{best_similarity:.4f}"
                })
                
                transformed_note_event_interval = transform_note_event_interval(midi_events, vertical_displacement, horizontal_displacement, scaling_factor)
                if has_negative_times(transformed_note_event_interval):
                    continue
                transformed_midi_function = convert_note_event_intervals_to_midi_function(transformed_note_event_interval)
                transformed_midi_contour = convert_midi_function_to_midi_contour(transformed_midi_function, time)
                transformed_pitch_contour = convert_midi_contour_to_pitch_contour(transformed_midi_contour)
                cosine_similarity = compute_similarity_metric(reference_contour, transformed_pitch_contour, metric='cosine', binarise=False)
                binary_cosine_similarity = compute_similarity_metric(reference_contour, transformed_pitch_contour, metric='cosine', binarise=True)
                mse = compute_similarity_metric(reference_contour, transformed_pitch_contour, metric='mse', binarise=False)
                binary_mse = compute_similarity_metric(reference_contour,transformed_pitch_contour,metric='mse',binarise=True)
                mae = compute_similarity_metric(reference_contour,transformed_pitch_contour,metric="mae",binarise=False)
                binary_mae = compute_similarity_metric(reference_contour,transformed_pitch_contour,metric="mae",binarise=True)
                similarity = cosine_similarity
                # Record the search parameters and similarity
                search_record.append((vertical_displacement, horizontal_displacement, scaling_factor, cosine_similarity,binary_cosine_similarity,mse, binary_mse,mae,binary_mae))
                if similarity > best_similarity:
                    improvement = similarity - best_similarity
                    best_similarity = similarity
                    # progress_bar.write(f"Best similarity updated! {best_similarity:.4f} improvement: {improvement:.4f} parameter ({vertical_displacement}, {horizontal_displacement}, {scaling_factor})")
                    best_parameters = (vertical_displacement, horizontal_displacement, scaling_factor)
                    
    progress_bar.close()
    return best_parameters, best_similarity


