# pip install praat-parselmouth

import pandas as pd
import parselmouth
from feature_extraction_utils import *
import wave, sys

def extract_content(path):

  !ffmpeg -i path -q:a 0 -map a /content/output.wav 
  sound = parselmouth.Sound('/content/output.wav')
  df = pd.DataFrame()

  attributes = {}
  intensity_second_wise = []
  pitch_second_wise = []

  intensity_attributes1 = get_intensity_attributes(sound,time_step=1,return_values=True)
  pitch_attributes1 = get_pitch_attributes(sound,time_step=1,return_values=True)

  intensity_attributes = intensity_attributes1[0]
  pitch_attributes = pitch_attributes1[0]

  intensity_attributes2 = intensity_attributes1[1]
  pitch_attributes2 = pitch_attributes1[1]

  attributes.update(intensity_attributes)
  attributes.update(pitch_attributes)

  intensity_second_wise.append(intensity_attributes2)
  pitch_second_wise.append(pitch_attributes2)

  hnr_attributes = get_harmonics_to_noise_ratio_attributes(sound,time_step=1)[0]
  gne_attributes = get_glottal_to_noise_ratio_attributes(sound)[0]
  attributes.update(hnr_attributes)
  attributes.update(gne_attributes)

  df['local_jitter'] = None
  df['local_shimmer'] = None
  df.at[0, 'local_jitter'] = get_local_jitter(sound)
  df.at[0, 'local_shimmer'] = get_local_shimmer(sound)

  spectrum_attributes = get_spectrum_attributes(sound)[0]
  attributes.update(spectrum_attributes)

  formant_attributes = get_formant_attributes(sound,time_step=1)[0]
  attributes.update(formant_attributes)

  lfcc_matrix, mfcc_matrix = get_lfcc(sound,time_step=1), get_mfcc(sound,time_step=1)
  df['lfcc'] = None
  df['mfcc'] = None
  df.at[0, 'lfcc'] = lfcc_matrix
  df.at[0, 'mfcc'] = mfcc_matrix

  delta_mfcc_matrix = get_delta(mfcc_matrix)
  delta_delta_mfcc_matrix = get_delta(delta_mfcc_matrix)
  df['delta_mfcc'] = None
  df['delta_delta_mfcc'] = None
  df.at[0, 'delta_mfcc'] = delta_mfcc_matrix
  df.at[0, 'delta_delta_mfcc'] = delta_delta_mfcc_matrix

  for attribute in attributes:
      df.at[0, attribute] = attributes[attribute]
      
  df.at[0, 'sound_filepath'] = sound_filepath
  rearranged_columns = df.columns.tolist()[-1:] + df.columns.tolist()[:-1]
  df = df[rearranged_columns]

  return df, intensity_second_wise, pitch_second_wise

extract_content("/content/youtube.mp4")
