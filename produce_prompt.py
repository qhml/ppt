"""
This script is used to generate prompt for elevator data.
Generated prompts includes fields about:
    - capacity
    - loading time
    - unloading time
    - mass
"""
import pandas as pd
from random import sample


def process_passenger_properties(fname):
    fname = "data/LunchPeakPassengerProfiles/Four_mass_capacity_loading_unloading/4_mass_capacity_loading_unloading(CIBSE-office-LunchPeak)0.txt"
    pdf = pd.read_csv(fname, header=None)
    pdf.columns = ["arrival_time", "arrival_floor", "destination_floor", "mass", "capacity", "loading_time",
                   "unloading_time", "placeholder", "placeholder"]
    pdf = pdf[["arrival_time", "arrival_floor", "destination_floor", "mass", "capacity", "loading_time",
               "unloading_time", ]]
    return pdf


def mask_passenger_property(property_names, properties, mask_value=-1, strategy="random"):
    """mask certain field in the passenger properties data (currently using random strategy)"""
    masked_property = sample(property_names, 1)[0]
    original_value = properties[masked_property]
    properties[masked_property] = mask_value
    return properties, masked_property, original_value


def prepare_prompt(pdf: pd.DataFrame()):
    """iterate through all samples and mask certain fields according to some stategy"""
    passenger_properties, masked_properties, original_values = [], [], []
    all_property_names = pdf.columns.tolist()
    for idx, row in pdf.iterrows():
        properties, masked_property, original_value = mask_passenger_property(all_property_names, row)
        passenger_properties.append(properties.tolist())
        masked_properties.append(masked_property)
        original_values.append(original_value)
    return passenger_properties, masked_properties, original_values


if __name__ == '__main__':
    fname = "data/LunchPeakPassengerProfiles/Four_mass_capacity_loading_unloading/4_mass_capacity_loading_unloading(CIBSE-office-LunchPeak)0.txt"
    pdf = process_passenger_properties(fname)
    prompts, masekd_properties, orgiinal_values = prepare_prompt(pdf)
    print(prompts[:10])
    # read file
    # process data: 1. mask target field 2. get labels
    # save files (pkl)
