import os
from dataclasses import dataclass, field
from typing import Any, List, Union
import json
from collections import defaultdict

from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from utils import log
import h3

class VocabDictionary(object):
    """
    dictionary to map trajectory semantics to tokens
    """

    def __init__(self, vocab_file_path) -> None:

        with open(vocab_file_path, "r", encoding="utf-8") as f:
            self.vocab = json.load(f)

        self.reverse_map_vocab = {value:item for item, value in self.vocab.items()}

    def __len__(self):
        return len(self.vocab)

    def encode(self, trajectory:Union[List[str], List[int]]):
        """
        encode a trajectory into token
        """
        tokens = []
        # Get the integer ID for padding once to handle unknown tokens
        pad_id = int(self.vocab[self.pad()]) # Assuming PAD value can be cast to int

        for item in trajectory:
            key = str(item) # Ensure the key is treated as a string for lookup
            if key in self.vocab:
                try:
                    # Explicitly cast the retrieved vocabulary value to an integer
                    tokens.append(int(self.vocab[key]))
                except ValueError:
                    # Handle case where vocab value isn't a valid integer string
                    print(f"Warning: Vocab value for key '{key}' is not an integer: {self.vocab[key]}. Using PAD token.")
                    tokens.append(pad_id)
            else:
                # Handle unknown tokens - use integer PAD ID
                # You could also log this: print(f"Unknown token: {key}")
                tokens.append(pad_id)

        return tokens
    
    def decode(self, tokens:List[int]):
        """
        decode a trajectory into token
        """
        trajectory = []
        for token in tokens:
            trajectory.append(self.reverse_map_vocab[token])

        return trajectory

    def pad(self):
        return "PAD"
    def eot(self):
            return "EOT"
    def pad_token(self):
        return self.vocab[self.pad()]
    def eot_token(self):
        return self.vocab[self.eot()]


@dataclass
class POLConfig:
    """
    dataclass for semantic trajectory
    """
    data_dir: str = "./data/work-outliers/checkin-atl"
    file_name: str = "data"
    features: List = field(default_factory=lambda: ["gps", "distance", "duration", "place"])
    block_size: int = 64 # length of maximum daily trajectories of a user
    grid_leng: int = 25 # the size of a cell in a grid
    include_outliers: bool = False
    outlier_days: int= 14
    # include_outliers: bool= False
    log_file: str = ""
    logging: bool = True
    start_time: int= 4

class POLDataset(Dataset):
    """
    semantic trajectory datset
    """
    def __init__(self, config: POLConfig) -> None:
        super().__init__()

        self.config = config

        dictionary_path = self.get_dictionary_path()

        self.dictionary = VocabDictionary(dictionary_path)
        file_path = os.path.join(self.config.data_dir, f"{self.config.file_name}_grouped.tsv")

        self.data, self.outliers = self.get_data(file_path)

        # pdb.set_trace()
        
    def get_data(self, file_path):
        """
        get all the data
        """

        message=f"loading the data..."
        if self.config.logging:
            log(message, self.config.log_file)
        else:
            print(message)

        data = pd.read_csv(file_path, delimiter="\t")
        
        data["date_formated"] = pd.to_datetime(data["date"])
       
        outlier_list = [546, 644, 347, 62, 551, 992, 554, 949, 900, 57] #TO DO can load this data from a file

        message=f"outliers: {outlier_list}"
        if self.config.logging:
            log(message, self.config.log_file)
        else:
            print(message)

        data["user_id_int"] = data["user_id"].str.split("_").str[-1].astype(int)
        # grouped = data.groupby(by=["user_id", "date"]).agg(list).reset_index()
        outliers = data[(data["date_formated"] > (data["date_formated"].max() - pd.DateOffset(self.config.outlier_days)))].copy()
        outliers = outliers.loc[outliers['user_id_int'].isin(outlier_list)][["user_id_int", "date"]]
        outliers["outlier"] = "outlier"

        data= pd.merge(data, outliers,  how='left', left_on=['user_id_int','date'], right_on = ['user_id_int','date'])
        data['outlier'] = np.where(data.outlier.notnull(), "outlier", "non outlier")

        if not self.config.include_outliers:
            data = data[data["outlier"] != "outlier"]

        message=f"inlcude outliers: {self.config.include_outliers}"

        if self.config.logging:
            log(message, self.config.log_file)
        else:
            print(message)

        # times len(self.config.features) because we may have more than one feature to include in the vector
        # plus 3 to account for the user_id, dayofweek and EOT
        self.config.block_size = (data.token.str.len().max() * len(self.config.features)) + 3

        message=f"context size: {self.config.block_size }"

        if self.config.logging:
            log(message, self.config.log_file)
        else:
            print(message)

        return data, outliers
    
    def get_dictionary_path(self):
        """get vocab file name"""
        file_name = "vocab"
        if "gps" in self.config.features:
            file_name += "_gps"
        if "distance" in self.config.features:
            file_name += "_distance"
        if "duration" in self.config.features:
            file_name += "_duration"
        if "place" in self.config.features:
            file_name += "_place"

        file_name += ".json"

        path = os.path.join(self.config.data_dir, file_name)

        return path
    def get_outliers(self):
        return self.outliers
    
    def partition_dataset(self, proportion=0.9, seed=123):
        np.random.seed(seed)
        train_num = int(len(self) * proportion)
        indices = np.random.permutation(len(self))
        train_indices, val_indices = indices[:train_num], indices[train_num:]
        return train_indices, val_indices
     
    def __len__(self):
        return len(self.data)
    
    def get_samples_for_user(self, user_id):
        """
        get the samples of a particular user given their user_id
        """
        indices = self.data[self.data["user_id"] == f"user_{user_id}"].index.tolist()
        samples = []
        for i in indices:
            samples.append(self.__getitem__(i))

        return samples

    def get_feature_vector(self, sample:pd.core.series.Series):
        """
        generate a feature vector for a given sample
        """

        daily_trajectory_feature = [sample["user_id"], eval(sample["dayofweek"])[0]]

        # [user_id, dayofweek, place, token, duration_bucket, distance, place, token, duration_bucket, distance, ..., EOT]
        # [user_35, day_4, Workplace, 88, 0-60, near, Restaurant, 88, 0-60, near, ..., EOT]

        places = eval(sample.place)
        tokens = eval(sample.token)
        duration_buckets = eval(sample.duration_bucket)
        distances = eval(sample.distance_label)

        for place, token, duration_bucket, distance in zip(places, tokens, duration_buckets, distances):
            current_seq = []

            if "place" in self.config.features:
                current_seq.append(place)

            if "gps" in self.config.features:
                current_seq.append(str(token))

            if "duration" in self.config.features:
                current_seq.append(duration_bucket)

            if "distance" in self.config.features:
                current_seq.append(distance)
            
            daily_trajectory_feature.extend(current_seq)
        
        daily_trajectory_feature.append("EOT")
        return daily_trajectory_feature
    
    def get_all_data(self):

        data = defaultdict(list)
        for i in tqdm(range(self.data.shape[0])):

            sample = self.data.iloc[i]
            daily_trajectory_feature = self.get_feature_vector(sample)

            data["user_id_int"].append(sample.user_id_int)
            data["date"].append(sample.date)
            data["feature"].append(daily_trajectory_feature)

            # pdb.set_trace()

        file_name = "data_with_features"
        if "gps" in self.config.features:
            file_name += "_gps"
        if "distance" in self.config.features:
            file_name += "_distance"
        if "duration" in self.config.features:
            file_name += "_duration"
        if "place" in self.config.features:
            file_name += "_place"

        file_name += ".tsv"

        data_df = pd.DataFrame(data)
        data_df.to_csv(f"{self.config.data_dir}/{file_name}", sep="\t", index=False)

    def __getitem__(self, index) -> Any:

        # pdb.set_trace()
        sample  = self.data.iloc[index]
        daily_trajectory_feature = self.get_feature_vector(sample)   
        tokens = self.dictionary.encode(daily_trajectory_feature)
        metadata = [sample.user_id, sample.date, sample.outlier]

        # ipdb.set_trace()
        return (metadata,  tokens)
    
    def collate(self, data):
        """
        collate function
        """
        # max_length = len(max(token_lists, key=len))
        masks = []
        token_lists = []
        all_metadata = []
        
        # start_time = time.time()
        max_lenth = max([len(item[-1]) for item in data])
        for metadata, tokens in data:

            mask = [1] * len(tokens) + [0] * (max_lenth - len(tokens))
            tokens_ = tokens +  (max_lenth - len(tokens)) * [self.dictionary.pad_token()]
            token_lists.append(tokens_)
            masks.append(mask)
            all_metadata.append(metadata)

        token_lists = torch.tensor(token_lists)
        masks = torch.tensor(masks)

        # print(f"to batchify it took {time.time() - start_time}")
        return {
            "data" : token_lists,
            "mask": masks,
            "metadata": all_metadata
        }
  

@dataclass
class PortoConfig:
    """
    dataclass for the porto taxi dataset
    """
    data_dir: str = "./data/porto"
    file_name: str = "porto_processed"
    grip_size: List = field(default_factory=lambda: (51, 158))
    data_split: str = None
    block_size: int = 1186 # length of maximum trajectory
    outlier_level: int = 3
    outlier_prob: float = 0.3
    outlier_ratio: float = 0.05
    outliers_list: List = field(default_factory=lambda: ["route_switch"])
    outliers_dir: str = "./data/porto/outliers"
    include_outliers: bool = True

class PortoDataset(Dataset):
    """
    semantic trajectory datset
    """
    def __init__(self, config: PortoConfig) -> None:
        super().__init__()

        self.config = config
        dictionary_path = os.path.join(self.config.data_dir, "vocab.json")
        self.dictionary = VocabDictionary(dictionary_path)
        file_path = os.path.join(self.config.data_dir, f"{self.config.file_name}.csv")

        self.data, self.metadata = self.get_data(file_path)
        # pdb.set_trace()
        
    def get_data(self, file_path):
        """
        get all the data
        """

        print(f"loading the dataset ...")
        # pdb.set_trace()
        trajectories = []
        labels = []
        sizes = []
        i = 0
        for traj in tqdm(open(file_path, 'r').readlines()):
            traj = eval(traj)
            trajectories.append(traj)
            labels.append("non outlier")
            sizes.append(len(traj))

            # if i > 2000:
            #     break
            # i+=1
        
        sizes = np.array(sizes)
        self.config.block_size = sizes.max() + 2 # to account for EOT and SOT 
        # pdb.set_trace()
        outlier_counts = 0
        skipped_long_trajectories = 0
        if self.config.include_outliers:
            print("loading outliers")
            # add outliers
            for key, values in self.get_outliers().items():
                label =""
                if key == "route_switch":
                    label = "route switch outlier"
                elif key == "detour":
                    label =  "detour outlier"

                for traj in values:

                    if len(traj) <= self.config.block_size - 2:
                        trajectories.append(traj)
                        labels.append(label)
                        outlier_counts += 1
                    else:
                        skipped_long_trajectories += 1

        # sizes.sort()
        
        print(f"total number of outliers: {outlier_counts}")
        print(f"number of spkipped trajectories: {skipped_long_trajectories}")
        print(f"context size {self.config.block_size}")

        # pdb.set_trace()
        sorted([trajectories], key=lambda k: len(k))
        return trajectories, labels
    def get_outliers(self):
        """
        load saved outliers
        """
        outliers = {}
        
        for outlier_type in self.config.outliers_list:
            # Construct the full path using the outliers_dir and the outlier_type (which is the base filename)
            file = os.path.join(self.config.outliers_dir, f"{outlier_type}.csv")

            try:
                # Use 'with open' for safer file handling
                with open(file, 'r') as f:
                    outlier_trajectories = f.readlines()
                outliers[outlier_type] = [eval(traj) for traj in outlier_trajectories]
            except FileNotFoundError:
                # More specific exception handling
                print(f"Error: Outlier file not found at {file}")
                # Decide how to handle this: skip, raise an error, etc.
                # For now, let's re-raise to make the issue clear
                raise Exception(f"The file {file} cannot be found. Check the --outliers_dir argument and file names.")
            except Exception as e:
                # Catch other potential errors during file reading or eval
                print(f"Error processing file {file}: {e}")
                raise e  # Re-raise the caught exception
            
            print(f"loaded {outlier_type} outliers from {file}")
        return outliers
        
    def generate_outliers(self):
        """Generate outliers with enhanced abnormal patterns"""
        outliers = {}
        trajectory_count = len(self)

        # Select longer trajectories for outlier generation
        long_trajectories = []
        min_length = 40  # Minimum length for outliers

        for idx in range(trajectory_count):
            # Ensure data point is a list or tuple before checking length
            if isinstance(self.data[idx], (list, tuple)) and len(self.data[idx]) >= min_length:
                long_trajectories.append(idx)

        print(f"Found {len(long_trajectories)} long trajectories out of {trajectory_count}")

        if len(long_trajectories) > 0:
            # Use longer trajectories for outlier generation
            np.random.seed(0)
            route_switching_idx = np.random.choice(long_trajectories,
                                                  size=min(int(trajectory_count * self.config.outlier_ratio),
                                                         len(long_trajectories)), replace=False) # Added replace=False

            # Standard route switching outliers
            # Use a moderate level for k-ring distance (e.g., level maps to k=2 or 3)
            outliers["route_switch"] = self.get_route_switch_outliers(
                [self.data[idx] for idx in route_switching_idx],
                level=3, # Corresponds to k=3 in new _perturb_point
                prob=self.config.outlier_prob)

            # Detour outliers
            np.random.seed(10)
            detour_idx = np.random.choice(long_trajectories,
                                        size=min(int(trajectory_count * self.config.outlier_ratio),
                                               len(long_trajectories)), replace=False) # Added replace=False

            outliers["detour"] = self.get_detour_outliers(
                [self.data[idx] for idx in detour_idx],
                level=2, # Corresponds to k=2 in new _perturb_point
                prob=self.config.outlier_prob * 1.2) # Slightly adjust prob if needed

            # Sharp turn outliers
            np.random.seed(20)
            sharp_turn_idx = np.random.choice(long_trajectories,
                                            size=min(int(trajectory_count * self.config.outlier_ratio),
                                                   len(long_trajectories)), replace=False) # Added replace=False

            # Sharp turn logic uses coordinate manipulation, level controls magnitude
            outliers["sharp_turn"] = self.get_sharp_turn_outliers(
                [self.data[idx] for idx in sharp_turn_idx],
                level=self.config.outlier_level) # Use original level or adjust as needed
        else:
            print("Warning: Not enough long trajectories found to generate outliers.")
            # Fallback might not be useful if base trajectories are too short

        # Save the outliers
        save_dir = f"{self.config.data_dir}/outliers"
        os.makedirs(save_dir, exist_ok=True)
        for key, values in outliers.items():
            # Check if any outliers were actually generated for this type
            if not values:
                 print(f"Warning: No '{key}' outliers were generated for config ratio={self.config.outlier_ratio}, level={self.config.outlier_level}, prob={self.config.outlier_prob}.")
                 continue # Skip saving empty file if desired, or save empty below

            current_save_dir = \
                f"{save_dir}/{key}_ratio_{self.config.outlier_ratio}_level_{self.config.outlier_level}_prob_{self.config.outlier_prob}.csv"

            print(f"Saving outlier file {current_save_dir} with {len(values)} outliers")

            try:
                with open(current_save_dir, "w") as fout:
                    for traj in values:
                        # Ensure traj is written in a way that eval() can read back (list format)
                        fout.write(f"{traj}\n")
            except Exception as e:
                 print(f"Error writing outlier file {current_save_dir}: {e}")

        return outliers

    def get_route_switch_outliers(self, batch_x, level, prob):
        """
        Get route switching outliers using the new _perturb_point.
        """
        outliers = []
        for traj in batch_x:
            if len(traj) < 3: continue # Need at least start, middle, end

            perturbed_middle = []
            for p in traj[1:-1]: # Perturb points between start and end
                # Perturb based on probability
                if np.random.random() < prob:
                     perturbed_p = self._perturb_point(p, level)
                     perturbed_middle.append(perturbed_p)
                else:
                     perturbed_middle.append(p) # Keep original point

            # Construct the new trajectory, ensuring start and end are kept
            new_traj = [traj[0]] + perturbed_middle + [traj[-1]]
            outliers.append(new_traj)
        return outliers

    def _perturb_point(self, point, level):
        """
        Perturb an H3 index by moving to a nearby hexagon within k-ring `level`.
        Ensures the returned point is different from the original, if possible.
        """
        try:
            # Ensure the input is a valid H3 index string
            if not isinstance(point, str) or not h3.is_valid_cell(point):
                # print(f"Warning: Invalid H3 index '{point}' passed to _perturb_point. Returning original.")
                return point

            # k-ring distance k should be at least 1 to find neighbors
            # Cap level reasonably to avoid excessive computation or jumping too far
            k = max(1, min(int(level), 5)) # Let level map to k, capped at 5

            # Get neighbors within k rings (use grid_disk for efficiency)
            # Explicitly ensure neighbors is a set before discarding
            neighbors = set(h3.grid_disk(point, k))
            neighbors.discard(point) # Remove the original point

            if not neighbors:
                # This should be rare, but if a point has no neighbors within k rings
                # print(f"Warning: No neighbors found for {point} within k={k}. Returning original.")
                return point

            # Randomly choose one of the neighbors
            perturbed_point = np.random.choice(list(neighbors))
            return perturbed_point

        except Exception as e:
            print(f"Error during _perturb_point for H3 index '{point}' with level {level}: {e}")
            return point # Return original point on error

    def get_detour_outliers(self, batch_x, level, prob, vary=False):
        """Generate detour outliers for H3 indices"""
        outliers = []

        # Simplified: level variation might be less meaningful now, prob controls length
        # if vary:
        #     level += np.random.randint(-2, 3)
        #     # ... prob variation ...

        for traj in batch_x:
            if len(traj) < 5: # Need at least a few points for a detour
                continue

            # Determine the part of the trajectory to perturb
            # Ensure anomaly_len is at least 1 if prob > 0
            anomaly_len = max(1, int((len(traj) - 2) * prob))
            if len(traj) <= anomaly_len + 2:  # Ensure valid indices
                # If prob is high, perturb most of the trajectory except start/end
                anomaly_len = len(traj) - 2

            if anomaly_len == 0: continue # Cannot perturb 0 points

            anomaly_st_loc = np.random.randint(1, len(traj) - anomaly_len)
            anomaly_ed_loc = anomaly_st_loc + anomaly_len

            # Create the outlier trajectory by perturbing each point in the segment
            # The new _perturb_point handles finding a random neighbor
            perturbed_segment = []
            for p in traj[anomaly_st_loc:anomaly_ed_loc]:
                 perturbed_segment.append(self._perturb_point(p, level))

            outliers.append(traj[:anomaly_st_loc] + perturbed_segment + traj[anomaly_ed_loc:])

        return outliers

    def get_sharp_turn_outliers(self, batch_x, level=5, min_length=40):
        """Generate outliers with abnormally sharp turns for longer trajectories"""
        outliers = []
        # target_resolution = 9 # Define the target resolution (resolution taken from cell)

        for traj in batch_x:
            # Only process longer trajectories
            if len(traj) < min_length:
                continue

            # Create a copy of the trajectory
            new_traj = list(traj)
            modified = False # Flag to check if modification happened

            # Select 2-4 points for sharp turns
            num_turns = np.random.randint(2, 5)
            # Avoid first and last few points
            margin = 5
            if len(traj) < 2 * margin + num_turns:
                continue # Not enough points to make safe turns

            possible_indices = list(range(margin, len(traj)-margin))
            if len(possible_indices) < num_turns:
                continue

            turn_indices = sorted(np.random.choice(possible_indices, num_turns, replace=False))

            indices_offset = 0 # Keep track of index changes due to insertions
            for idx_orig in turn_indices:
                idx = idx_orig + indices_offset # Adjust index based on previous insertions

                # Ensure index is valid after potential insertions
                # Need idx+1 to exist for midpoint insertion logic
                if idx >= len(new_traj) -1:
                    continue

                try:
                    current_cell = new_traj[idx]
                    # Ensure valid H3 cell before proceeding
                    if not isinstance(current_cell, str) or not h3.is_valid_cell(current_cell):
                        continue

                    # Get coordinates for the current point
                    lat, lng = h3.cell_to_latlng(current_cell)
                    res = h3.get_resolution(current_cell) # Get current resolution

                    # Create a significant deviation
                    # Higher level = sharper turn. Use a larger multiplier
                    # Use a slightly larger scale for sharper turns compared to generic perturbation
                    offset_scale = 0.01 # Adjusted scale for noticeable turns
                    offset_lat = (np.random.random() - 0.5) * offset_scale * level
                    offset_lng = (np.random.random() - 0.5) * offset_scale * level

                    # Create the perturbation with sharp angles
                    perturbed_lat = lat + offset_lat
                    perturbed_lng = lng + offset_lng

                    # Convert back to H3 index at the correct resolution
                    perturbed_cell = h3.latlng_to_cell(perturbed_lat, perturbed_lng, res)

                    # Replace the point in the trajectory only if it's different
                    original_cell_at_idx = new_traj[idx] # Store original before replacing
                    if perturbed_cell != original_cell_at_idx:
                        new_traj[idx] = perturbed_cell
                        modified = True
                    else:
                        # If perturbation didn't change cell, try a neighbor instead
                        # Explicitly ensure it's a set, though grid_disk should return one
                        neighbors_set = set(h3.grid_disk(original_cell_at_idx, 1))
                        neighbors_set.discard(original_cell_at_idx) # Now this should work on the confirmed set
                        if neighbors_set:
                            new_traj[idx] = np.random.choice(list(neighbors_set)) # Convert to list for choice
                            modified = True
                            perturbed_cell = new_traj[idx] # Update perturbed_cell for midpoint calc
                        else:
                            continue # Cannot modify this point

                    # Add additional points to create the sharp turn effect (zigzag)
                    # Make sure not to insert past the list bounds
                    if idx + 1 < len(new_traj):
                        # Get coordinate of the genuinely perturbed cell (might be neighbor)
                        perturbed_lat_mid, perturbed_lng_mid = h3.cell_to_latlng(perturbed_cell)

                        # Point away from the next point, using the perturbation direction
                        # Reduce the zigzag effect slightly compared to original sharp angle calc
                        mid_lat = perturbed_lat_mid + offset_lat * 0.3
                        mid_lng = perturbed_lng_mid + offset_lng * 0.3
                        mid_cell = h3.latlng_to_cell(mid_lat, mid_lng, res)

                        # Insert the midpoint only if different from surrounding points
                        if mid_cell != perturbed_cell and mid_cell != new_traj[idx+1]:
                             new_traj.insert(idx + 1, mid_cell)
                             indices_offset += 1 # Account for insertion
                             modified = True

                except Exception as e:
                     print(f"Error during sharp turn generation for index {idx_orig} in trajectory: {e}")
                     continue # Skip this turn point on error

            # Only add if modifications were actually made
            if modified:
                 outliers.append(new_traj)

        return outliers
    
    def partition_dataset(self, proportion=0.9, seed=123):
        np.random.seed(seed)
        train_num = int(len(self) * proportion)
        indices = np.random.permutation(len(self))
        train_indices, val_indices = indices[:train_num], indices[train_num:]
        return train_indices, val_indices
     
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index) -> Any:

        sample  = self.data[index]
        sample = ["SOT"] + list(sample) + ["EOT"]  # Convert tuple to list
        metadata = self.metadata[index]
        return sample, metadata
    
    def collate(self, data):
        """
        collate function
        """
        masks = []
        token_lists = []
        metadatas = []
        
        max_lenth = max([len(item[0]) for item in data])
        for tokens_, metadata in data: 
            
            mask = [1] * len(tokens_) + [0] * (max_lenth - len(tokens_))
            padded_sequence = tokens_ + [self.dictionary.pad()] * (max_lenth - len(tokens_))
            tokens = self.dictionary.encode(padded_sequence)
            token_lists.append(tokens)
            masks.append(mask)
            metadatas.append(metadata)

        token_lists = torch.tensor(token_lists)
        masks = torch.tensor(masks)
        return {
            "data" : token_lists,
            "mask": masks,
            "metadata": metadatas
        }

