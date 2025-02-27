# -*- coding: utf-8 -*
import matplotlib
matplotlib.use('Agg')
import torch
import torch.nn as nn
from utils import v_wrap, set_init, push_and_pull, record
import torch.nn.functional as F
import torch.multiprocessing as mp
from shared_adam import SharedAdam
from client_pensieve import Client
import random
import os
import numpy as np
import time
import sys # For receiving parameters
import platform
from torch.optim.lr_scheduler import StepLR  # Add this import line


M_IN_K = 1000
BITRATES = [350, 600, 1000, 2000, 3000]
if platform.system() == "Linux":
    PRINTFLAG = False
else:
    PRINTFLAG = True
S_LEN = 7
A_DIM = 18
PUREFLAF = False
throughput_mean = 2297514.2311790097
throughput_std = 4369117.906444455
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SERVERS = ["edge1", "edge2", "edge3"]


def takeSecond(elem):
    return elem[1]


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        # Actor network
        self.linear_1_a = nn.Linear(S_LEN, 200)
        self.ln_1_a = nn.LayerNorm(200)
        self.linear_2_a = nn.Linear(200, 100)
        self.ln_2_a = nn.LayerNorm(100)
        self.output_a = nn.Linear(100, A_DIM)
        
        # Critic network
        self.linear_1_c = nn.Linear(S_LEN, 200)
        self.ln_1_c = nn.LayerNorm(200)
        self.linear_2_c = nn.Linear(200, 100)
        self.ln_2_c = nn.LayerNorm(100)
        self.output_c = nn.Linear(100, 1)
        
        # Initialize weights
        set_init([self.linear_1_a, self.linear_2_a, self.output_a,
                 self.linear_1_c, self.linear_2_c, self.output_c])
        
        self.distribution = torch.distributions.Categorical
        
        # Move model to GPU
        self.to(device)

    def forward(self, x):
        # Ensure input is on GPU
        x = x.to(device)
        
        # Actor forward propagation
        a = F.relu(self.ln_1_a(self.linear_1_a(x)))
        a = F.relu(self.ln_2_a(self.linear_2_a(a)))
        logits = self.output_a(a)
        
        # Critic forward propagation
        c = F.relu(self.ln_1_c(self.linear_1_c(x)))
        c = F.relu(self.ln_2_c(self.linear_2_c(c)))
        values = self.output_c(c)
        
        return logits, values


    # def choose_action(self, mask, state):
    #     self.eval()
        
    #     # Ensure model and all inputs are on the same device
    #     self = self.to(device)
        
    #     # Move input to the correct device
    #     if isinstance(state, np.ndarray):
    #         state = torch.FloatTensor(state)
    #     state = state.to(device)
        
    #     if isinstance(mask, list):
    #         mask = torch.tensor(mask)
    #     mask = mask.to(device).bool()
        
    #     # Add batch dimension (if needed)
    #     if len(state.shape) == 1:
    #         state = state.unsqueeze(0)
    #     if len(mask.shape) == 1:
    #         mask = mask.unsqueeze(0)
        
    #     with torch.no_grad():
    #         # Correctly unpack the return values of forward
    #         logits, value = self.forward(state)
            
    #         # Now operate on logits
    #         masked_logits = logits.clone()
    #         masked_logits[~mask] = float('-inf')
    #         masked_logits = masked_logits - masked_logits.max()
    #         probs = F.softmax(masked_logits, dim=-1)
            
    #         # Debug information
    #         # print("\nAction probabilities after masking:")
    #         # for i, prob in enumerate(probs[0]):
    #         #     if prob > 0:
    #         #         server_idx = i // 6
    #         #         action_idx = i % 6
    #         #         server_name = ["edge1", "edge2", "edge3"][server_idx]
    #         #         print(f"Action {i} ({server_name}, {'miss' if action_idx == 5 else f'bitrate_{action_idx}'}): {prob:.4f}")
            
    #         if torch.isnan(probs).any():
    #             print("Warning: NaN values in probabilities:", probs)
            
    #         dist = self.distribution(probs[0])
    #         action = dist.sample()
    #         action_log_prob = dist.log_prob(action)
            
    #     return action.cpu().item(), action_log_prob.cpu().item(), value.cpu().item()
    def choose_action(self, mask, state):
        """
        Choose action during testing phase, no exploration needed
        Args:
            mask: Action mask
            state: Current state
        Returns:
            action_idx: Selected action index
            action_log_prob: Action log probability
            value: State value
        """
        self.eval()  # Set to evaluation mode
        
        # Ensure model and all inputs are on the same device
        self = self.to(device)
        
        # Move input to the correct device and process format
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state)
        state = state.to(device)
        
        if isinstance(mask, list):
            mask = torch.tensor(mask)
        mask = mask.to(device).bool()
        
        # Add batch dimension (if needed)
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
        if len(mask.shape) == 1:
            mask = mask.unsqueeze(0)
        
        with torch.no_grad():  # No gradients needed during testing
            # Get policy network output
            logits, value = self.forward(state)
            
            # Apply action mask
            masked_logits = logits.clone()
            masked_logits[~mask] = float('-inf')
            
            # Directly use softmax to get probability distribution
            probs = F.softmax(masked_logits, dim=-1)
            
            # Directly select the action with the highest probability during testing
            action = torch.argmax(probs, dim=-1)
            
            # Calculate action log probability (for recording)
            dist = self.distribution(probs)
            action_log_prob = dist.log_prob(action)
        
        return action.cpu().item(), action_log_prob.cpu().item(), value.cpu().item()


def get_rttDict(rttDict):
    lines = open("../file/rtt/rtt.txt").readlines()
    for line in lines:
        ip = line.split(" ")[0]
        rtt = float(line.split(" ")[1].strip())
        if ip not in rttDict:
            rttDict[ip] = rtt


class Worker(mp.Process):
    def __init__(self, model_dir,  model_name, policy, trace_index, cache_files, bw_type):
        super(Worker, self).__init__()
        
        # Save basic parameters
        self.model_dir = model_dir
        self.model_name = model_name
        self.policy = policy
        self.trace_index = trace_index
        self.cache_files = cache_files
        self.bw_type = bw_type
        
        # Initialize neural network
        self.lnet = Net()
        if policy == "RL":
            self.lnet.load_state_dict(torch.load(model_dir + model_name, map_location=device, weights_only=True))
        
        # Initialize logitsDict
        self.logitsDict = {}
        
        # Initialize videoTraceIndex
        self.videoTraceIndex = 0
        
        # Get RTT information
        self.rttDict = {}
        get_rttDict(self.rttDict)
        
        # Initialize busy trace
        self.busyTraceL = os.listdir("../../data/trace/busy/2")
        
        # Set trace directory
        trace_dir = f"{model_dir}/test_trace_{trace_index}/{bw_type}/{policy}"
        
        # Initialize client, set type to "test" and pass trace_dir
        client_type = "mixed"  # Default value
        if self.bw_type == "FCC":
            client_type = "FCC"
        elif self.bw_type == "HSDPA":
            client_type = "HSDPA"
        elif self.bw_type == "mine":
            client_type = "mine"
        
        # Modify here: Set type to "test" and pass trace_dir
        self.client = Client(
            type="test",  # Change to "test" to enable trace recording
            traceDir=trace_dir  # Add trace directory
        )
        
        # Load video list
        video_file = open("../file/videoList/video.txt")
        self.videoList = video_file.readlines()
        self.videoList = [(i.split(" ")[0], float(i.split(" ")[1])) for i in self.videoList]
        video_file.close()
        
        # Load video trace
        videoTraceFile = open("../file/videoTrace.txt")
        self.videoTraceIndex = 0
        self.videoTraceList = videoTraceFile.readlines()
        self.videoTraceList = [i.strip() for i in self.videoTraceList]
        videoTraceFile.close()
        
        # Assign different types of bandwidth files for each server
        self.bandwidth_files = {
            "edge1": [],  # FCC/HSDPA/mine
            "edge2": [],  # 5G Netflix
            "edge3": []   # 5G Netflix
        }
        
        # Load edge1's bandwidth files (based on bw_type)
        if self.bw_type == "FCC":
            edge1_dir = "../data/bandwidth/test/FCC"
            if os.path.exists(edge1_dir):
                self.bandwidth_files["edge1"] = sorted(
                    [os.path.join(edge1_dir, f) for f in os.listdir(edge1_dir) if f.endswith('.log')]
                )
        elif self.bw_type == "HSDPA":
            edge1_dir = "../data/bandwidth/test/HSDPA"
            if os.path.exists(edge1_dir):
                self.bandwidth_files["edge1"] = sorted(
                    [os.path.join(edge1_dir, f) for f in os.listdir(edge1_dir) if f.endswith('.log')]
                )
        else:  # mine
            edge1_dir = "../data/bandwidth/test/mine"
            if os.path.exists(edge1_dir):
                for ip_dir in os.listdir(edge1_dir):
                    ip_path = os.path.join(edge1_dir, ip_dir)
                    if os.path.isdir(ip_path):
                        self.bandwidth_files["edge1"].extend(
                            sorted([os.path.join(ip_path, f) for f in os.listdir(ip_path) if f.endswith('.log')])
                        )
        
        # Load 5G Netflix bandwidth files for edge2 and edge3
        netflix_dir = "../data/bandwidth/train/5G_Neflix_static"
        if os.path.exists(netflix_dir):
            netflix_files = sorted([os.path.join(netflix_dir, f) for f in os.listdir(netflix_dir) if f.endswith('.log')])
            # Average allocate Netflix files to edge2 and edge3
            mid = len(netflix_files) // 2
            self.bandwidth_files["edge2"] = netflix_files[:mid]
            self.bandwidth_files["edge3"] = netflix_files[mid:]
        
        # Modify the structure of cached_lists to store video names and their corresponding bitrate versions
        self.cached_lists = {
            "edge1": {},  # {video_name: set(version_numbers)}
            "edge2": {},
            "edge3": {}
        }
        
        # Load data from cache files
        # Iterate over each server and its corresponding cache files
        for server_name, cache_file in cache_files.items():
            # Check if cache file exists
            if os.path.exists(cache_file):
                # Open and read cache file
                with open(cache_file, 'r') as f:
                    # Iterate over each line in the file
                    for line in f:
                        # Split each line into parts by space
                        parts = line.strip().split()
                        # Ensure each line has at least 3 parts (video name, version number, size)
                        if len(parts) >= 3:
                            video_name = parts[0].strip()  # First part is video name
                            version = int(parts[1].strip())  # Second part is bitrate version number
                            # Check if server is in cache list
                            if server_name in self.cached_lists:
                                # If this video is not in the server's cache, create a new set
                                if video_name not in self.cached_lists[server_name]:
                                    self.cached_lists[server_name][video_name] = set()
                                # Add this version number to the video's version set
                                self.cached_lists[server_name][video_name].add(version)


    def get_action_mask(self, videoName):
        """Get action mask, considering specific bitrate versions"""
        pure_video_name = videoName.strip()
        
        # Initialize all actions as unavailable
        mask = [0] * A_DIM
        # print(f"\nGenerating mask for video: {pure_video_name}")
        
        for server_idx, server_name in enumerate(SERVERS):
            cache_dict = self.cached_lists[server_name]
            base_idx = server_idx * 6
            # print(f"\nChecking {server_name}:")
            
            # Set miss action as available
            mask[base_idx + 5] = 1
            
            if pure_video_name in cache_dict:
                versions = cache_dict[pure_video_name]
                #print(f"{server_name} has versions: {sorted(list(versions))}")
                # Set versions in cache as available
                for version in versions:
                    action_idx = base_idx + (version - 1)
                    mask[action_idx] = 1
                    # print(f"Setting version {version} (action_idx={action_idx}) as available")
            # else:
            #     print(f"{server_name} does not have this video")
            
            #print(f"Current mask after {server_name}: {mask}")
        
        #print(f"Final mask: {mask}")
        #print(f"Available actions: {[i for i, m in enumerate(mask) if m == 1]}")
        return mask

    def choose_action_lower(self, mask, reqBI):
        """Choose the highest bitrate version that is less than or equal to the requested bitrate"""
        best_action = -1
        best_bitrate = -1
        
        # Iterate over all servers and bitrate
        for server_idx in range(3):  # 3 servers
            base_idx = server_idx * 6
            # Check each bitrate version (except miss action)
            for bitrate_idx in range(reqBI + 1):  # Only consider bitrates less than or equal to reqBI
                action_idx = base_idx + bitrate_idx
                if mask[action_idx] == 1:  # If this action is available
                    if bitrate_idx > best_bitrate:
                        best_action = action_idx
                        best_bitrate = bitrate_idx
        
        # If no suitable bitrate is found, return any available miss action
        if best_action == -1:
            for server_idx in range(3):
                miss_idx = server_idx * 6 + 5
                if mask[miss_idx] == 1:
                    return miss_idx
        
        return best_action

    def choose_action_closest(self, mask, reqBI):
        """Choose the version closest to the requested bitrate"""
        best_action = -1
        min_dist = float('inf')
        
        # Iterate over all servers and bitrate
        for server_idx in range(3):
            base_idx = server_idx * 6
            # Check each bitrate version (except miss action)
            for bitrate_idx in range(5):  # 5 bitrate versions
                action_idx = base_idx + bitrate_idx
                if mask[action_idx] == 1:
                    dist = abs(bitrate_idx - reqBI)
                    if dist < min_dist:
                        min_dist = dist
                        best_action = action_idx
        
        # If no suitable bitrate is found, return any available miss action
        if best_action == -1:
            for server_idx in range(3):
                miss_idx = server_idx * 6 + 5
                if mask[miss_idx] == 1:
                    return miss_idx
        
        return best_action

    def choose_action_highest(self, mask, reqBI):
        """Choose the highest available bitrate version"""
        best_action = -1
        best_bitrate = -1
        
        # Iterate over all servers and bitrate
        for server_idx in range(3):
            base_idx = server_idx * 6
            # Check each bitrate version (except miss action)
            for bitrate_idx in range(5):  # 5 bitrate versions
                action_idx = base_idx + bitrate_idx
                if mask[action_idx] == 1:
                    if bitrate_idx > best_bitrate:
                        best_action = action_idx
                        best_bitrate = bitrate_idx
        
        # If no suitable bitrate is found, return any available miss action
        if best_action == -1:
            for server_idx in range(3):
                miss_idx = server_idx * 6 + 5
                if mask[miss_idx] == 1:
                    return miss_idx
        
        return best_action

    def choose_action_prefetch(self, mask, reqBI):
        """Prefer to choose the requested bitrate, if unavailable randomly choose miss or prefetch (with cost)"""
        # First try to find the requested bitrate on each server
        for server_idx in range(3):
            action_idx = server_idx * 6 + reqBI
            if mask[action_idx] == 1:
                return action_idx, False  # Return action and whether it's prefetch
        
        # If the requested bitrate cannot be found, randomly decide whether to use miss
        if random.random() < 0.5:
            # Return any available miss action
            for server_idx in range(3):
                miss_idx = server_idx * 6 + 5
                if mask[miss_idx] == 1:
                    return miss_idx, False
        else:
            # 50% probability to prefetch to the requested bitrate, but increase download time
            return reqBI, True  # Return action and prefetch mark


    def getBandwidthFileList(self):
        rtt = -1
        if PUREFLAF:
            dir = "../../data/bandwidth/train/" + self.bwType
        else:

            dir = "../../data/bandwidth/test/" + self.bwType

        if self.bwType == "mine":
            ipDirList = os.listdir(dir)
            for ip in ipDirList:
                if ip not in self.rttDict:
                    print("no this ip:", ip, "in the bandwidth directory")
                    dir = "../../data/bandwidth/test/HSDPA"
                else:
                    rtt = self.rttDict[ip]
                bandwidthFileList = os.listdir(dir + "/" + ip)
                for fileName in bandwidthFileList:
                    self.bandwidth_fileList.append([dir + "/" + ip + "/" + fileName, rtt, self.bwType])
        else:
            bandwidthFileList = os.listdir(dir)
            for fileName in bandwidthFileList:
                self.bandwidth_fileList.append([dir + "/" + fileName, rtt, self.bwType])


    def get_busyTrace(self):
        """Get busy trace"""
        busyList = []
        busyTrace = random.choice(self.busyTraceL)
        busyTrace = "../../data/trace/busy/2/" + busyTrace
        lines = open(busyTrace).readlines()
        for line in lines:
            busyList.append(float(line.strip()))
        return busyList


    def getSizeDict():
        file = open("./file/videoSizePerVision.txt")
        lines = file.readlines()
        virsionSizeDict = {}  # videoName_bIndex size
        for line in lines:
            virsion = line.split(" ")[0]
            size = int(line.split(" ")[1])
            if virsion not in virsionSizeDict:
                virsionSizeDict[virsion] = size
            else:
                print(virsion, "show twice")
        return virsionSizeDict


    def saveLogits(self, videoName, logits):
        # Method to save logits
        key = videoName
        if key not in self.logitsDict:
            self.logitsDict[key] = []
        self.logitsDict[key].append(logits)
        
        # Optional: Save to file periodically
        if len(self.logitsDict[key]) % 100 == 0:  # Save every 100 logits
            save_dir = os.path.join(self.model_dir, "logits")
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"{key}_logits.npy")
            np.save(save_path, np.array(self.logitsDict[key]))


    def run(self):
        # Initialize reward statistics for each server
        server_rewards = {
            "edge1": {"sum": 0, "count": 0},
            "edge2": {"sum": 0, "count": 0},
            "edge3": {"sum": 0, "count": 0}
        }
        
        r_avg_sum = 0
        resFile = open(f"{self.model_dir}/test_result_{self.bw_type}_{self.policy}.txt", "w")
        
        for file_index in range(len(self.bandwidth_files['edge1'])):
            # Get busy trace
            busyList = self.get_busyTrace()
            
            # Build bandwidth files for each server
            bandwidth_files = {
                "edge1": self.bandwidth_files["edge1"][file_index],
                "edge2": self.bandwidth_files["edge2"][file_index % len(self.bandwidth_files["edge2"])],
                "edge3": self.bandwidth_files["edge3"][file_index % len(self.bandwidth_files["edge3"])]
            }
            
            # Build RTT dictionary
            rtt_values = {}
            
            # edge1's RTT from file
            if self.bw_type == "mine":
                ip = bandwidth_files["edge1"].split("/")[-2]  # Extract IP from path
                if ip not in self.rttDict:
                    print("no this ip:", ip, "in the bandwidth directory")
                    edge1_dir = "../data/bandwidth/train/HSDPA"
                    bandwidth_files["edge1"] = random.choice(os.listdir(edge1_dir))
                else:
                    rtt_values["edge1"] = self.rttDict[ip]
            else:
                rtt_values["edge1"] = self.rttDict.get(self.bw_type, -1)
            
            # edge2 and edge3 use fixed 5G network RTT values
            rtt_values["edge2"] = 20  # Typical 5G network RTT value
            rtt_values["edge3"] = 20  # Typical 5G network RTT value
            
            # Get video name
            if PUREFLAF:
                video_random = random.random()
                videoName = ""
                for i in range(len(self.videoList)):
                    if video_random < self.videoList[i][1]:
                        videoName = self.videoList[i - 1][0]
                        break
                if videoName == "":
                    videoName = self.videoList[-1][0]
            else:
                if self.videoTraceIndex == len(self.videoTraceList):
                    self.videoTraceIndex = 0
                videoName = self.videoTraceList[self.videoTraceIndex].strip()  # Ensure no whitespace
            
            # Initialize client
            reqBI = self.client.init(
                videoName=videoName,
                bandwidthFiles=bandwidth_files,
                rtt=rtt_values,
                bwType=self.bw_type
            )
            
            # Initialize state variables
            reqBitrate = BITRATES[reqBI]
            lastBitrate = 0
            buffer = 0
            hThroughput = throughput_mean
            mThroughput = throughput_mean
            server_idx = 0
            last_server_id = -1
            
            # Initialize state vector
            state_ = [
                reqBitrate / BITRATES[-1],
                lastBitrate / BITRATES[-1],
                (buffer/1000 - 30) / 10,
                (hThroughput - throughput_mean) / throughput_std,
                (mThroughput - throughput_mean) / throughput_std,
                server_idx / 2,
                (last_server_id + 1) / 3
            ]
            
            state = state_.copy()
            
            # Start testing------------------------------
            total_step = 0
            segNum = 0
            r_sum = 0
            
            while True:
                # Get action mask
                mask = self.get_action_mask(videoName)
                
                # Initialize is_prefetch as False
                is_prefetch = False
                
                if sum(mask) == 1:
                    a = mask.index(1)
                else:
                    if self.policy == "no_policy":
                        # Return first available miss action, but keep original requested bitrate
                        for server_idx in range(3):
                            miss_idx = server_idx * 6 + 5
                            if mask[miss_idx] == 1:
                                a = miss_idx
                                is_prefetch = False
                                break
                    elif self.policy == "RL":
                        # Convert state data
                        state_array = np.array(state)
                        state_tensor = v_wrap(state_array[None, :])
                        # Use weights_only=True to load model
                        a, logits, _ = self.lnet.choose_action(mask, state_tensor)
                        
                        # Debug information (only output when debug=True)
                        # if self.debug:
                        #     server_idx = a // 6
                        #     action_idx = a % 6
                        #     server_name = ["edge1", "edge2", "edge3"][server_idx]
                        #     bitrate = BITRATES[action_idx] if action_idx < 5 else "miss"
                        #     print(f"\nAction Details:")
                        #     print(f"Selected Server: {server_name}")
                        #     print(f"Selected Bitrate: {bitrate}Kbps")
                        #     print(f"State shape: {state_array.shape}")
                        #     print(f"State values: {state_array}")
                        
                        self.saveLogits(videoName, logits)
                    elif self.policy == "lower":
                        a = self.choose_action_lower(mask, reqBI)
                    elif self.policy == "closest":
                        a = self.choose_action_closest(mask, reqBI)
                    elif self.policy == "highest":
                        a = self.choose_action_highest(mask, reqBI)
                    elif self.policy == "prefetch":
                        a, is_prefetch = self.choose_action_prefetch(mask, reqBI)
                    else:
                        print("想啥呢")
                        return

                # if random.randint(0, 1000) == 1:
                #     print("reqb=", reqBitrate, "lb=", lastBitrate, "buffer=", int(buffer), "hT=", int(hThroughput),
                #           "mT=", int(mThroughput), "busy=", round(busy, 2),
                #           "mask=", mask, "action=", a, "reqBI=", reqBI, "reward=", round(reward, 2), "logits=", logits)

                # Use busyList
                busy = busyList[segNum % len(busyList)]
                if a % 6 == 5:
                    hitFlag = False
                    action_idx = reqBI  # Use original requested bitrate
                    # print(f"Miss action - using original reqBI: {reqBI}")
                else:
                    hitFlag = True
                    action_idx = a % 6
                    # print(f"Hit action - using original reqBI: {reqBI}")

                # Get server index and specific action
                server_idx = a // 6
                server_names = ["edge1", "edge2", "edge3"]
                server_name = server_names[server_idx]
                
                # Update current server
                self.current_server = server_idx
                
                # Pass prefetch mark
                reqBitrate, lastBitrate, buffer, hThroughput, mThroughput, reward, reqBI, done, segNum = self.client.run(
                    action_idx, busy, hitFlag, server_name, is_prefetch)
                reward = reward / 5 


                # Get previous server ID
                last_server_id = -1 if self.client.last_server is None else \
                                int(self.client.last_server.split('edge')[1]) - 1

                # Update state vector
                state_[0] = reqBitrate / BITRATES[-1]
                state_[1] = lastBitrate / BITRATES[-1]
                state_[2] = (buffer/1000 - 30) / 10
                state_[3] = (hThroughput - throughput_mean) / throughput_std
                state_[4] = (mThroughput - throughput_mean) / throughput_std
                state_[5] = server_idx / 2
                state_[6] = (last_server_id + 1) / 3

                r_sum += reward
                total_step += 1
                if done:
                    break
                state = state_.copy()
            # End testing------------------------------
            r_avg = r_sum / total_step if total_step > 0 else 0
            r_avg_sum += r_avg
            
            # Record server reward in loop
            current_server = SERVERS[a // 6]  # Use a instead of action
            server_rewards[current_server]["sum"] += reward
            server_rewards[current_server]["count"] += 1
            
            # Write result to file
            resFile.write(f"{r_avg}\n")
            resFile.flush()
            
            # Output result
            print(f"Summary: {self.bw_type} | {self.policy} | Video:{videoName} | "
                  f"Progress:{file_index}/{len(self.bandwidth_files['edge1'])} | "
                  f"Avg Reward:{r_avg:.2f}")
            
            self.videoTraceIndex += 1
        
        # Calculate and print statistics for each server
        for server, stats in server_rewards.items():
            if stats["count"] > 0:
                avg = stats["sum"] / stats["count"]
                print(f"{server} Average reward: {avg:.2f} (Used times: {stats['count']})")
        
        # Calculate overall average reward
        total_reward = sum(s["sum"] for s in server_rewards.values())
        total_count = sum(s["count"] for s in server_rewards.values())
        
        resFile.close()
        return total_reward / total_count if total_count > 0 else 0




def main():
    bwTypes = ["FCC", "HSDPA",  "mine"]
    modelDir = "../data/RL_model/2024-12-19_14-30-50"
    modelName = "/model/best_model.pth"
    policys = ["RL", "no_policy", "lower", "closest", "highest", "prefetch"]
    #policys = ["lower", "closest", "highest", "prefetch"]
    traceIndex = "Pensieve_3"
    
    # Define cache files for each server
    cache_files = {
        "edge1": "../file/cachedFile/cachedFile_20190612_pure.txt",
        "edge2": "../file/cachedFile/cachedFile_20241210.txt",
        "edge3": "../file/cachedFile/cachedFile_20241211.txt"
    }

    for bwType in bwTypes:
        for policy in policys:  
            worker = Worker(modelDir, modelName, policy, traceIndex, cache_files, bwType)
            worker.debug = True  # Debug output
            worker.run()


if __name__ == "__main__":
    main()