def run(self):
    # 初始化每个客户端的状态
    client_states = []
    for i, client in enumerate(self.clients):
        reqBI = client.init(
            videoName=video_names[i],
            bandwidthFiles=current_bandwidth_files,
            rtt=self.rtt,
            bwType=self.bw_type
        )
        # 初始化时设置 last_server_id 为 -1
        client.last_server_id = -1
        client_states.append({
            'done': False,
            'state': [
                BITRATES[reqBI] / BITRATES[-1],
                0,  # lastBitrate
                0,  # buffer
                throughput_mean,  # hThroughput
                throughput_mean,  # mThroughput
                0,  # server_idx
                -1/3  # last_server_id
            ],
            'segNum': 1,
            'r_sum': 0,
            'total_step': 0
        })

    # ... 在执行动作后更新 last_server_id ...
    # 在执行动作的循环中添加：
    server_idx = a // 6
    client.last_server_id = server_idx

    state['state'] = [
        reqBitrate / BITRATES[-1],
        lastBitrate / BITRATES[-1],
        (buffer/1000 - 30) / 10,
        (hThroughput - throughput_mean) / throughput_std,
        (mThroughput - throughput_mean) / throughput_std,
        server_idx / 2,
        (client.last_server_id + 1) / 3
    ] 