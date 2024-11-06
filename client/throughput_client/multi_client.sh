#!/bin/bash
# network_type traceIndex client_id TURN_COUNT
./clear_tc.sh 
./initial_tc.sh
for i in `seq 40`
do
	echo $i
	sleep 1
	sudo python3 throughput_client.py -1 6 $i 100&
	
done

sleep 1800
ps -ef | grep python3|awk '{print $2}'| xargs kill -9