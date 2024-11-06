tc qdisc add dev ifb0 root handle 1: htb default 1
tc class add dev ifb0 parent 1:0 classid 1:1 htb rate 10Mbit

tc class add dev ifb0 parent 1:1 classid 1:2 htb rate 2Mbit
tc qdisc add dev ifb0 parent 1:2  handle 2: netem latency 100ms 20ms distribution normal loss 1% 25%
tc filter add dev ifb0 parent 1:0 protocol ip prio 100 u32 match ip sport 80 0xffff flowid 1:2

ifconfig ifb0 up

tc qdisc add dev enp129s0f0 ingress
# redirect all IP packets arriving in eth0 to ifb0 
# use mark 1 --> puts them onto class 1:1
tc filter add dev enp129s0f0 parent ffff: protocol ip prio 10 u32 \
match u32 0 0 flowid 1:1 \
action mirred egress redirect dev ifb0