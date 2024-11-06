scp ./client.py tsinghua_shiwx17@219.243.208.60:~/wangchao/client/client.py
scp ./client.py tsinghua_shiwx17@194.29.178.14:~/wangchao/client/client.py
scp ./client.py tsinghua_shiwx17@200.19.159.34:~/wangchao/client/client.py


ssh -l tsinghua_shiwx17 -i ~/.ssh/id_rsa 219.243.208.60 "rm -r ~/wangchao/client/trace"
ssh -l tsinghua_shiwx17 -i ~/.ssh/id_rsa 194.29.178.14 "rm -r ~/wangchao/client/trace"
ssh -l tsinghua_shiwx17 -i ~/.ssh/id_rsa 200.19.159.34 "rm -r ~/wangchao/client/trace"

