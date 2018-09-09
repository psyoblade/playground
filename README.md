# Playground for pommerman

## build docker images & run
```bash
$> docker build -f examples/docker-agent/Dockerfile -t psyoblade/psyoblade-pommerman-agent:1 .
..
Successfully built ad17f6a2858d
Successfully tagged psyoblade/psyoblade-pommerman-agent:1

$> docker run -d -p 10080:10080 psyoblade/psyoblade-pommerman-agent:1
..

$> docker ps
bash-3.2$ docker ps
CONTAINER ID        IMAGE                                   COMMAND                  CREATED             STATUS              PORTS                                                    NAMES
183af8d84592        psyoblade/psyoblade-pommerman-agent:1   "python run.py"          4 seconds ago       Up 2 seconds        0.0.0.0:10080->10080/tcp                                 dazzling_bell

$> curl -X GET http://localhost:10080/ping
{ 
  "success": true
}

$> docker logs -f 183af8d84592
172.17.0.1 - - [01/Sep/2018 15:46:14] "GET /ping HTTP/1.1" 200 -

$> echo '{"obs":{},"action_space":{[0,1,2,3,4,5]}}' > data.json
$> curl -d "@data.json" -X POST http://localhost:10080/action
$> curl -X POST -H "Content-Type: application/json" -d '{"obs":{}, "action_space":{[0,1,2,3,4,5]}}'
```
