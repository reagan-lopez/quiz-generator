#!/bin/bash
docker stop psdbcon
docker rm psdbcon
docker rmi psdb
docker build -t psdb .
docker run -d -p 5432:5432 --name psdbcon psdb
docker logs psdbcon
#docker exec -it psdbcon psql -U admin -d prescreendb -c "SELECT * FROM users;"