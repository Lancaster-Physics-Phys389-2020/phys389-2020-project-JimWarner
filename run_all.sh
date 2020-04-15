#!/bin/sh

python grav_sim.py g 20 1e6 100 -t 10 -i -s 1
python grav_sim.py g 50 1e6 100 -t 10 -i -s 1
python grav_sim.py g 100 1e6 100 -t 10 -i -s 10
python grav_sim.py g 200 1e6 100 -t 10 -m -e 10 -f -i -s 1
python grav_sim.py g 500 1e6 100 -t 10 -i -s 1
python grav_sim.py g 1000 1e6 100 -t 10 -i -s 1
python grav_sim.py g 2000 1e6 100 -t 10 -i -s 1
python grav_sim.py g 200 1e4 100 -m -e 10 -i -s 1
python grav_sim.py g 200 1e5 100 -m -e 10 -i -s 1
python grav_sim.py g 200 1e7 100 -m -e 10 -i -s 1

python grav_sim.py u 100 1e9 100 -t 10 -i -s 1
python grav_sim.py u 200 1e9 100 -t 10 -i -s 1
python grav_sim.py u 500 1e9 100 -t 10 -i -s 1
python grav_sim.py u 1000 1e9 100 -t 10 -i -s 1
python grav_sim.py u 2000 1e9 100 -t 10 -m -f -i -s 1
python grav_sim.py u 5000 1e9 100 -t 10 -i -s 1
python grav_sim.py u 10000 1e9 100 -t 10 -i -s 1
python grav_sim.py u 2000 1e7 100 -m -i -s 1
python grav_sim.py u 2000 1e8 100 -m -i -s 1
python grav_sim.py u 2000 1e10 100 -m -i -s 1
python grav_sim.py u 2000 1e11 100 -m -i -s 1








