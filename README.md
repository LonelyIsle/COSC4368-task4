# COSC 4368 – Task 4: Q-Learning & SARSA (Pickup-Delivery World)

## Contributors:
- Cole Plagens
- Javier Alvarez
- Satiago Segovia
- William Stewart
---

## Overview:
This project implements **Q-Learning** and **SARSA** agents operating in a Pickup-Delivery (PD) world.  
Two agents (Female `F` and Male `M`) learn to coordinate using shared Q-values and three exploration policies:  
- **PRANDOM** – pure random  
- **PGREEDY** – always takes the best Q-value action  
- **PEXPLOIT** – ε-greedy (80% exploit, 20% explore)

The goal is to compare convergence rates, reward trends, and adaptation when environment pickups change.

---

## Experiments Implemented:

| ID | Description | Learning | Policy | α | γ | Steps |
|----|--------------|-----------|--------|---|---|-------|
| **1(b)** | Q-Learning baseline | PRANDOM | 0.3 | 0.5 | 8000 |
| **1(c)** | Q-Learning greedy | PGREEDY | 0.3 | 0.5 | 8000 |
| **1(d)** | Q-Learning ε-greedy | PEXPLOIT | 0.3 | 0.5 | 8000 |
| **2** | SARSA (greedy) | PGREEDY | 0.3 | 0.5 | 8000 |
| **3(q)** | Q-Learning w/ lower α,γ | PGREEDY | 0.15 | 0.45 | 8000 |
| **3(sarsa)** | SARSA w/ lower α,γ | PGREEDY | 0.15 | 0.45 | 8000 |
| **4(q)** | Q-Learning w/ pickup switch | PEXPLOIT | 0.3 | 0.5 | ≤20000 |
| **4(sarsa)** | SARSA w/ pickup switch | PEXPLOIT | 0.3 | 0.5 | ≤20000 |

After the 3rd terminal in Experiment 4, the pickup locations change to **(1, 2)** and **(4, 5)** while reusing the learned Q-table.

Each experiment runs twice (different seeds) to compare stability.

---

## How to Run:
```bash
python3 main.py --exp 1b
python3 main.py --exp 1c
python3 main.py --exp 1d
python3 main.py --exp 2
python3 main.py --exp 3q
python3 main.py --exp 3sarsa
python3 main.py --exp 4q
python3 main.py --exp 4sarsa
```
