##  RESCUER AGENT
### @Author: Tacla (UTFPR)
### Demo of use of VictimSim
### This rescuer version implements:
### - clustering of victims by quadrants of the explored region
### - definition of a sequence of rescue of victims of a cluster
### - assigning one cluster to one rescuer
### - calculating paths between pair of victims using breadth-first search
###
### One of the rescuers is the master in charge of unifying the maps and the information
### about the found victims.

import os
import random
import math
import csv
import sys
from map import Map
from vs.abstract_agent import AbstAgent
from vs.physical_agent import PhysAgent
from vs.constants import VS
from bfs import BFS
from abc import ABC, abstractmethod
import numpy as np
import heapq
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from deap import base, creator, tools, algorithms

N_CLUSTERS = 8


class No:
    def __init__(self):
        self.parent_x = 0  # coordenada pai x
        self.parent_y = 0
        self.fn = float("inf")  # custo total g+h
        self.gn = float("inf")  # Custo apartir do nó inicial
        self.hn = 0  # Custo do nó até a vitima


## Classe que define o Agente Rescuer com um plano fixo
class Rescuer(AbstAgent):
    def __init__(self, env, config_file, nb_of_explorers=1, clusters=[], id=1):
        """
        @param env: a reference to an instance of the environment class
        @param config_file: the absolute path to the agent's config file
        @param nb_of_explorers: number of explorer agents to wait for
        @param clusters: list of clusters of victims in the charge of this agent"""

        super().__init__(env, config_file)

        # Specific initialization for the rescuer
        self.nb_of_explorers = (
            nb_of_explorers  # number of explorer agents to wait for start
        )
        self.received_maps = 0  # counts the number of explorers' maps
        self.map = Map()  # explorer will pass the map
        self.victims = {}  # a dictionary of found victims: [vic_id]: ((x,y), [<vs>])
        self.plan = []  # a list of planned actions in increments of x and y
        self.plan_x = 0  # the x position of the rescuer during the planning phase
        self.plan_y = 0  # the y position of the rescuer during the planning phase
        self.plan_visited = set()  # positions already planned to be visited
        self.plan_rtime = self.TLIM  # the remaing time during the planning phase
        self.plan_walk_time = 0.0  # previewed time to walk during rescue
        self.x = 0  # the current x position of the rescuer when executing the plan
        self.y = 0  # the current y position of the rescuer when executing the plan
        self.clusters = clusters  # the clusters of victims this agent should take care of - see the method cluster_victims
        self.sequences = clusters  # the sequence of visit of victims for each cluster
        self.id = id

        # Starts in IDLE state.
        # It changes to ACTIVE when the map arrives
        self.set_state(VS.IDLE)

    # Check if a no is unblocked
    def is_unblocked(grid, row, col):
        return grid[row][col] == 1

    # Check if a no is the destination
    def is_destination(self, row, col, dest):
        return row == dest[0] and col == dest[1]

    # Calcula o valor de h
    def calculate_h_value(self, row, col, dest):
        return ((row - dest[0]) ** 2 + (col - dest[1]) ** 2) ** 0.5

    def track(self, no_details, dest):
        row, col = dest
        self.plan_x = row
        self.plan_y = col

        plano = []

        # Caminha do destino até a origem
        while not (
            no_details[row][col].parent_i == row
            and no_details[row][col].parent_j == col
        ):
            temp_row = no_details[row][col].parent_i
            temp_col = no_details[row][col].parent_j

            dx = row - temp_row  # Correção para funcionar com negativos e positivos
            dy = col - temp_col  # Correção para funcionar com negativos e positivos

            row, col = temp_row, temp_col
            plano.append((dx, dy, False))  # Adiciona passos normais

        # Ajustamos o último movimento para ser o primeiro real
        if plano:
            last_dx, last_dy, _ = plano[0]
            plano[0] = (last_dx, last_dy, True)  # Marca o último passo como True

        plano.reverse()  # Reverte para ficar na ordem correta
        self.plan.extend(plano)

    def a_star_search(self, src, dest, start):
        # Check if the source and destination are valid

        # Check if we are already at the destination
        if self.is_destination(src[0], src[1], dest):
            print("We are already at the destination")
            return

        # TODO: Não tenho tamannho do mapa para inicializar as listas
        min_x, max_x, min_y, max_y = self.map.get_min_max_map()
        max_y = max_y - min_y + 1
        max_x = max_x - min_x + 1

        # Inicializa lista fechada de nós
        closed_list = [[False for _ in range(max_y)] for _ in range(max_x)]
        # Inicializa os nós do mapa inteiro
        no_details = [[No() for _ in range(max_y)] for _ in range(max_x)]

        # Initialize the start no details
        i = src[0]
        j = src[1]
        no_details[i][j].fn = 0
        no_details[i][j].gn = 0
        no_details[i][j].hn = 0
        no_details[i][j].parent_i = i
        no_details[i][j].parent_j = j

        # Inicializa lista aberta (Nos para ser visitado) com o começo em src
        open_list = []
        heapq.heappush(open_list, (0.0, i, j))

        # Initialize the flag for whether destination is found
        found_dest = False

        # Main loop of A* search algorithm
        while len(open_list) > 0:
            # Remove o elemento com o menor f
            p = heapq.heappop(open_list)

            # Marca na lista fechada que a coordenada (i,j) foi visitada
            i = p[1]
            j = p[2]
            closed_list[i][j] = True

            # Pega direção disponivel
            actions_res = self.map.get((i, j))[2]

            for k, ar in enumerate(actions_res):
                # Se caminho não for livre, apenas pulamos.
                # No mapa não foi implementado VS.UNKS
                if ar != VS.CLEAR:
                    # print(f"{self.NAME} {k} not clear")
                    continue

                dir = Rescuer.AC_INCR[k]

                new_i = i + dir[0]
                new_j = j + dir[1]

                # Verifica se o sucessor é valido.
                difficulty = self.map.get((new_i, new_j))
                if difficulty == None or ar != VS.CLEAR:
                    # print("posição mapa NUla")
                    continue

                if self.is_destination(new_i, new_j, dest):
                    # Set the parent of the destination no
                    if dir[0] == 0 or dir[1] == 0:
                        g_new = no_details[i][j].gn + self.COST_LINE * difficulty[0]
                    else:
                        g_new = no_details[i][j].gn + self.COST_DIAG * difficulty[0]

                    no_details[new_i][new_j].parent_i = i
                    no_details[new_i][new_j].parent_j = j
                    # print("The destination no is found")
                    self.plan_rtime -= g_new + self.COST_FIRST_AID
                    self.plan_walk_time += g_new + self.COST_FIRST_AID
                    if self.plan_walk_time > self.plan_rtime * 2 and dest != start:
                        self.plan_rtime += g_new + self.COST_FIRST_AID
                        self.plan_walk_time -= g_new + self.COST_FIRST_AID
                        return
                    self.track(no_details, (new_i, new_j))
                    found_dest = True

                    # print("AQUI\n CUSTOS: ", self.plan_rtime, self.plan_visited)
                    return
                else:

                    # Calculate the new f, g, and h values
                    if dir[0] == 0 or dir[1] == 0:
                        g_new = no_details[i][j].gn + self.COST_LINE * difficulty[0]
                    else:
                        g_new = no_details[i][j].gn + self.COST_DIAG * difficulty[0]

                    h_new = self.calculate_h_value(new_i, new_j, dest)
                    f_new = g_new + h_new

                    # If the no is not in the open list or the new f value is smaller
                    if (
                        no_details[new_i][new_j].fn == float("inf")
                        or no_details[new_i][new_j].fn > f_new
                    ):
                        # Add the no to the open list
                        heapq.heappush(open_list, (f_new, new_i, new_j))
                        # Update the no details
                        no_details[new_i][new_j].fn = f_new
                        no_details[new_i][new_j].gn = g_new
                        no_details[new_i][new_j].hn = h_new
                        no_details[new_i][new_j].parent_i = i
                        no_details[new_i][new_j].parent_j = j

        # If the destination is not found after visiting all nos

        # Starts in IDLE state.
        # It changes to ACTIVE when the map arrives
        self.set_state(VS.IDLE)

    def save_cluster_csv(self, cluster, cluster_id):
        filename = f"./clusters/cluster{cluster_id}.txt"
        with open(filename, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            for vic_id, values in cluster.items():
                x, y = values[0]  # x,y coordinates
                vs = values[1]  # list of vital signals
                writer.writerow([vic_id, x, y, vs[6], vs[7]])

    def save_sequence_csv(self, sequence, sequence_id):
        filename = f"./clusters/seq{sequence_id}.txt"
        with open(filename, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            for id, values in sequence.items():
                x, y = values[0]  # x,y coordinates
                vs = values[1]  # list of vital signals
                writer.writerow([id, x, y, vs[6], vs[7]])

    def cluster_victims(self):
        """this method does a naive clustering of victims per quadrant: victims in the
        upper left quadrant compose a cluster, victims in the upper right quadrant, another one, and so on.

        @returns: a list of clusters where each cluster is a dictionary in the format [vic_id]: ((x,y), [<vs>])
                  such as vic_id is the victim id, (x,y) is the victim's position, and [<vs>] the list of vital signals
                  including the severity value and the corresponding label"""
        vic = []
        for i in self.victims:
            vic.append(self.victims[i])

        data = []
        while vic:
            coord, vs = vic.pop()
            data.append(
                [
                    coord[0],
                    coord[1],
                    vs[0],
                    vs[1],
                    vs[2],
                    vs[3],
                    vs[4],
                    vs[5],
                    vs[6],
                    vs[7],
                ]
            )

        df = pd.DataFrame(
            data,
            columns=[
                "x",
                "y",
                "Id",
                "pSist",
                "pDiast",
                "qPA",
                "pulso",
                "freq_resp",
                "grav",
                "classe",
            ],
        )

        # Create clusters directory if it doesn't exist
        os.makedirs("clusters", exist_ok=True)

        # Prepare data for clustering
        # Scale features for clustering
        feature_scaler = MinMaxScaler()

        # Scale spatial features (x, y)
        spatial_features = feature_scaler.fit_transform(df[["x", "y"]])

        # Scale gravity (already predicted)
        gravity_scaled = feature_scaler.fit_transform(df[["grav"]])

        # Convert class to numeric importance (inverse of class number since class 1 is most critical)
        class_importance = 5 - df["classe"].values.reshape(
            -1, 1
        )  # Class 1 (critical) becomes 4, Class 4 (stable) becomes 1
        class_scaled = feature_scaler.fit_transform(class_importance)

        # Combine features with weights
        # Higher weights for gravity and class to prioritize severity
        # Format: [x, y, gravity * weight, class * weight]
        SPATIAL_FEATURE_WEIGHT = 1.4
        GRAVITY_WEIGHT = 1
        CLASS_WEIGHT = 1

        clustering_features = np.hstack(
            [
                spatial_features
                * SPATIAL_FEATURE_WEIGHT,  # x, y coordinates (weight 1.0)
                gravity_scaled * GRAVITY_WEIGHT,  # gravity with higher weight
                class_scaled * CLASS_WEIGHT,  # class importance with higher weight
            ]
        )

        # Perform K-means clustering
        kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
        df["cluster"] = kmeans.fit_predict(clustering_features)

        # Calculate cluster priorities based on average gravity and class
        cluster_priorities = []
        for i in range(N_CLUSTERS):
            cluster_data = df[df["cluster"] == i]
            avg_gravity = cluster_data["grav"].mean()
            avg_class = cluster_data["classe"].mean()
            # Priority score: higher gravity and lower class number (more severe) means higher priority
            priority = avg_gravity * (5 - avg_class)
            cluster_priorities.append((i, priority))

        # Sort clusters by priority
        cluster_priorities.sort(key=lambda x: x[1], reverse=True)

        # Reassign cluster numbers based on priority (highest priority = cluster 1)
        cluster_map = {
            old_cluster: new_cluster + 1
            for new_cluster, (old_cluster, _) in enumerate(cluster_priorities)
        }
        df["cluster"] = df["cluster"].map(cluster_map)

        # Print cluster statistics
        print("\nCluster Statistics:")
        for i in range(1, N_CLUSTERS + 1):
            cluster_data = df[df["cluster"] == i]
            print(f"\nCluster {i}:")
            print(f"Number of victims: {len(cluster_data)}")
            print(f"Average gravity: {cluster_data['grav'].mean():.2f}")
            print(f"Average class: {cluster_data['classe'].mean():.2f}")
            print("Class distribution:")
            print(cluster_data["classe"].value_counts().sort_index())

        clusters = [{} for _ in range(N_CLUSTERS)]

        for _, row in df.iterrows():
            vic_id = row["Id"]

            cluster = self.victims[vic_id]
            # print(f"Cluster {row['cluster']}")
            # print(f"Cluster {int(row['cluster'])}")
            clusters[int(row["cluster"]) - 1][vic_id] = cluster

        self.plot_clusters(df)

        # for i, cluster in enumerate(clusters):
        #     print(f"Cluster {i+1} length: {len(cluster)}")

        return clusters

    def predict_severity_and_class(self):
        """@TODO to be replaced by a classifier and a regressor to calculate the class of severity and the severity values.
        This method should add the vital signals(vs) of the self.victims dictionary with these two values.

        This implementation assigns random values to both, severity value and class"""

        classifier = joblib.load("models/best_classifier.joblib")
        regressor = joblib.load("models/best_regressor.joblib")
        scaler = joblib.load("models/scaler.joblib")

        for vic_id, values in self.victims.items():
            # features = scaler.transform(values[1][["qPA", "pulso", "freq_resp"]])
            feature_df = pd.DataFrame(
                [[values[1][3], values[1][4], values[1][5]]],
                columns=["qPA", "pulso", "freq_resp"],
            )
            features = scaler.transform(feature_df)
            severity_value = regressor.predict(features)
            severity_class = classifier.predict(features)
            values[1].extend([severity_value[0], severity_class[0]])

    def sequencing(self):
        """Currently, this method sort the victims by the x coordinate followed by the y coordinate
        @TODO It must be replaced by a Genetic Algorithm that finds the possibly best visiting order
        """

        """ We consider an agent may have different sequences of rescue. The idea is the rescuer can execute
            sequence[0], sequence[1], ...
            A sequence is a dictionary with the following structure: [vic_id]: ((x,y), [<vs>]"""

        new_sequences = []

        # Access the dictionary inside the list
        # order by id
        sequence_dict = {}
        for i, seq in enumerate(self.sequences):
            seq = dict(sorted(seq.items(), key=lambda item: item[0]))
            sequence_dict.update(seq)

        # Create a mapping from victim IDs to indices
        victim_id_to_index = {
            int(vic_id): idx for idx, vic_id in enumerate(sequence_dict.keys())
        }

        # Convert the keys of sequence_dict to a list of indices
        victim_indices = list(victim_id_to_index.values())

        # Create positions and danger levels arrays
        positions = np.array([values[0] for _, values in sequence_dict.items()])

        danger_levels = np.array([values[1][7] for _, values in sequence_dict.items()])

        # print("--------------------------------")
        # print("Victim IDs:")
        # print(victim_indices)
        # print("--------------------------------")
        # print("Positions:")
        # print(positions)
        # print("--------------------------------")
        # print("Danger Levels:")
        # print(danger_levels)
        # print("--------------------------------")

        # Função de avaliação (fitness)
        # Considera que resgata todas as vitimas de uma vez
        # se for ter que alterar por perigo de vitima, altera-se a distancia de cada nó.
        # a distancia será a diferença euclidiana da origem até a vitima.(no final a distancia seria a mesma)
        def evaluate(individuo):
            total_distance = 0
            urgency_score = 0
            for i in range(len(individuo) - 1):
                v1, v2 = individuo[i], individuo[i + 1]
                total_distance += np.linalg.norm(
                    positions[v1] - positions[v2]
                )  # Distância euclidiana
                urgency_score += danger_levels[v1] / (
                    i + 1
                )  # Penalizar atrasos no atendimento

            return (total_distance, urgency_score)

        # Configuração do DEAP
        creator.create("FitnessMin", base.Fitness, weights=(-0.5, 0.5))  # Minimização
        creator.create("EstrIndividuos", list, fitness=creator.FitnessMin)

        # Armazena as premissas do problema
        toolbox = base.Toolbox()

        # Register a function to generate a random permutation of victim IDs
        toolbox.register("Genes", np.random.permutation, victim_indices)

        # Register a function to create an individual using the permutation of victim IDs
        toolbox.register(
            "individuos", tools.initIterate, creator.EstrIndividuos, toolbox.Genes
        )
        # Register a function to create a population of individuals
        toolbox.register("populacao", tools.initRepeat, list, toolbox.individuos)

        # Operadores genéticos
        toolbox.register("mate", tools.cxPartialyMatched)  # Cruzamento para permutações
        toolbox.register(
            "mutate", tools.mutShuffleIndexes, indpb=0.1
        )  # Mutação por troca -> 10%
        toolbox.register(
            "select", tools.selTournament, tournsize=2
        )  # Torneio para seleção
        toolbox.register("evaluate", evaluate)

        # Algoritmo Genético com DEAP
        def genetic_algorithm(
            n_generations=100, population_size=50, cxpb=0.7, mutpb=0.2
        ):
            population = toolbox.populacao(n=population_size)
            hall_of_fame = tools.HallOfFame(1)  # Armazena o melhor indivíduo

            stats = tools.Statistics(lambda ind: ind.fitness.values)
            stats.register("min", np.min)
            stats.register("avg", np.mean)

            population, logbook = algorithms.eaSimple(
                population,
                toolbox,
                cxpb=cxpb,
                mutpb=mutpb,
                ngen=n_generations,
                stats=stats,
                halloffame=hall_of_fame,
                verbose=True,
            )

            return hall_of_fame[0], hall_of_fame[0].fitness.values[0]

        # Executar o algoritmo genético
        best_route, best_score = genetic_algorithm(
            # n_generations=500, population_size=200
        )

        # Mostrar resultados
        print("\nMelhor ordem de visitação:", best_route)
        print("Custo total (fitness):", best_score)

        positions = np.array(positions)
        best_route_positions = positions[best_route]

        plt.figure(figsize=(10, 6))
        plt.scatter(positions[:, 0], positions[:, 1], c="red", label="Vítimas")
        plt.plot(
            np.append(best_route_positions[:, 0], best_route_positions[0, 0]),
            np.append(best_route_positions[:, 1], best_route_positions[0, 1]),
            c="blue",
            label="Melhor Rota",
        )
        for i, (x, y) in enumerate(positions):
            plt.text(x, y, f"{i}", fontsize=12, ha="center", va="center")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.legend()
        plt.title(f"Melhor Rota para Resgate {self.id}")
        plt.savefig(f"melhor_rota_rescuer_{self.id}.png")

        # Assuming victim_id_to_index is already defined
        index_to_victim_id = {
            index: vic_id for vic_id, index in victim_id_to_index.items()
        }

        # Convert best_route indices back to original victim IDs
        best_route_victim_ids = [index_to_victim_id[index] for index in best_route]

        ordered_seq = {
            vic_id: sequence_dict[vic_id]
            for vic_id in best_route_victim_ids
            if vic_id in sequence_dict
        }

        # Append the ordered sequence to new_sequences
        new_sequences.append(ordered_seq)

        # Update self.sequences with the new order
        self.sequences = new_sequences

    def planner(self):
        """A method that calculates the path between victims: walk actions in a OFF-LINE MANNER (the agent plans, stores the plan, and
        after it executes. Eeach element of the plan is a pair dx, dy that defines the increments for the the x-axis and  y-axis.
        """

        # let's instantiate the breadth-first search

        self.plan_visited.add(
            (0, 0)
        )  # always start from the base, so it is already visited
        difficulty, vic_seq, actions_res = self.map.get((0, 0))
        # self.__depth_search(actions_res)
        vic = []
        sequence = self.sequences[0]
        start = (self.plan_x, self.plan_y)

        for vic_id in sequence:
            goal = sequence[vic_id][0]
            self.a_star_search((self.plan_x, self.plan_y), goal, start)

        self.a_star_search((self.plan_x, self.plan_y), start, start)

    def sync_explorers(self, explorer_map, victims):
        """This method should be invoked only to the master agent

        Each explorer sends the map containing the obstacles and
        victims' location. The master rescuer updates its map with the
        received one. It does the same for the victims' vital signals.
        After, it should classify each severity of each victim (critical, ..., stable);
        Following, using some clustering method, it should group the victims and
        and pass one (or more)clusters to each rescuer"""

        self.received_maps += 1

        print(f"{self.NAME} Map received from the explorer")
        self.map.update(explorer_map)
        self.victims.update(victims)

        if self.received_maps == self.nb_of_explorers:
            print(f"{self.NAME} all maps received from the explorers")
            # self.map.draw()
            # print(f"{self.NAME} found victims by all explorers:\n{self.victims}")

            # @TODO predict the severity and the class of victims' using a classifier
            self.predict_severity_and_class()

            # @TODO cluster the victims possibly using the severity and other criteria
            # Here, there 4 clusters
            clusters_of_vic = self.cluster_victims()

            for i, cluster in enumerate(clusters_of_vic):
                self.save_cluster_csv(cluster, i + 1)  # file names start at 1

            # Instantiate the other rescuers
            rescuers = [None] * 4
            rescuers[0] = self  # the master rescuer is the index 0 agent

            # Assign the cluster the master agent is in charge of
            # TODO: @THIAGO OLHAR ESSA PARTE
            self.clusters = [clusters_of_vic[0], clusters_of_vic[1]]  # the first one

            # Instantiate the other rescuers and assign the clusters to them
            for i in range(1, 4):
                # print(f"{self.NAME} instantianting rescuer {i+1}, {self.get_env()}")
                filename = f"rescuer_{i+1:1d}_config.txt"
                config_file = os.path.join(self.config_folder, filename)
                # each rescuer receives one cluster of victims
                rescuers[i] = Rescuer(
                    self.get_env(),
                    config_file,
                    4,
                    [clusters_of_vic[i * 2], clusters_of_vic[(i * 2) + 1]],
                    i + 1,
                )
                rescuers[i].map = self.map  # each rescuer have the map

            # Calculate the sequence of rescue for each agent
            # In this case, each agent has just one cluster and one sequence
            self.sequences = self.clusters

            # For each rescuer, we calculate the rescue sequence
            for i, rescuer in enumerate(rescuers):
                rescuer.sequencing()  # the sequencing will reorder the cluster

                for j, sequence in enumerate(rescuer.sequences):
                    if j == 0:
                        self.save_sequence_csv(
                            sequence, i + 1
                        )  # primeira sequencia do 1o. cluster 1: seq1
                    else:
                        self.save_sequence_csv(
                            sequence, (i + 1) + j * 10
                        )  # demais sequencias do 1o. cluster: seq11, seq12, seq13, ...

                print(f"id {rescuer.id} planner")
                rescuer.planner()  # make the plan for the trajectory
                rescuer.set_state(
                    VS.ACTIVE
                )  # from now, the simulator calls the deliberation method

    def deliberate(self):
        """This is the choice of the next action. The simulator calls this
        method at each reasonning cycle if the agent is ACTIVE.
        Must be implemented in every agent
        @return True: there's one or more actions to do
        @return False: there's no more action to do"""

        # No more actions to do
        if self.plan == []:  # empty list, no more actions to do
            print(f"{self.NAME} has finished the plan [ENTER]")
            return False

        # Takes the first action of the plan (walk action) and removes it from the plan
        # print(f"{self.NAME} plan {self.plan}")
        dx, dy, there_is_vict = self.plan.pop(0)
        # print(f"{self.NAME} pop dx: {dx} dy: {dy} there_is_vict: {there_is_vict}")

        # Walk - just one step per deliberation
        walked = self.walk(dx, dy)

        # Rescue the victim at the current position
        if walked == VS.EXECUTED:
            self.x += dx
            self.y += dy
            # print(f"{self.NAME} Walk ok - Rescuer at position ({self.x}, {self.y})")

            # check if there is a victim at the current position
            if self.map.in_map((self.x, self.y)):
                vic_id = self.map.get_vic_id((self.x, self.y))
                if vic_id != VS.NO_VICTIM:
                    self.first_aid()
                    # if self.first_aid(): # True when rescued
                    # print(f"{self.NAME} Victim rescued at ({self.x}, {self.y})")
        else:
            print(
                f"{self.NAME} Plan fail - walk error - agent at ({self.x}, {self.y}) - {walked}"
            )

        return True

    def plot_clusters(self, df, save_path="clusters/cluster_visualization.png"):
        """Plot the clusters with victims colored by cluster and markers by severity class."""
        plt.figure(figsize=(12, 8))

        # Create color palette for clusters
        cluster_colors = sns.color_palette("husl", n_colors=df["cluster"].nunique())

        # Create marker styles for different classes
        markers = {
            1: "X",  # Critical
            2: "s",  # Unstable
            3: "^",  # Potentially stable
            4: "o",  # Stable
        }
        marker_sizes = {
            1: 150,
            2: 100,
            3: 80,
            4: 60,
        }

        # Plot each class separately for proper legend
        for classe in sorted(df["classe"].unique()):
            for cluster in sorted(df["cluster"].unique()):
                mask = (df["classe"] == classe) & (df["cluster"] == cluster)
                plt.scatter(
                    df[mask]["x"],
                    df[mask]["y"],
                    c=[cluster_colors[cluster - 1]],
                    marker=markers[classe],
                    s=marker_sizes[classe],
                    alpha=0.7,
                    label=f"Cluster {cluster} - Class {classe}",
                )

        # Create a custom legend
        legend_labels = {
            1: "Critical",
            2: "Unstable",
            3: "Potentially Stable",
            4: "Stable",
        }

        # Sort legend entries by cluster first, then by class
        handles, labels = plt.gca().get_legend_handles_labels()
        # Create a list of tuples (cluster_number, class_number, handle, label)
        legend_entries = []
        for h, l in zip(handles, labels):
            cluster = int(l.split()[1])
            classe = int(l.split()[-1])
            legend_entries.append((cluster, classe, h, l))

        # Sort by cluster first, then by class
        legend_entries.sort(key=lambda x: (x[0], x[1]))

        # Create new labels with descriptive text
        new_labels = [
            f"Cluster {entry[0]} - {legend_labels[entry[1]]}"
            for entry in legend_entries
        ]

        # Extract sorted handles
        sorted_handles = [entry[2] for entry in legend_entries]

        # Add the legend with both colors and markers
        plt.legend(
            sorted_handles,
            new_labels,
            loc="center left",
            bbox_to_anchor=(1, 0.5),
            title="Clusters and Severity Classes",
            borderaxespad=0.5,
        )

        plt.title("Victim Clusters and Severity Classes")
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")

        # Add grid
        plt.grid(True, linestyle="--", alpha=0.7)

        # Adjust layout to prevent legend cutoff
        plt.tight_layout()

        # Save the plot
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
