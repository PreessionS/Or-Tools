from fastapi import FastAPI
from pydantic import BaseModel
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
from fastapi.middleware.cors import CORSMiddleware
import traceback

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class RouteRequest(BaseModel):
    num_vehicles: int = 1
    starts: list[int]
    ends: list[int]
    first_nodes: list[int] = []
    last_nodes: list[int] = []
    distance_matrix: list[list[int]]
    time_matrix: list[list[int]]
    time_windows: list[tuple[int, int]]

@app.post("/solve_tsptw")
def solve_tsptw(request: RouteRequest):
    try:
        data = {
            'distance_matrix': request.distance_matrix,
            'time_matrix': request.time_matrix,
            'time_windows': request.time_windows,
            'num_vehicles': request.num_vehicles,
            'starts': request.starts,
            'ends': request.ends,
            'first_nodes': request.first_nodes,
            'last_nodes': request.last_nodes
        }

        # ══════════════════════════════════════════════
        # MANAGER I MODEL
        # ══════════════════════════════════════════════
        manager = pywrapcp.RoutingIndexManager(
            len(data['distance_matrix']),
            data['num_vehicles'],
            data['starts'],
            data['ends']
        )
        routing = pywrapcp.RoutingModel(manager)

        # ══════════════════════════════════════════════
        # CALLBACK – COMBINED COST (czas + dystans)
        # ══════════════════════════════════════════════
        def combined_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            distance = data['distance_matrix'][from_node][to_node]
            time = data['time_matrix'][from_node][to_node]
            return int(distance + time * 2)  # waga czasu większa niż dystansu

        combined_callback_index = routing.RegisterTransitCallback(combined_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(combined_callback_index)

        # ══════════════════════════════════════════════
        # DIMENSION – CZAS (Time Windows)
        # ══════════════════════════════════════════════
        def time_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return data['time_matrix'][from_node][to_node]

        time_callback_index = routing.RegisterTransitCallback(time_callback)
        routing.AddDimension(
            time_callback_index,
            1800,   # Max oczekiwanie: 30 minut
            86400,  # Max łączny czas: 24h
            False,
            'Time'
        )
        time_dimension = routing.GetDimensionOrDie('Time')

        # ustawienie okienek czasowych
        for location_idx, time_window in enumerate(data['time_windows']):
            if location_idx in data['starts'] or location_idx in data['ends']:
                continue
            index = manager.NodeToIndex(location_idx)
            time_dimension.CumulVar(index).SetRange(time_window[0], time_window[1])

        # ══════════════════════════════════════════════
        # STEP COUNTER – opcjonalnie dla first/last nodes
        # ══════════════════════════════════════════════
        if data['first_nodes'] or data['last_nodes']:
            routing.AddConstantDimension(
                1,
                len(data['distance_matrix']) + 1,
                True,
                'StepCounter'
            )
            step_dimension = routing.GetDimensionOrDie('StepCounter')

            num_first = len(data['first_nodes'])
            num_last = len(data['last_nodes'])
            middle_nodes = [
                i for i in range(len(data['distance_matrix']))
                if i not in data['starts']
                and i not in data['ends']
                and i not in data['first_nodes']
                and i not in data['last_nodes']
            ]
            num_mid = len(middle_nodes)

            for node in range(len(data['distance_matrix'])):
                if node in data['starts'] or node in data['ends']:
                    continue
                index = manager.NodeToIndex(node)
                if node in data['first_nodes']:
                    step_dimension.CumulVar(index).SetRange(1, num_first)
                elif node in data['last_nodes']:
                    step_dimension.CumulVar(index).SetRange(
                        num_first + num_mid + 1,
                        num_first + num_mid + num_last
                    )
                else:
                    step_dimension.CumulVar(index).SetRange(
                        num_first + 1,
                        num_first + num_mid
                    )

        # ══════════════════════════════════════════════
        # GLOBAL SPAN – kara za długie trasy
        # ══════════════════════════════════════════════
        distance_dimension = routing.GetDimensionOrDie('StepCounter')  # używamy StepCounter
        distance_dimension.SetGlobalSpanCostCoefficient(100)

        # ══════════════════════════════════════════════
        # PARAMETRY SZUKANIA
        # ══════════════════════════════════════════════
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION
        )
        search_parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        )
        search_parameters.guided_local_search_lambda_coefficient = 0.15

        # dynamiczny limit czasu
        n = len(data['distance_matrix'])
        if n <= 10:
            time_limit = 2
        elif n <= 20:
            time_limit = 5
        elif n <= 40:
            time_limit = 12
        elif n <= 80:
            time_limit = 25
        else:
            time_limit = 50  # zwiększony limit
        search_parameters.time_limit.seconds = time_limit

        # ══════════════════════════════════════════════
        # ROZWIĄZYWANIE
        # ══════════════════════════════════════════════
        solution = routing.SolveWithParameters(search_parameters)

        # ══════════════════════════════════════════════
        # WYNIK
        # ══════════════════════════════════════════════
        if solution:
            index = routing.Start(0)
            route = []

            while not routing.IsEnd(index):
                time_var = time_dimension.CumulVar(index)
                route.append({
                    "node": manager.IndexToNode(index),
                    "arrival_min": solution.Min(time_var),
                    "arrival_max": solution.Max(time_var)
                })
                index = solution.Value(routing.NextVar(index))

            time_var = time_dimension.CumulVar(index)
            route.append({
                "node": manager.IndexToNode(index),
                "arrival_min": solution.Min(time_var),
                "arrival_max": solution.Max(time_var)
            })

            return {
                "status": "success",
                "route": route,
                "computing_time_used": time_limit
            }
        else:
            return {
                "status": "failed",
                "message": "Nie znaleziono trasy. Możliwe przyczyny: "
                           "1) Okienka czasowe wykluczają się nawzajem, "
                           "2) First/Last nodes tworzą niemożliwą kolejność, "
                           "3) Czas przejazdu przekracza okienka."
            }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "traceback": traceback.format_exc()
        }
