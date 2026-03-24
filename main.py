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
        # WYKRYWANIE CZY SĄ AKTYWNE OGRANICZENIA
        # ══════════════════════════════════════════════
        has_real_time_windows = any(
            tw[0] > 0 or tw[1] < 86400
            for i, tw in enumerate(data['time_windows'])
            if i not in data['starts'] and i not in data['ends']
        )
        has_ordering = bool(data['first_nodes'] or data['last_nodes'])

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
        # CALLBACK ODLEGŁOŚCI (minimalizacja kilometrów)
        # ══════════════════════════════════════════════
        def distance_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return data['distance_matrix'][from_node][to_node]

        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        # ══════════════════════════════════════════════
        # CALLBACK CZASU (weryfikacja okienek czasowych)
        # ══════════════════════════════════════════════
        def time_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return data['time_matrix'][from_node][to_node]

        time_callback_index = routing.RegisterTransitCallback(time_callback)

        routing.AddDimension(
            time_callback_index,
            1800,
            86400,
            False,
            'Time'
        )
        time_dimension = routing.GetDimensionOrDie('Time')

        # ══════════════════════════════════════════════
        # OKIENKA CZASOWE
        # ══════════════════════════════════════════════
        for location_idx, time_window in enumerate(data['time_windows']):
            if location_idx in data['starts'] or location_idx in data['ends']:
                continue
            index = manager.NodeToIndex(location_idx)
            time_dimension.CumulVar(index).SetRange(time_window[0], time_window[1])

        # ══════════════════════════════════════════════
        # STEP COUNTER (tylko gdy first/last są podane)
        # ══════════════════════════════════════════════
        if has_ordering:
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
        # PARAMETRY SZUKANIA — KLUCZOWA ZMIANA
        # ══════════════════════════════════════════════
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()

        # ── Strategia startowa zależna od ograniczeń ──
        if has_real_time_windows or has_ordering:
            # Dużo ograniczeń → strategia uwzględniająca constrainty
            search_parameters.first_solution_strategy = (
                routing_enums_pb2.FirstSolutionStrategy.PATH_MOST_CONSTRAINED_ARC
            )
        else:
            # Brak ograniczeń → buduj trasę "zachłannie" po najkrótszych łukach
            # To naturalnie tworzy klastry geograficzne (miasta)
            search_parameters.first_solution_strategy = (
                routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
            )

        # ── Metaheurystyka: GLS z agresywniejszym współczynnikiem ──
        search_parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        )
        # 0.15 to za mało — solver zbyt wolno ucieka z lokalnych minimów
        # 0.5 penalizuje mocniej powtórne użycie długich łuków (przeskoków)
        search_parameters.guided_local_search_lambda_coefficient = 0.5

        # ── Czas zależny od wielkości problemu ──
        n = len(data['distance_matrix'])
        if n <= 10:
            time_limit = 3
        elif n <= 20:
            time_limit = 8
        elif n <= 40:
            time_limit = 15
        elif n <= 80:
            time_limit = 30
        else:
            time_limit = 50
        search_parameters.time_limit.seconds = time_limit

        # ── Logi do debugowania (opcjonalnie) ──
        search_parameters.log_search = True

        # ══════════════════════════════════════════════
        # ROZWIĄZYWANIE
        # ══════════════════════════════════════════════
        solution = routing.SolveWithParameters(search_parameters)

        # ══════════════════════════════════════════════
        # WYNIK
        # ══════════════════════════════════════════════
        if solution:
            total_distance = 0
            all_routes = []

            for vehicle_id in range(data['num_vehicles']):
                index = routing.Start(vehicle_id)
                route = []

                while not routing.IsEnd(index):
                    time_var = time_dimension.CumulVar(index)
                    node = manager.IndexToNode(index)
                    route.append({
                        "node": node,
                        "arrival_min": solution.Min(time_var),
                        "arrival_max": solution.Max(time_var)
                    })
                    next_index = solution.Value(routing.NextVar(index))
                    total_distance += routing.GetArcCostForVehicle(
                        index, next_index, vehicle_id
                    )
                    index = next_index

                time_var = time_dimension.CumulVar(index)
                route.append({
                    "node": manager.IndexToNode(index),
                    "arrival_min": solution.Min(time_var),
                    "arrival_max": solution.Max(time_var)
                })
                all_routes.append(route)

            result = {
                "status": "success",
                "total_distance": total_distance,
                "computing_time_used": time_limit,
                "strategy_used": "constrained" if (
                    has_real_time_windows or has_ordering
                ) else "cheapest_arc",
            }

            if data['num_vehicles'] == 1:
                result["route"] = all_routes[0]
            else:
                result["routes"] = all_routes

            return result
        else:
            return {
                "status": "failed",
                "message": "Nie znaleziono trasy."
            }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "traceback": traceback.format_exc()
        }
