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
    start_seconds: int = 0

    starts: list[int]
    ends: list[int]

    first_nodes: list[int] = []
    last_nodes: list[int] = []

    distance_matrix: list[list[int]]
    time_matrix: list[list[int]]

    # [(start, end)]
    time_windows: list[tuple[int, int]]

    # Kara za wcześniejszy przyjazd
    early_penalty: int = 1000

    # Kara za spóźnienie
    late_penalty: int = 2000


SOLVER_STATUS_MAP = {
    0: "Solver nie został uruchomiony",
    1: "Rozwiązanie znalezione",
    2: "Brak feasible solution — dane mogą być sprzeczne",
    3: "Przekroczono limit czasu — brak rozwiązania w wyznaczonym czasie",
    4: "Nieprawidłowe dane wejściowe",
}

TRIVIAL_WINDOW = (0, 86400)


@app.post("/solve_tsptw")
def solve_tsptw(request: RouteRequest):

    try:

        data = {
            'distance_matrix': request.distance_matrix,
            'time_matrix': request.time_matrix,
            'time_windows': request.time_windows,
            'num_vehicles': request.num_vehicles,
            'start_seconds': request.start_seconds,
            'starts': request.starts,
            'ends': request.ends,
            'first_nodes': request.first_nodes,
            'last_nodes': request.last_nodes,
            'early_penalty': request.early_penalty,
            'late_penalty': request.late_penalty
        }

        starts_set = set(data['starts'])
        ends_set = set(data['ends'])

        # ══════════════════════════════════════════════
        # WYKRYWANIE OGRANICZEŃ
        # ══════════════════════════════════════════════

        has_real_time_windows = any(
            tuple(tw) != TRIVIAL_WINDOW
            and (tw[0] > 0 or tw[1] < 86400)
            for i, tw in enumerate(data['time_windows'])
            if i not in starts_set and i not in ends_set
        )

        has_ordering = bool(
            data['first_nodes'] or data['last_nodes']
        )

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
        # CALLBACK ODLEGŁOŚCI
        # ══════════════════════════════════════════════

        def distance_callback(from_index, to_index):

            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)

            return data['distance_matrix'][from_node][to_node]

        transit_callback_index = routing.RegisterTransitCallback(
            distance_callback
        )

        routing.SetArcCostEvaluatorOfAllVehicles(
            transit_callback_index
        )

        # ══════════════════════════════════════════════
        # CALLBACK CZASU
        # ══════════════════════════════════════════════

        def time_callback(from_index, to_index):

            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)

            return data['time_matrix'][from_node][to_node]

        time_callback_index = routing.RegisterTransitCallback(
            time_callback
        )

        # ══════════════════════════════════════════════
        # WYMIAR CZASU
        # ══════════════════════════════════════════════

        routing.AddDimension(
            time_callback_index,
            0,          # brak czekania — okna miękkie obsługują tolerancję
            172800,
            False,
            'Time'
        )

        time_dimension = routing.GetDimensionOrDie('Time')

        # ══════════════════════════════════════════════
        # START POJAZDÓW
        # ══════════════════════════════════════════════

        for vehicle_id in range(data['num_vehicles']):

            start_index = routing.Start(vehicle_id)

            time_dimension.CumulVar(start_index).SetRange(
                data['start_seconds'],
                data['start_seconds']
            )

        # ══════════════════════════════════════════════
        # OKNA CZASOWE
        # ±30 MIN ODCHYLENIA
        # ══════════════════════════════════════════════

        MAX_DEVIATION = 1800  # 30 minut

        if has_real_time_windows:

            for location_idx, time_window in enumerate(
                data['time_windows']
            ):

                if (
                    location_idx in starts_set
                    or location_idx in ends_set
                ):
                    continue

                # Pomiń trywialne okna
                if tuple(time_window) == TRIVIAL_WINDOW:
                    continue

                index = manager.NodeToIndex(location_idx)

                # Twardy zakres ±30 minut
                hard_start = max(
                    0,
                    time_window[0] - MAX_DEVIATION
                )

                hard_end = (
                    time_window[1] + MAX_DEVIATION
                )

                time_dimension.CumulVar(index).SetRange(
                    hard_start,
                    hard_end
                )

                # Miękka kara za wcześniejszy przyjazd
                time_dimension.SetCumulVarSoftLowerBound(
                    index,
                    time_window[0],
                    data['early_penalty']
                )

                # Miękka kara za spóźnienie
                time_dimension.SetCumulVarSoftUpperBound(
                    index,
                    time_window[1],
                    data['late_penalty']
                )

        # ══════════════════════════════════════════════
        # STEP COUNTER
        # ══════════════════════════════════════════════

        if has_ordering:

            routing.AddConstantDimension(
                1,
                len(data['distance_matrix']) + 1,
                True,
                'StepCounter'
            )

            step_dimension = routing.GetDimensionOrDie(
                'StepCounter'
            )

            num_first = len(data['first_nodes'])
            num_last = len(data['last_nodes'])

            middle_nodes = [

                i for i in range(
                    len(data['distance_matrix'])
                )

                if (
                    i not in starts_set
                    and i not in ends_set
                    and i not in data['first_nodes']
                    and i not in data['last_nodes']
                )
            ]

            num_mid = len(middle_nodes)

            for node in range(
                len(data['distance_matrix'])
            ):

                if (
                    node in starts_set
                    or node in ends_set
                ):
                    continue

                index = manager.NodeToIndex(node)

                if node in data['first_nodes']:

                    step_dimension.CumulVar(index).SetRange(
                        1,
                        num_first
                    )

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
        # PARAMETRY SOLVERA
        # ══════════════════════════════════════════════

        search_parameters = (
            pywrapcp.DefaultRoutingSearchParameters()
        )

        if has_real_time_windows or has_ordering:

            search_parameters.first_solution_strategy = (
                routing_enums_pb2.FirstSolutionStrategy.PATH_MOST_CONSTRAINED_ARC
            )

        else:

            search_parameters.first_solution_strategy = (
                routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
            )

        search_parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        )

        search_parameters.guided_local_search_lambda_coefficient = 0.5

        # ══════════════════════════════════════════════
        # LIMIT CZASU
        # ══════════════════════════════════════════════

        n = len(data['distance_matrix'])

        if n <= 10:
            time_limit = 3

        elif n <= 20:
            time_limit = 10

        elif n <= 42:
            time_limit = 25

        elif n <= 80:
            time_limit = 40

        elif n <= 150:
            time_limit = 60

        elif n <= 300:
            time_limit = 140

        else:
            time_limit = 180

        search_parameters.time_limit.seconds = time_limit

        search_parameters.log_search = True

        # ══════════════════════════════════════════════
        # ROZWIĄZYWANIE
        # ══════════════════════════════════════════════

        solution = routing.SolveWithParameters(
            search_parameters
        )

        # ══════════════════════════════════════════════
        # WYNIK
        # ══════════════════════════════════════════════

        if solution:

            total_distance = 0
            all_routes = []

            for vehicle_id in range(
                data['num_vehicles']
            ):

                index = routing.Start(vehicle_id)

                route = []

                while not routing.IsEnd(index):

                    time_var = (
                        time_dimension.CumulVar(index)
                    )

                    node = manager.IndexToNode(index)

                    route.append({
                        "node": node,
                        "arrival_min": solution.Min(time_var),
                        "arrival_max": solution.Max(time_var)
                    })

                    next_index = solution.Value(
                        routing.NextVar(index)
                    )

                    total_distance += (
                        routing.GetArcCostForVehicle(
                            index,
                            next_index,
                            vehicle_id
                        )
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

                "strategy_used": (
                    "constrained"
                    if (
                        has_real_time_windows
                        or has_ordering
                    )
                    else "cheapest_arc"
                )
            }

            if data['num_vehicles'] == 1:

                result["route"] = all_routes[0]

            else:

                result["routes"] = all_routes

            return result

        else:

            solver_status = routing.status()

            return {
                "status": "failed",
                "solver_status": solver_status,
                "message": SOLVER_STATUS_MAP.get(
                    solver_status,
                    "Nieznany błąd solvera"
                )
            }

    except Exception as e:

        return {
            "status": "error",
            "message": str(e),
            "traceback": traceback.format_exc()
        }
