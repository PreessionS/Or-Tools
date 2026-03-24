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
    # Nowe opcje sterujące
    optimize_by: str = "time"  # "time", "distance", "balanced"


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

        n = len(data['distance_matrix'])

        # ══════════════════════════════════════════════
        # WALIDACJA DANYCH
        # ══════════════════════════════════════════════
        for i in range(n):
            for j in range(n):
                if i != j and data['distance_matrix'][i][j] == 0:
                    return {
                        "status": "error",
                        "message": f"distance_matrix[{i}][{j}] = 0, "
                                   f"ale i != j. Sprawdź dane."
                    }

        # ══════════════════════════════════════════════
        # MANAGER I MODEL
        # ══════════════════════════════════════════════
        manager = pywrapcp.RoutingIndexManager(
            n,
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

        distance_callback_index = routing.RegisterTransitCallback(
            distance_callback
        )

        # ══════════════════════════════════════════════
        # CALLBACK CZASU
        # ══════════════════════════════════════════════
        def time_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return data['time_matrix'][from_node][to_node]

        time_callback_index = routing.RegisterTransitCallback(time_callback)

        # ══════════════════════════════════════════════
        # ⭐ WYBÓR CO MINIMALIZUJEMY
        # ══════════════════════════════════════════════
        if request.optimize_by == "time":
            # Minimalizuj CZAS przejazdu
            routing.SetArcCostEvaluatorOfAllVehicles(time_callback_index)

        elif request.optimize_by == "distance":
            # Minimalizuj KILOMETRY
            routing.SetArcCostEvaluatorOfAllVehicles(distance_callback_index)

        elif request.optimize_by == "balanced":
            # Minimalizuj kombinację: czas + odległość
            def combined_callback(from_index, to_index):
                from_node = manager.IndexToNode(from_index)
                to_node = manager.IndexToNode(to_index)
                time_cost = data['time_matrix'][from_node][to_node]
                dist_cost = data['distance_matrix'][from_node][to_node]
                # Waga: 70% czas, 30% dystans (dostosuj do potrzeb)
                return int(time_cost * 0.7 + dist_cost * 0.3)

            combined_callback_index = routing.RegisterTransitCallback(
                combined_callback
            )
            routing.SetArcCostEvaluatorOfAllVehicles(combined_callback_index)

        else:
            # Domyślnie: czas
            routing.SetArcCostEvaluatorOfAllVehicles(time_callback_index)

        # ══════════════════════════════════════════════
        # WYMIAR CZASU (okienka czasowe)
        # ══════════════════════════════════════════════
        routing.AddDimension(
            time_callback_index,
            1800,    # Max oczekiwanie: 30 minut
            86400,   # Max łączny czas: 24h
            False,
            'Time'
        )
        time_dimension = routing.GetDimensionOrDie('Time')

        # ⭐ MINIMALIZUJ CAŁKOWITY CZAS TRASY (span)
        # To każe solverowi skracać czas od startu do końca
        for vehicle_id in range(data['num_vehicles']):
            end_index = routing.End(vehicle_id)
            start_index = routing.Start(vehicle_id)

            # Minimalizuj czas dotarcia do końca trasy
            time_dimension.SetSpanCostCoefficientForVehicle(100, vehicle_id)

            # Minimalizuj oczekiwanie (slack) na każdym węźle
            time_dimension.SetGlobalSpanCostCoefficient(100)

        # ══════════════════════════════════════════════
        # OKIENKA CZASOWE
        # ══════════════════════════════════════════════
        for location_idx, time_window in enumerate(data['time_windows']):
            if location_idx in data['starts'] or location_idx in data['ends']:
                continue
            index = manager.NodeToIndex(location_idx)
            time_dimension.CumulVar(index).SetRange(
                time_window[0], time_window[1]
            )

        # Okienka dla startów i końców
        for vehicle_id in range(data['num_vehicles']):
            start_node = data['starts'][vehicle_id]
            start_tw = data['time_windows'][start_node]
            start_index = routing.Start(vehicle_id)
            time_dimension.CumulVar(start_index).SetRange(
                start_tw[0], start_tw[1]
            )

            end_node = data['ends'][vehicle_id]
            end_tw = data['time_windows'][end_node]
            end_index = routing.End(vehicle_id)
            time_dimension.CumulVar(end_index).SetRange(
                end_tw[0], end_tw[1]
            )

        # ══════════════════════════════════════════════
        # WYMIAR ODLEGŁOŚCI (do raportowania)
        # ══════════════════════════════════════════════
        routing.AddDimension(
            distance_callback_index,
            0,          # zero slack
            999999999,  # max dystans
            True,       # start od zera
            'Distance'
        )
        distance_dimension = routing.GetDimensionOrDie('Distance')

        # ══════════════════════════════════════════════
        # STEP COUNTER (first/last nodes)
        # ══════════════════════════════════════════════
        if data['first_nodes'] or data['last_nodes']:
            routing.AddConstantDimension(
                1,
                n + 1,
                True,
                'StepCounter'
            )
            step_dimension = routing.GetDimensionOrDie('StepCounter')

            num_first = len(data['first_nodes'])
            num_last = len(data['last_nodes'])
            middle_nodes = [
                i for i in range(n)
                if i not in data['starts']
                and i not in data['ends']
                and i not in data['first_nodes']
                and i not in data['last_nodes']
            ]
            num_mid = len(middle_nodes)

            for node in range(n):
                if node in data['starts'] or node in data['ends']:
                    continue
                index = manager.NodeToIndex(node)
                if node in data['first_nodes']:
                    step_dimension.CumulVar(index).SetRange(
                        1, num_first
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
        # PARAMETRY SZUKANIA
        # ══════════════════════════════════════════════
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()

        # ⭐ Lepsza strategia startowa
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        )

        search_parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        )
        search_parameters.guided_local_search_lambda_coefficient = 0.1

        if n <= 10:
            time_limit = 3
        elif n <= 20:
            time_limit = 8
        elif n <= 40:
            time_limit = 15
        elif n <= 80:
            time_limit = 30
        else:
            time_limit = 45
        search_parameters.time_limit.seconds = time_limit

        # Logowanie (opcjonalne, do debugowania)
        search_parameters.log_search = False

        # ══════════════════════════════════════════════
        # ROZWIĄZYWANIE
        # ══════════════════════════════════════════════
        solution = routing.SolveWithParameters(search_parameters)

        # ══════════════════════════════════════════════
        # WYNIK
        # ══════════════════════════════════════════════
        if solution:
            all_routes = []

            for vehicle_id in range(data['num_vehicles']):
                index = routing.Start(vehicle_id)
                route = []
                total_distance = 0
                total_time = 0

                while not routing.IsEnd(index):
                    time_var = time_dimension.CumulVar(index)
                    dist_var = distance_dimension.CumulVar(index)
                    node = manager.IndexToNode(index)

                    route.append({
                        "node": node,
                        "arrival_min": solution.Min(time_var),
                        "arrival_max": solution.Max(time_var),
                        "cumul_distance": solution.Min(dist_var),
                        "time_window": list(
                            data['time_windows'][node]
                        ) if node < len(data['time_windows']) else None
                    })
                    index = solution.Value(routing.NextVar(index))

                # Ostatni węzeł (koniec trasy)
                time_var = time_dimension.CumulVar(index)
                dist_var = distance_dimension.CumulVar(index)
                node = manager.IndexToNode(index)

                route.append({
                    "node": node,
                    "arrival_min": solution.Min(time_var),
                    "arrival_max": solution.Max(time_var),
                    "cumul_distance": solution.Min(dist_var),
                    "time_window": list(
                        data['time_windows'][node]
                    ) if node < len(data['time_windows']) else None
                })

                total_distance = solution.Min(
                    distance_dimension.CumulVar(index)
                )
                total_time = (
                    solution.Min(time_dimension.CumulVar(index))
                    - solution.Min(
                        time_dimension.CumulVar(routing.Start(vehicle_id))
                    )
                )

                all_routes.append({
                    "vehicle_id": vehicle_id,
                    "route": route,
                    "total_distance_meters": total_distance,
                    "total_time_seconds": total_time,
                    "total_distance_km": round(total_distance / 1000, 1),
                    "total_time_minutes": round(total_time / 60, 1)
                })

            # Dla kompatybilności wstecznej (1 pojazd)
            return {
                "status": "success",
                "route": all_routes[0]["route"],
                "routes": all_routes,
                "optimize_by": request.optimize_by,
                "computing_time_used": time_limit,
                "objective_value": solution.ObjectiveValue()
            }
        else:
            # ⭐ Diagnostyka dlaczego nie znaleziono
            status = routing.status()
            status_map = {
                0: "ROUTING_NOT_SOLVED",
                1: "ROUTING_SUCCESS",
                2: "ROUTING_FAIL",
                3: "ROUTING_FAIL_TIMEOUT",
                4: "ROUTING_INVALID"
            }
            return {
                "status": "failed",
                "solver_status": status_map.get(status, f"UNKNOWN({status})"),
                "message": (
                    "Nie znaleziono trasy. Możliwe przyczyny: "
                    "1) Okienka czasowe wykluczają się nawzajem, "
                    "2) First/Last nodes tworzą niemożliwą kolejność, "
                    "3) Czas przejazdu przekracza okienka, "
                    "4) Macierze czasu/odległości są niespójne."
                )
            }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "traceback": traceback.format_exc()
        }


@app.get("/health")
def health():
    return {"status": "ok"}
