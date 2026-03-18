from fastapi import FastAPI
from pydantic import BaseModel
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

app = FastAPI()

class RouteRequest(BaseModel):
    num_vehicles: int = 1
    starts: list[int]
    ends: list[int]
    # NOWE POLA: Listy adresów na początek i na koniec (domyślnie puste)
    first_nodes: list[int] = [] 
    last_nodes: list[int] = []
    
    time_matrix: list[list[int]]
    time_windows: list[tuple[int, int]]

@app.post("/solve_tsptw")
def solve_tsptw(request: RouteRequest):
    data = {
        'time_matrix': request.time_matrix,
        'time_windows': request.time_windows,
        'num_vehicles': request.num_vehicles,
        'starts': request.starts,
        'ends': request.ends,
        'first_nodes': request.first_nodes,
        'last_nodes': request.last_nodes
    }

    # 1. Inicjalizacja
    manager = pywrapcp.RoutingIndexManager(
        len(data['time_matrix']), 
        data['num_vehicles'], 
        data['starts'], 
        data['ends']
    )
    routing = pywrapcp.RoutingModel(manager)

    # 2. Definicja czasu przejazdu
    def time_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['time_matrix'][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(time_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # 3. DODANIE OKIENEK CZASOWYCH
    time = 'Time'
    routing.AddDimension(
        transit_callback_index,
        3600,  # Max czas oczekiwania
        86400, # Max czas całej trasy
        False, 
        time)
    time_dimension = routing.GetDimensionOrDie(time)

    for location_idx, time_window in enumerate(data['time_windows']):
        if location_idx in data['starts'] or location_idx in data['ends']:
            continue
        index = manager.NodeToIndex(location_idx)
        time_dimension.CumulVar(index).SetRange(time_window[0], time_window[1])

    # ---------------------------------------------------------------------
    # 4. NOWOŚĆ: WYMUSZANIE ADRESÓW NA POCZĄTKU I NA KOŃCU (Tiers System)
    # ---------------------------------------------------------------------
    def get_node_tier(node):
        if node in data['starts']: return 0
        if node in data['first_nodes']: return 1
        if node in data['last_nodes']: return 3
        if node in data['ends']: return 4
        return 2 # Zwykłe adresy (środek)

    # Fizyczne usuwanie dróg pozwalających na "cofanie się"
    num_nodes = len(data['time_matrix'])
    for from_node in range(num_nodes):
        for to_node in range(num_nodes):
            if from_node == to_node:
                continue
                
            from_tier = get_node_tier(from_node)
            to_tier = get_node_tier(to_node)
            
            # Jeśli próbujemy jechać z wyższego poziomu do niższego (np. z poziomu 2 do 1)
            # zabraniamy tego ruchu (algorytm nigdy nie sprawdzi tej drogi).
            if from_tier > to_tier:
                from_index = manager.NodeToIndex(from_node)
                to_index = manager.NodeToIndex(to_node)
                # Upewniamy się, że nie modyfikujemy punktu końcowego (nie ma następcy)
                if not routing.IsEnd(from_index):
                    routing.NextVar(from_index).RemoveValue(to_index)
    # ---------------------------------------------------------------------

    # 5. Parametry wyszukiwania
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    search_parameters.time_limit.seconds = 25

    # 6. Rozwiązywanie
    solution = routing.SolveWithParameters(search_parameters)

    # 7. Zwracanie wyników
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
        
        return {"status": "success", "route": route}
    else:
        return {"status": "failed", "message": "Nie znaleziono trasy w 25 sekund. Sprawdź, czy okienka czasowe nie wykluczają się nawzajem!"}
