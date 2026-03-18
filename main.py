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
    time_matrix: list[list[int]]
    time_windows: list[tuple[int, int]]

@app.post("/solve_tsptw")
def solve_tsptw(request: RouteRequest):
    # Dodaliśmy "łapacz błędów" - teraz API nigdy nie rzuci błędem 500
    try:
        data = {
            'time_matrix': request.time_matrix,
            'time_windows': request.time_windows,
            'num_vehicles': request.num_vehicles,
            'starts': request.starts,
            'ends': request.ends,
            'first_nodes': request.first_nodes,
            'last_nodes': request.last_nodes
        }

        manager = pywrapcp.RoutingIndexManager(
            len(data['time_matrix']), 
            data['num_vehicles'], 
            data['starts'], 
            data['ends']
        )
        routing = pywrapcp.RoutingModel(manager)

        def time_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return data['time_matrix'][from_node][to_node]

        transit_callback_index = routing.RegisterTransitCallback(time_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        time = 'Time'
        routing.AddDimension(
            transit_callback_index,
            3600,  
            86400, 
            False, 
            time)
        time_dimension = routing.GetDimensionOrDie(time)

        for location_idx, time_window in enumerate(data['time_windows']):
            if location_idx in data['starts'] or location_idx in data['ends']:
                continue
            index = manager.NodeToIndex(location_idx)
            time_dimension.CumulVar(index).SetRange(time_window[0], time_window[1])

        # ---------------------------------------------------------------------
        # NAPRAWIONY BŁĄD! Usunięto 5-ty argument. Funkcja przyjmuje tylko 4.
        # ---------------------------------------------------------------------
        routing.AddConstantDimension(
            1, 
            len(data['time_matrix']) + 1, 
            True, 
            'StepCounter'
        )
        step_dimension = routing.GetDimensionOrDie('StepCounter')

        num_first = len(data['first_nodes'])
        num_last = len(data['last_nodes'])
        
        middle_nodes = [i for i in range(len(data['time_matrix'])) if i not in data['starts'] and i not in data['ends'] and i not in data['first_nodes'] and i not in data['last_nodes']]
        num_mid = len(middle_nodes)

        for node in range(len(data['time_matrix'])):
            if node in data['starts'] or node in data['ends']:
                continue
                
            index = manager.NodeToIndex(node)
            
            if node in data['first_nodes']:
                step_dimension.CumulVar(index).SetRange(1, num_first)
            elif node in data['last_nodes']:
                step_dimension.CumulVar(index).SetRange(num_first + num_mid + 1, num_first + num_mid + num_last)
            else:
                step_dimension.CumulVar(index).SetRange(num_first + 1, num_first + num_mid)
        # ---------------------------------------------------------------------

        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        search_parameters.time_limit.seconds = 25

        solution = routing.SolveWithParameters(search_parameters)

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
            return {"status": "failed", "message": "Nie znaleziono trasy. Sprawdź okienka czasowe i logikę priorytetów."}
            
    # Jeśli Python napotka błąd, wyświetli nam go jako odpowiedź JSON!
    except Exception as e:
        return {
            "status": "error", 
            "message": "Błąd Pythona!", 
            "details": str(e),
            "traceback": traceback.format_exc()
        }
