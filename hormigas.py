import pandas as pd
import numpy as np
import random
import folium
import matplotlib.pyplot as plt
import requests
import time
from math import radians, sin, cos, sqrt, atan2
import json

# ——————————————————————————————————————————————
# 0) Parámetros y carga de datos
file_path = "Listado.xlsx"
df = (
    pd.read_excel(file_path)
      .rename(columns={
          'Positivo':             'lat',
          'Negativo':             'lon',
          'Peso de la caja(kg)':  'peso'
      })
)
coords = list(zip(df['lat'], df['lon']))
n = len(coords)  # incluye el depósito en índice 0

# ——————————————————————————————————————————————
# Parámetros ACO para CVRP - Optimizados para rutas reales
alpha       = 1.0     # influencia de feromona
beta        = 3.0     # mayor influencia de heurística para rutas reales
rho         = 0.2     # tasa de evaporación más conservadora
num_ants    = 30      # reducido para compensar el tiempo de cálculo de rutas reales
num_iters   = 100     # reducido por el costo computacional
capacity    = 500     # capacidad de cada vehículo (500 kg)
tau0        = 0.1     # feromona inicial
elitist     = True    # usar estrategia elitista
local_search = True   # aplicar búsqueda local

# ——————————————————————————————————————————————
# 1) Funciones para obtener distancias reales por carretera

def haversine_fallback(a, b):
    """Función de respaldo usando distancia haversine"""
    lat1, lon1 = a; lat2, lon2 = b
    R = 6371.0  # km
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    h = sin(dlat/2)**2 + cos(radians(lat1))*cos(radians(lat2))*sin(dlon/2)**2
    return R * 2 * atan2(sqrt(h), sqrt(1-h))

def get_osrm_route_data(coord1, coord2, max_retries=3):
    """
    Obtiene tanto la distancia como la geometría de la ruta real usando OSRM
    Formato: lon,lat (OSRM usa lon,lat no lat,lon)
    Retorna: (distance_km, route_geometry)
    """
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    
    # Solicitar geometría completa de la ruta
    url = f"http://router.project-osrm.org/route/v1/driving/{lon1},{lat1};{lon2},{lat2}?overview=full&geometries=geojson&alternatives=false&steps=false"
    
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data['code'] == 'Ok' and 'routes' in data and len(data['routes']) > 0:
                    route = data['routes'][0]
                    # OSRM devuelve distancia en metros, convertir a km
                    distance_km = route['distance'] / 1000.0
                    
                    # Obtener geometría de la ruta (coordenadas que siguen las carreteras)
                    geometry = route['geometry']['coordinates']
                    # Convertir de [lon, lat] a [lat, lon] para Folium
                    route_coords = [[coord[1], coord[0]] for coord in geometry]
                    
                    return distance_km, route_coords
            
            # Si hay error, esperar antes del siguiente intento
            time.sleep(0.5)
            
        except Exception as e:
            print(f"Error en OSRM (intento {attempt+1}): {e}")
            time.sleep(1)
    
    # Si falla OSRM, usar distancia haversine y línea recta como respaldo
    distance = haversine_fallback(coord1, coord2) * 1.3
    straight_line = [list(coord1), list(coord2)]
    return distance, straight_line

def get_osrm_distance(coord1, coord2, max_retries=3):
    """
    Función auxiliar que solo retorna la distancia (para compatibilidad)
    """
    distance, _ = get_osrm_route_data(coord1, coord2, max_retries)
    return distance

def build_distance_matrix_with_cache(coords, cache_file="distance_cache.json"):
    """
    Construye matriz de distancias usando rutas reales con caché para evitar recálculos
    También guarda las geometrías de las rutas para visualización
    """
    n = len(coords)
    
    # Intentar cargar caché existente
    cache = {}
    geometry_cache = {}
    try:
        with open(cache_file, 'r') as f:
            data = json.load(f)
            cache = data.get('distances', {})
            geometry_cache = data.get('geometries', {})
        print(f"Caché cargado: {len(cache)} distancias, {len(geometry_cache)} geometrías")
    except FileNotFoundError:
        print("No se encontró caché previo, calculando distancias...")
    
    # Inicializar matriz
    D = np.zeros((n, n))
    total_calculations = n * (n - 1) // 2
    calculated = 0
    
    print(f"Calculando matriz de distancias reales ({total_calculations} cálculos)...")
    
    for i in range(n):
        for j in range(i+1, n):
            cache_key = f"{coords[i][0]:.6f},{coords[i][1]:.6f}-{coords[j][0]:.6f},{coords[j][1]:.6f}"
            reverse_key = f"{coords[j][0]:.6f},{coords[j][1]:.6f}-{coords[i][0]:.6f},{coords[i][1]:.6f}"
            
            # Buscar en caché
            if cache_key in cache:
                distance = cache[cache_key]
            elif reverse_key in cache:
                distance = cache[reverse_key]
            else:
                # Calcular nueva distancia y geometría
                distance, geometry = get_osrm_route_data(coords[i], coords[j])
                cache[cache_key] = distance
                geometry_cache[cache_key] = geometry
                time.sleep(0.1)  # Pausa para no sobrecargar la API
            
            D[i,j] = distance
            D[j,i] = distance  # Matriz simétrica
            
            calculated += 1
            if calculated % 10 == 0:
                print(f"Progreso: {calculated}/{total_calculations} ({calculated/total_calculations*100:.1f}%)")
    
    # Guardar caché actualizado
    try:
        cache_data = {
            'distances': cache,
            'geometries': geometry_cache
        }
        with open(cache_file, 'w') as f:
            json.dump(cache_data, f)
        print(f"Caché guardado con {len(cache)} distancias y {len(geometry_cache)} geometrías")
    except Exception as e:
        print(f"Error guardando caché: {e}")
    
    return D, geometry_cache

# Construir matriz de distancias reales
print("Construyendo matriz de distancias por carreteras reales...")
D, route_geometries = build_distance_matrix_with_cache(coords)

# ——————————————————————————————————————————————
# 2) Inicializar feromonas y heurística mejorada para rutas reales
def nearest_neighbor_heuristic():
    """Heurística del vecino más cercano para inicialización"""
    current = 0
    unvisited = set(range(1, n))
    total_dist = 0
    
    while unvisited:
        next_node = min(unvisited, key=lambda j: D[current, j])
        total_dist += D[current, next_node]
        current = next_node
        unvisited.remove(current)
    
    return total_dist

# Inicializar con heurística mejorada
nn_distance = nearest_neighbor_heuristic()
tau0 = 1.0 / (n * nn_distance)
pheromone = np.full((n, n), tau0)

# Matriz heurística considerando tanto distancia como capacidad
with np.errstate(divide='ignore', invalid='ignore'):
    # Heurística básica de distancia
    distance_heuristic = 1.0 / D
    distance_heuristic[np.isinf(distance_heuristic)] = 0
    distance_heuristic[np.isnan(distance_heuristic)] = 0
    
    # Heurística de capacidad (favorece nodos con menos peso)
    capacity_heuristic = np.ones((n, n))
    for i in range(n):
        for j in range(1, n):  # Excluir depósito
            if i != j:
                weight_factor = 1.0 - (df.loc[j, 'peso'] / capacity)
                capacity_heuristic[i, j] = weight_factor
    
    # Combinar heurísticas
    heuristic = distance_heuristic * (1 + 0.3 * capacity_heuristic)

# ——————————————————————————————————————————————
# 3) Clase Hormiga mejorada para rutas reales
class Hormiga:
    def __init__(self, start=0):
        self.start = start
        self.reset()

    def reset(self):
        self.subrutas = []  # Lista de rutas [0, ..., punto_final] (sin regreso)
        self.visitados = {self.start}
        self.carga_total = 0

    def total_distance(self):
        """Calcula distancia total considerando que no hay regreso al depósito"""
        d = 0
        for ruta in self.subrutas:
            for k in range(len(ruta)-1):
                d += D[ruta[k], ruta[k+1]]
        return d

    def apply_2opt_no_depot_return(self, ruta):
        """2-opt optimizado para rutas sin regreso al depósito"""
        if len(ruta) <= 3:
            return ruta

        mejor_ruta = ruta.copy()
        mejor_dist = sum(D[mejor_ruta[i], mejor_ruta[i+1]] for i in range(len(mejor_ruta)-1))

        # Solo optimizar entre los puntos de entrega (no tocar inicio/fin)
        for i in range(1, len(ruta)-2):
            for j in range(i+1, len(ruta)-1):
                new_ruta = mejor_ruta.copy()
                new_ruta[i:j+1] = reversed(new_ruta[i:j+1])

                new_dist = sum(D[new_ruta[k], new_ruta[k+1]] for k in range(len(new_ruta)-1))

                if new_dist < mejor_dist:
                    mejor_dist = new_dist
                    mejor_ruta = new_ruta.copy()

        return mejor_ruta

# ——————————————————————————————————————————————
# 4) Función de selección mejorada para rutas reales
def elegir_siguiente_ciudad(current, carga, visitados, iteration_factor=1.0):
    """
    Selección de siguiente ciudad considerando múltiples factores
    """
    candidatos = [
        j for j in range(1, n)
        if j not in visitados
           and carga + df.loc[j,'peso'] <= capacity
    ]

    if not candidatos:
        return None

    # Estrategia adaptativa según progreso del algoritmo
    exploration_rate = max(0.05, 0.3 * (1 - iteration_factor))
    
    if random.random() < exploration_rate:
        # Exploración: selección con sesgo hacia mejores opciones
        weights = []
        for j in candidatos:
            # Combinar distancia, peso y clustering
            dist_score = 1.0 / (1 + D[current, j])
            weight_score = 1.0 - (df.loc[j, 'peso'] / capacity)
            
            # Bonus por clustering (favorece puntos cercanos entre sí)
            cluster_score = 0
            for visited in visitados:
                if visited != 0:  # No considerar depósito
                    cluster_score += 1.0 / (1 + D[j, visited])
            cluster_score = min(cluster_score / len(visitados), 1.0)
            
            final_score = dist_score * 0.5 + weight_score * 0.3 + cluster_score * 0.2
            weights.append(final_score)
        
        # Selección proporcional a los pesos
        total_weight = sum(weights)
        if total_weight > 0:
            probs = [w/total_weight for w in weights]
            return np.random.choice(candidatos, p=probs)
        else:
            return random.choice(candidatos)
    else:
        # Explotación: regla ACO estándar mejorada
        scores = np.zeros(len(candidatos))
        for idx, j in enumerate(candidatos):
            tau = pheromone[current, j]
            eta = heuristic[current, j]
            scores[idx] = (tau ** alpha) * (eta ** beta)

        if sum(scores) == 0:
            return random.choice(candidatos)

        probs = scores / sum(scores)
        return np.random.choice(candidatos, p=probs)

# ——————————————————————————————————————————————
# 5) Construcción de rutas sin regreso al depósito
def construir_ruta_sin_regreso(start, visitados, carga_actual=0, iteration_factor=1.0):
    """
    Construye una ruta que empieza en el depósito pero no regresa
    """
    ruta = [start]
    current = start
    carga = carga_actual

    while True:
        nxt = elegir_siguiente_ciudad(current, carga, visitados, iteration_factor)
        if nxt is None:
            break

        ruta.append(nxt)
        carga += df.loc[nxt, 'peso']
        visitados.add(nxt)
        current = nxt

    return ruta, visitados

# ——————————————————————————————————————————————
# 6) Algoritmo ACO principal optimizado
def solve_cvrp_real_roads():
    global pheromone  # Declare pheromone as global

    best_global = None
    best_dist = float('inf')
    history = []
    estancamiento = 0
    max_estancamiento = 15

    print("Iniciando CVRP con rutas reales por carretera...")
    print(f"Parámetros: {num_ants} hormigas, {num_iters} iteraciones")
    print(f"Capacidad: {capacity}kg, Restricción: SIN regreso al depósito")

    for iteration in range(num_iters):
        hormigas = [Hormiga() for _ in range(num_ants)]
        best_iter_dist = float('inf')
        best_iter_sol = None
        iteration_factor = iteration / num_iters  # Factor de progreso

        # Cada hormiga construye su solución
        for ant_idx, h in enumerate(hormigas):
            h.reset()

            # Construir todas las rutas necesarias
            while len(h.visitados) < n:
                nueva_ruta, h.visitados = construir_ruta_sin_regreso(
                    0, h.visitados, 0, iteration_factor
                )

                # Aplicar búsqueda local si la ruta tiene suficientes puntos
                if local_search and len(nueva_ruta) > 3:
                    nueva_ruta = h.apply_2opt_no_depot_return(nueva_ruta)

                if len(nueva_ruta) > 1:  # Solo agregar rutas válidas
                    h.subrutas.append(nueva_ruta)

            # Evaluar solución
            dist_total = h.total_distance()

            if dist_total < best_iter_dist:
                best_iter_dist = dist_total
                best_iter_sol = [r.copy() for r in h.subrutas]

        # Actualización de feromonas
        pheromone *= (1 - rho)

        # Depósito por todas las hormigas
        for h in hormigas:
            dist_h = h.total_distance()
            if dist_h > 0:
                deposit = 1.0 / dist_h
                for ruta in h.subrutas:
                    for k in range(len(ruta)-1):
                        i, j = ruta[k], ruta[k+1]
                        pheromone[i,j] += deposit
                        pheromone[j,i] += deposit

        # Refuerzo elitista
        if elitist and best_global is not None:
            elite_deposit = 2.0 / best_dist
            for ruta in best_global:
                for k in range(len(ruta)-1):
                    i, j = ruta[k], ruta[k+1]
                    pheromone[i,j] += elite_deposit
                    pheromone[j,i] += elite_deposit

        # Actualizar mejor global
        if best_iter_dist < best_dist:
            best_dist = best_iter_dist
            best_global = best_iter_sol
            estancamiento = 0
        else:
            estancamiento += 1

        # Reinicio por estancamiento
        if estancamiento >= max_estancamiento:
            print(f"Reiniciando feromonas en iteración {iteration+1}")
            pheromone = np.full((n, n), tau0)
            estancamiento = 0

        history.append(best_dist)

        # Progreso
        if (iteration+1) % 10 == 0 or iteration < 3 or iteration == num_iters-1:
            print(f"Iter {iteration+1}/{num_iters} — "
                  f"Mejor iter: {best_iter_dist:.2f}km | Global: {best_dist:.2f}km")

    return best_global, best_dist, history

# ——————————————————————————————————————————————
# 7) Ejecutar optimización
best_solution, best_distance, convergence_history = solve_cvrp_real_roads()

print(f"\n{'='*60}")
print(f"SOLUCIÓN ÓPTIMA ENCONTRADA")
print(f"{'='*60}")
print(f"Distancia total: {best_distance:.2f} km")
print(f"Número de vehículos necesarios: {len(best_solution)}")
print(f"Ahorro vs. rutas individuales: ~{((sum(D[0,i] + D[i,0] for i in range(1,n)) - best_distance) / sum(D[0,i] + D[i,0] for i in range(1,n)) * 100):.1f}%")

# ——————————————————————————————————————————————
# 8) Detalles de cada ruta
print(f"\n{'='*60}")
print("DETALLES DE RUTAS POR VEHÍCULO")
print(f"{'='*60}")

total_carga = 0
for i, ruta in enumerate(best_solution):
    # Calcular métricas de la ruta
    dist_ruta = sum(D[ruta[j], ruta[j+1]] for j in range(len(ruta)-1))
    carga_ruta = sum(df.loc[nodo, 'peso'] for nodo in ruta if nodo != 0)
    total_carga += carga_ruta
    
    print(f"\nVEHÍCULO {i+1}:")
    print(f"  Ruta: {ruta}")
    print(f"  Distancia: {dist_ruta:.2f} km")
    print(f"  Carga: {carga_ruta} kg / {capacity} kg ({carga_ruta/capacity*100:.1f}%)")
    print(f"  Entregas: {len(ruta)-1} puntos")
    print(f"  Eficiencia: {carga_ruta/dist_ruta:.2f} kg/km")
    
    # Secuencia de destinos
    nombres = []
    for nodo in ruta:
        if nodo == 0:
            nombres.append("DEPÓSITO")
        else:
            nombres.append(df.iloc[nodo]['Nombre'])
    
    print(f"  Secuencia: {' → '.join(nombres)}")

print(f"\nRESUMEN FINAL:")
print(f"  Carga total transportada: {total_carga} kg")
print(f"  Utilización promedio: {total_carga/(len(best_solution)*capacity)*100:.1f}%")

# ——————————————————————————————————————————————
# 9) Visualización de convergencia
plt.figure(figsize=(12,6))
plt.plot(convergence_history, marker='o', markersize=3, linestyle='-', linewidth=2)
plt.title("Convergencia CVRP con Rutas Reales por Carretera", fontsize=14, fontweight='bold')
plt.xlabel("Iteración", fontsize=12)
plt.ylabel("Distancia Total (km)", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

def get_route_geometry(coord1, coord2, geometry_cache):
    """
    Obtiene la geometría de ruta desde el caché o calcula una nueva
    """
    cache_key = f"{coord1[0]:.6f},{coord1[1]:.6f}-{coord2[0]:.6f},{coord2[1]:.6f}"
    reverse_key = f"{coord2[0]:.6f},{coord2[1]:.6f}-{coord1[0]:.6f},{coord1[1]:.6f}"
    
    if cache_key in geometry_cache:
        return geometry_cache[cache_key]
    elif reverse_key in geometry_cache:
        # Invertir la geometría si está en sentido contrario
        return list(reversed(geometry_cache[reverse_key]))
    else:
        # Si no está en caché, obtener nueva geometría
        _, geometry = get_osrm_route_data(coord1, coord2)
        geometry_cache[cache_key] = geometry
        return geometry

# ——————————————————————————————————————————————
# 10) Mapa interactivo con rutas reales por carreteras
def create_interactive_map_with_real_roads(solution, coords, df, geometry_cache):
    """Crea mapa interactivo con las rutas que siguen carreteras reales"""
    depot_coord = coords[0]
    mapa = folium.Map(location=depot_coord, zoom_start=11)

    # Colores para las rutas
    colores = ['red', 'blue', 'green', 'purple', 'orange', 'darkred',
               'darkblue', 'darkgreen', 'cadetblue', 'darkpurple', 'pink',
               'lightblue', 'lightgreen', 'gray', 'black', 'lightred']

    # Marcador del depósito
    folium.Marker(
        location=depot_coord,
        icon=folium.Icon(color='red', icon='home', prefix='fa'),
        popup=folium.Popup(f'<b>DEPÓSITO CENTRAL</b><br>Punto de partida de todos los vehículos', 
                          max_width=200)
    ).add_to(mapa)

    # Dibujar cada ruta siguiendo carreteras reales
    print("Generando rutas por carreteras reales en el mapa...")
    
    for i, ruta in enumerate(solution):
        color = colores[i % len(colores)]
        
        # Calcular métricas de la ruta
        dist_ruta = sum(D[ruta[j], ruta[j+1]] for j in range(len(ruta)-1))
        carga_ruta = sum(df.loc[nodo, 'peso'] for nodo in ruta if nodo != 0)
        
        print(f"Procesando Vehículo {i+1}: {len(ruta)-1} segmentos...")
        
        # Para cada segmento de la ruta, obtener la geometría real
        for j in range(len(ruta)-1):
            punto_inicio = ruta[j]
            punto_fin = ruta[j+1]
            
            coord_inicio = coords[punto_inicio]
            coord_fin = coords[punto_fin]
            
            # Obtener geometría del segmento por carretera
            try:
                geometria_segmento = get_route_geometry(coord_inicio, coord_fin, geometry_cache)
                
                # Dibujar el segmento de carretera
                folium.PolyLine(
                    geometria_segmento,
                    weight=5,
                    opacity=0.8,
                    color=color,
                    popup=folium.Popup(
                        f'<b>Vehículo {i+1} - Segmento {j+1}</b><br>'
                        f'De: {df.iloc[punto_inicio]["Nombre"] if punto_inicio != 0 else "DEPÓSITO"}<br>'
                        f'A: {df.iloc[punto_fin]["Nombre"] if punto_fin != 0 else "DEPÓSITO"}<br>'
                        f'Distancia: {D[punto_inicio, punto_fin]:.2f} km',
                        max_width=250
                    )
                ).add_to(mapa)
                
            except Exception as e:
                print(f"Error obteniendo geometría para segmento {j+1} del vehículo {i+1}: {e}")
                # Fallback: línea recta si no se puede obtener la geometría
                folium.PolyLine(
                    [coord_inicio, coord_fin],
                    weight=3,
                    opacity=0.6,
                    color=color,
                    dashArray='5, 5'  # Línea punteada para indicar que es fallback
                ).add_to(mapa)

        # Marcadores para puntos de entrega
        for j, punto in enumerate(ruta[1:], 1):  # Empezar desde el segundo punto
            folium.CircleMarker(
                location=coords[punto],
                radius=8,
                color=color,
                fill=True,
                fillColor=color,
                fillOpacity=0.7,
                popup=folium.Popup(
                    f'<b>{df.iloc[punto]["Nombre"]}</b><br>'
                    f'Vehículo: {i+1}<br>'
                    f'Orden: {j}<br>'
                    f'Peso: {df.iloc[punto]["peso"]} kg<br>'
                    f'Coordenadas: {coords[punto][0]:.4f}, {coords[punto][1]:.4f}',
                    max_width=200
                )
            ).add_to(mapa)
            
            # Etiqueta con número de orden
            folium.Marker(
                location=coords[punto],
                icon=folium.DivIcon(
                    html=f'<div style="font-size: 12px; color: white; font-weight: bold; '
                         f'text-align: center; background-color: {color}; '
                         f'border-radius: 50%; width: 20px; height: 20px; '
                         f'line-height: 20px;">{j}</div>',
                    icon_size=(20, 20),
                    icon_anchor=(10, 10)
                )
            ).add_to(mapa)
        
        # Información adicional de la ruta completa
        folium.Marker(
            location=coords[ruta[-1]],  # Último punto de la ruta
            icon=folium.Icon(color=color, icon='info-sign'),
            popup=folium.Popup(
                f'<b>FIN RUTA VEHÍCULO {i+1}</b><br>'
                f'Distancia total: {dist_ruta:.2f} km<br>'
                f'Carga total: {carga_ruta} kg<br>'
                f'Entregas: {len(ruta)-1} puntos<br>'
                f'Eficiencia: {carga_ruta/dist_ruta:.2f} kg/km<br>'
                f'Utilización: {carga_ruta/capacity*100:.1f}%',
                max_width=200
            )
        ).add_to(mapa)

    # Agregar leyenda mejorada
    legend_html = f'''
    <div style="position: fixed; 
                top: 10px; right: 10px; width: 280px; height: auto; 
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:14px; padding: 10px">
    <h4><i class="fa fa-truck"></i> Rutas CVRP por Carreteras</h4>
    <p><b>Total de vehículos:</b> {len(solution)}</p>
    <p><b>Distancia total:</b> {best_distance:.2f} km</p>
    <p><b>Capacidad por vehículo:</b> {capacity} kg</p>
    <hr>
    <p><i class="fa fa-home" style="color: red;"></i> Depósito (Inicio)</p>
    <p><i class="fa fa-circle"></i> Puntos de entrega</p>
    <p><i class="fa fa-info-circle"></i> Fin de ruta</p>
    <p><strong>Líneas sólidas:</strong> Rutas por carretera</p>
    <p><strong>Líneas punteadas:</strong> Estimaciones</p>
    <p>Los números indican orden de visita</p>
    </div>
    '''
    mapa.get_root().html.add_child(folium.Element(legend_html))

    return mapa

# Crear y mostrar el mapa con rutas reales
print("\nGenerando mapa interactivo con rutas por carreteras reales...")
mapa_final = create_interactive_map_with_real_roads(best_solution, coords, df, route_geometries)

# Guardar el mapa
mapa_final.save("mapa_cvrp_rutas_reales.html")
print("Mapa guardado como 'mapa_cvrp_rutas_reales.html'")
print("Las rutas ahora siguen las carreteras reales en lugar de líneas rectas")

# Mostrar el mapa
mapa_final