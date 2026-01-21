from flask import Flask, render_template, jsonify, request
import pandas as pd
import numpy as np
import math
from geopy.distance import geodesic
from scipy.spatial import KDTree
from xml.dom import minidom
import os
import re

app = Flask(__name__)


# -----------------------------
# Global Variables For Configuration
# -----------------------------
KML_FILE = "data/ls_centerline_lla.kml"

START_FINISH = (36.586462, -121.756647)  # starting point of the race track

# Laguna Seca Track Segments (Complete)
# SEGMENTS = {
#     "s1": ((36.5839366, -121.7577759), (36.5831308, -121.7577501)),  # S1: Front Straight
#     "s2": ((36.5831308, -121.7577501), (36.5831968, -121.7570164)),  # T1: Turn 1 (Andretti Hairpin)
#     "s3": ((36.5831968, -121.7570164), (36.5846042, -121.7569929)),  # S2: Straight after T1
#     "s4": ((36.5846042, -121.7569929), (36.584936, -121.7562342)),  # T2: Turn 2
#     "s5": ((36.584936, -121.7562342), (36.5845646, -121.7543319)),  # S3: Straight before T3
#     "s6": ((36.5845646, -121.7543319), (36.583913, -121.7537695)),  # T3: Turn 3
#     "s7": ((36.583913, -121.7537695), (36.5809529, -121.7542137)),  # S4: Straight before T4
#     "s8": ((36.5809529, -121.7542137), (36.5799276, -121.753068)),  # T4: Turn 4
#     "s9": ((36.5799276, -121.753068), (36.5802836, -121.7504375)),  # S5: Straight to T5
#     "s10": ((36.5802836, -121.7504375), (36.5809126, -121.749543)),  # T5: Turn 5
#     "s11": ((36.5809126, -121.749543), (36.583667, -121.7492081)),  # S6: Straight to T6
#     "s12": ((36.583667, -121.7492081), (36.5845225, -121.7490881)),  # T6: Turn 6
#     "s13": ((36.5845225, -121.7490881), (36.5849412, -121.7492574)),  # S7: Straight before T7
#     "s14": ((36.5849412, -121.7492574), (36.5853208, -121.7495999)),  # T7: Turn 7
#     "s15": ((36.5853208, -121.7495999), (36.5863669, -121.7496373)),  # S8: Straight to Corkscrew
#     "s16": ((36.5863669, -121.7496373), (36.5869293, -121.7506676)),  # T8: Corkscrew (Turns 8 & 8A)
#     "s17": ((36.5869293, -121.7506676), (36.5865319, -121.7523104)),  # S9: Downhill to Rainey Curve
#     "s18": ((36.5865319, -121.7523104), (36.5868615, -121.7532494)),  # T9: Rainey Curve (Turn 9)
#     "s19": ((36.5868615, -121.7532494), (36.5882794, -121.7541799)),  # S10: Straight to T10
#     "s20": ((36.5882794, -121.7541799), (36.5882868, -121.7549781)),  # T10: Turn 10
#     "s21": ((36.5882868, -121.7549781), (36.5852389, -121.7574097)),  # S11: Final Straight to T11
#     "s22": ((36.5852389, -121.7574097), (36.5839366, -121.7577759)),  # T11: Hairpin onto Main Straight
#     "s23": ((36.58645928791499, -121.7566425689053), (36.58242537195816, -121.7571374083648)),  # S12: Main Straight
# }


SEGMENT_POINTS = [
    (36.5839366, -121.7577759),
    (36.5831308, -121.7577501),
    (36.5831968, -121.7570164),
    (36.5846042, -121.7569929),
    (36.584936, -121.7562342),
    (36.5845646, -121.7543319),
    (36.583913, -121.7537695),
    (36.5809529, -121.7542137),
    (36.5799276, -121.753068),
    (36.5802836, -121.7504375),
    (36.5809126, -121.749543),
    (36.583667, -121.7492081),
    (36.5845225, -121.7490881),
    (36.5849412, -121.7492574),
    (36.5853208, -121.7495999),
    (36.5863669, -121.7496373),
    (36.5869293, -121.7506676),
    (36.5865319, -121.7523104),
    (36.5868615, -121.7532494),
    (36.5882794, -121.7541799),
    (36.5882868, -121.7549781),
    (36.5852389, -121.7574097)
]


SEGMENTS = {
    "s1":  (SEGMENT_POINTS[0], SEGMENT_POINTS[1]),
    "s2":  (SEGMENT_POINTS[1], SEGMENT_POINTS[2]),
    "s3":  (SEGMENT_POINTS[2], SEGMENT_POINTS[3]),
    "s4":  (SEGMENT_POINTS[3], SEGMENT_POINTS[4]),
    "s5":  (SEGMENT_POINTS[4], SEGMENT_POINTS[5]),
    "s6":  (SEGMENT_POINTS[5], SEGMENT_POINTS[6]),
    "s7":  (SEGMENT_POINTS[6], SEGMENT_POINTS[7]),
    "s8":  (SEGMENT_POINTS[7], SEGMENT_POINTS[8]),
    "s9":  (SEGMENT_POINTS[8], SEGMENT_POINTS[9]),
    "s10": (SEGMENT_POINTS[9], SEGMENT_POINTS[10]),
    "s11": (SEGMENT_POINTS[10], SEGMENT_POINTS[11]),
    "s12": (SEGMENT_POINTS[11], SEGMENT_POINTS[12]),
    "s13": (SEGMENT_POINTS[12], SEGMENT_POINTS[13]),
    "s14": (SEGMENT_POINTS[13], SEGMENT_POINTS[14]),
    "s15": (SEGMENT_POINTS[14], SEGMENT_POINTS[15]),
    "s16": (SEGMENT_POINTS[15], SEGMENT_POINTS[16]),
    "s17": (SEGMENT_POINTS[16], SEGMENT_POINTS[17]),
    "s18": (SEGMENT_POINTS[17], SEGMENT_POINTS[18]),
    "s19": (SEGMENT_POINTS[18], SEGMENT_POINTS[19]),
    "s20": (SEGMENT_POINTS[19], SEGMENT_POINTS[20]),
    "s21": (SEGMENT_POINTS[20], SEGMENT_POINTS[21]),
    "s22": (SEGMENT_POINTS[21], SEGMENT_POINTS[0]),
    "s23": ((36.58645928791499, -121.7566425689053), (36.58242537195816, -121.7571374083648))    # not on map
}

BRIDGE_POINTS = [
    (36.5862415, -121.7569703),
    (36.5860538, -121.7566500),
    (36.5845751, -121.7552588),
    (36.5848552, -121.7551896),
    (36.5812926, -121.7542798),
    (36.5812075, -121.7539104),
    (36.5803227, -121.7511641),
    (36.5801531, -121.7510778),
    (36.5858456, -121.7497938),
    (36.5857975, -121.7494675)
]

BRIDGES = {
    "b1": (BRIDGE_POINTS[0], BRIDGE_POINTS[1]),
    "b2": (BRIDGE_POINTS[2], BRIDGE_POINTS[3]),
    "b3": (BRIDGE_POINTS[4], BRIDGE_POINTS[5]),
    "b4": (BRIDGE_POINTS[6], BRIDGE_POINTS[7]),
    "b5": (BRIDGE_POINTS[8], BRIDGE_POINTS[9])
}

# East North Up Constants
A  = 6378137.0           # Earth's radius 
E2 = 6.69437999014e-3    # Earth's eccentricity
F  = 1 / 298.257223563   # Earth's flattening

PLOT_DIR = os.path.join("static", "plots")

CSV_PATH_TRACK = "data/laguna_seca_profile.csv"
CSV_PATH_DATA = "data/combined_uploaded.csv"   

KEYWORD_MAP = {
    "gg_plot": "GG Plot",
    "acceleration": "Acceleration",
    "angular_velocity": "Angular Velocity", 
    "brake": "Brake",   
    "brake_act_force": "Brake Acting Force",
    "brk_pos_cmd": "Brake Position Commanded",
    "brk_pos_fbk": "Brake Position Feedback",    
    "brake_pressure_cmd": "Brake Pressure Commanded",
    "brake_pressure_fdbk": "Brake Pressure Feedback",
    "controller_computation_time": "Controller Computation Time",
    "coolant_temperature": "Coolant Temperature",
    "cross_track_error": "Cross Track Error",
    "current_state_rpy": "Current State Roll Pitch Yaw",
    "driver_traction": "Driver Traction Switch Feedback",
    "engine": "Engine",
    "fuel_pressure": "Fuel Pressure",
    "gear": "Gear",
    "heading_error": "Heading Error",
    "steering_degree": "Steering Degree",
    "linear_acceleration": "Linear Acceleration",
    "magnetic_field": "Magnetic Field",
    "mpc_failed": "MPC Failed",
    "mpc_steering_cmd": "MPC Steering Commanded",
    "slip_ratio": "Slip Ratio",
    "slip_angle": "Slip Angle",
    "tire_pressure_gauge": "Tire Pressure Gauge",
    "tire_temp": "Tire Temperature",
    "torque_wheels" : "Torque Wheels",
    "transmission_pressure": "Transmission Pressure",
    "transmission_temperature": "Transmission Temperature",
    "throttle": "Throttle",
    "vehicle_speed_kmph": "Vehicle Speed kmph",
    "velocity_error": "Velocity Error",
    "velocity_mps": "Velocity mph",
    "wheel_potentiometer": "Wheel Potentiometer",
    "wheel_speed": "Wheel Speed",
    "wheel_strain_gauge": "Wheel Strain Gauge",
    "yawpitchroll": "Yaw Pitch Roll"
}    

UPLOADED_FILES_LIST = []
STAT_FILE_NAME = None

# -----------------------------
# Main Processing Function
# -----------------------------
def process_kml_data():
    # Load and parse KML
    xmldoc = minidom.parse(KML_FILE)
    coordinates = xmldoc.getElementsByTagName('coordinates')
    coord_text = coordinates[0].firstChild.nodeValue.strip()
    coord_lines = coord_text.split()

    coords = []
    for line in coord_lines:
        lon, lat, alt = map(float, line.split(','))
        coords.append((lon, lat, alt))

    df = pd.DataFrame(coords, columns=["lon", "lat", "alt"])
    
    # Compute cumulative distances and assign segments based on provided zones
    df['cum_dist_m'] = cumulative_distance_from_ref(df, x_col="lat", y_col="lon", reference_point=None)
    df = assign_zones_to_segments(df)
    
    # Save processed data
    df.to_csv(CSV_PATH_TRACK, index=False)
    return df


def normalize_filename(name):
    return str(name).replace(" ", "_").replace("/", "_").lower()


def cumulative_distance_from_ref(df, x_col="lat", y_col="lon", reference_point=None):
    # Make reference point as start point if not explicitly mentioned
    if reference_point is None:
        reference_point = (df.loc[0, x_col], df.loc[0, y_col])
            
    # Distance from reference to first point
    first_pt = (df.loc[0, x_col], df.loc[0, y_col])
    distances = [geodesic(reference_point, first_pt).meters]

    # Add pair-wise distances
    for i in range(1, len(df)):
        p_prev = (df.loc[i-1, x_col], df.loc[i-1, y_col])
        p_curr = (df.loc[i,   x_col], df.loc[i,   y_col])
        distances.append(distances[-1] + geodesic(p_prev, p_curr).meters)

    return distances


def assign_zones_to_segments(df):
    df = df.copy()
    df['zone'] = 'default'

    for zone_name, (start_ll, end_ll) in SEGMENTS.items():
        start_idx = df.iloc[((df['lat'] - start_ll[0])**2 + (df['lon'] - start_ll[1])**2).idxmin()].name
        end_idx = df.iloc[((df['lat'] - end_ll[0])**2 + (df['lon'] - end_ll[1])**2).idxmin()].name
        
        print(f"ZONE: {zone_name} -> start_idx={start_idx}, end_idx={end_idx}, total={len(df)}")

        if start_idx <= end_idx:
            filter = (df.index >= start_idx) & (df.index <= end_idx) & (df['zone'] == 'default')
        else:
            filter = ((df.index >= start_idx) | (df.index <= end_idx)) & (df['zone'] == 'default') 
        df.loc[filter, 'zone'] = zone_name       

    return df  


def assign_zones_by_reference(stat_df, reference_df):
    stat_df = stat_df.copy()
    reference_df = reference_df.copy()
    reference_coords = np.radians(reference_df[['lat', 'lon']].to_numpy())
    tree = KDTree(reference_coords)
    stat_coords = np.radians(stat_df[['lat', 'lon']].to_numpy())
    dists, idxs = tree.query(stat_coords, k=1)

    # Assign the zone from the closest reference point
    stat_df['zone'] = reference_df.iloc[idxs]['zone'].values

    return stat_df


def fix_reference_coord(reference_point):
    # Return ECEF-origin and rotation matrix for ENU at reference LLA
    lat = reference_point[0]
    lon = reference_point[1]
    alt = 0  # Assuming altitude is zero (only in 2D)
    
    c_lat = math.cos(math.radians(lat))
    c_lon = math.cos(math.radians(lon))
    s_lat = math.sin(math.radians(lat))
    s_lon = math.sin(math.radians(lon))
    
    N = A / math.sqrt(1.0 - E2 * s_lat**2)
    
    ecef0 = [
        (alt + N) * c_lat * c_lon,
        (alt + N) * c_lat * s_lon,
        (alt + N * (1 - E2)) * s_lat,
    ]
    
    R_ecef2enu = [
        -s_lon, c_lon, 0,
        -s_lat * c_lon, -s_lat * s_lon, c_lat,
        c_lat * c_lon, c_lat * s_lon, s_lat
    ]
    
    return ecef0, R_ecef2enu


def enu_to_lla(enu_pos, ecef0, R_ecef2enu):
    enu_pos.append(0)  # For 2D, we assume Z is zero 
    
    ecef_delta = [
        R_ecef2enu[0] * enu_pos[0] + R_ecef2enu[3] * enu_pos[1] + R_ecef2enu[6] * enu_pos[2],
        R_ecef2enu[1] * enu_pos[0] + R_ecef2enu[4] * enu_pos[1] + R_ecef2enu[7] * enu_pos[2],
        R_ecef2enu[2] * enu_pos[0] + R_ecef2enu[5] * enu_pos[1] + R_ecef2enu[8] * enu_pos[2],
    ]    
    
    ecef = [ecef0[i] + ecef_delta[i] for i in range(3)]
    
    b = A * (1 - F)
    ep2 = (A**2 - b**2) / b**2
    
    p = math.sqrt(ecef[0]**2 + ecef[1]**2)
    theta = math.atan2(ecef[2] * A, p * b)
    
    lon = math.atan2(ecef[1], ecef[0])
    lat = math.atan2(ecef[2] + ep2 * b * math.sin(theta)**3, p - E2 * A * math.cos(theta)**3)
    
    lat = math.degrees(lat) 
    lon = math.degrees(lon)
    
    return [lat, lon]


def convert_stat_file(stat_df):
    # Convert position_x, position_y, position_z (enu) to lat/lon/alt (lla) - not handling position_z (working in 2D only)
    if not all(k in stat_df.columns for k in ["position_x", "position_y"]):
        raise ValueError("Stat file missing 'position_x/y' columns.")

    # Get ECEF origin and rotation matrix
    ecef0, R_ecef2enu = fix_reference_coord(START_FINISH)
    
    lla = [enu_to_lla([x, y], ecef0, R_ecef2enu) for x, y in zip(stat_df["position_x"], stat_df["position_y"])]
    stat_df["lat"] = [p[0] for p in lla]
    stat_df["lon"] = [p[1] for p in lla]   
    
    # Compute cumulative distance and assign zones based on segments
    stat_df['cumulative_distance'] = cumulative_distance_from_ref(stat_df, x_col="lat", y_col="lon", reference_point=None)
    stat_df = assign_zones_by_reference(stat_df, pd.read_csv(CSV_PATH_TRACK))

    # Drop ENU columns and save combined result
    for col in ["position_x", "position_y", "position_z"]:
        if col in stat_df.columns:
            del stat_df[col]  
            
    return stat_df 
    

def ros_time_to_seconds_fast(ros_time_str):
    if isinstance(ros_time_str, str) and 'sec=' in ros_time_str:
        sec = int(ros_time_str.split('sec=')[1].split(',')[0])
        nanosec = int(ros_time_str.split('nanosec=')[1].split(')')[0])
        return sec + nanosec * 1e-9
    elif isinstance(ros_time_str, (int, float)) and ros_time_str > 1e12:
        return ros_time_str / 1e9
    else:
        return ros_time_str


def find_closest_indices(target_timestamps, reference_timestamps):
    sorted_indices = np.argsort(reference_timestamps)
    sorted_timestamps = reference_timestamps[sorted_indices]
    idx = np.searchsorted(sorted_timestamps, target_timestamps, side="left")
    idx = np.clip(idx, 1, len(sorted_timestamps) - 1)
    left_distances = np.abs(sorted_timestamps[idx - 1] - target_timestamps)
    right_distances = np.abs(sorted_timestamps[idx] - target_timestamps)
    closest = np.where(left_distances <= right_distances, idx - 1, idx)
    return sorted_indices[closest]


def detect_laps(df, start_finish=START_FINISH, threshold_m=20, lap_distance_threshold=500, min_lap_distance=3500):
    coords = df[["lat", "lon"]].values
    cum_dists = df["cumulative_distance"].values
    lap_numbers = np.full(len(df), -1, dtype=int)

    lap_num = 1
    lap_start_idx = 0
    lap_start_dist = cum_dists[0]
    lap_numbers[0] = lap_num

    close_to_start = [
        i for i, (lat, lon) in enumerate(coords)
        if geodesic((lat, lon), start_finish).meters <= threshold_m
    ]

    if not close_to_start:
        df["lap"] = lap_numbers
        return df, 1

    first_close_idx = close_to_start[0]

    if cum_dists[first_close_idx] < lap_distance_threshold:
        lap_numbers[:first_close_idx] = 0
        lap_num = 1
        lap_start_idx = first_close_idx
        lap_start_dist = cum_dists[first_close_idx]
    else:
        lap_start_idx = 0
        lap_start_dist = cum_dists[0]

    for i in range(lap_start_idx + 1, len(df)):
        lat, lon = coords[i]
        dist_to_start = geodesic((lat, lon), start_finish).meters
        current_cum_dist = cum_dists[i]

        if dist_to_start <= threshold_m:
            if (current_cum_dist - lap_start_dist) >= min_lap_distance:
                lap_num += 1
                lap_start_idx = i
                lap_start_dist = current_cum_dist

        lap_numbers[i] = lap_num

    df["lap"] = lap_numbers
    return df, lap_num + 1


# -----------------------------
# Flask Routes
# -----------------------------
@app.route('/')
def index():
    df = process_kml_data()

    return render_template(
        'index.html',
        initial_zone="full",
        initial_profile="default",
        lat=df['lat'].tolist(),
        lon=df['lon'].tolist()        
    )        


@app.route('/map')
def show_map():
    return render_template('map.html') 


@app.route("/get_track_data")
def get_track_data():   
    df = pd.read_csv(CSV_PATH_TRACK)

    segments_data = []
    
    # Plot segments
    for zone in df['zone'].unique():
        segment_df = df[df['zone'] == zone]
        points = list(zip(segment_df['lat'], segment_df['lon']))       

        segments_data.append({
            "zone": zone,
            "points": points,
            "color": "#8f4b46" 
        })             
        
    # Prepare bridge line data
    bridges = [
        {"id": bid, "points": [p1, p2]}
        for bid, (p1, p2) in BRIDGES.items()
    ]        
        
    return jsonify({
        "segments": segments_data,
        "start_finish": START_FINISH,
        "segment_points": SEGMENT_POINTS,
        "bridges": bridges
    })


@app.route("/get_zone_labels")
def get_zone_labels():
    csv_req_path = CSV_PATH_DATA if os.path.exists(CSV_PATH_DATA) else CSV_PATH_TRACK
    df = pd.read_csv(csv_req_path)

    zones = sorted(df['zone'].dropna().unique().tolist(), key=lambda z: int(z[1:]) if z[1:].isdigit() else z)
    return jsonify(zones)


@app.route('/segment_data', methods=['POST'])
def segment_data():
    data = request.get_json()
    segment = data.get('zone') 
    start_zone = data.get('start_zone')
    end_zone = data.get('end_zone')     

    print(f"{segment=}, {start_zone=}, {end_zone=}")
    
    plot_df = pd.read_csv(CSV_PATH_DATA) if os.path.exists(CSV_PATH_DATA) else pd.read_csv(CSV_PATH_TRACK)
    # Always use default file to highlight segments
    highlight_df = pd.read_csv(CSV_PATH_TRACK)    
    
    selected_df = None
    zone_label = "full"    

    # Assign segments based on provided zones
    
    # Case 1: Segment Range Given (start and end zone)
    if start_zone and end_zone:
        segment_keys = list(SEGMENTS.keys())    
        start_index = segment_keys.index(start_zone)
        end_index = segment_keys.index(end_zone)

        if start_index <= end_index:
            selected_zones = segment_keys[start_index:end_index + 1]
        else:
            selected_zones = segment_keys[start_index:] + segment_keys[:end_index + 1]

        selected_df = plot_df[plot_df['zone'].isin(selected_zones)].copy() 
        
        # Select segments based on start and end coordinate of selected zones
        start_coord = SEGMENTS[start_zone][0]
        end_coord = SEGMENTS[end_zone][1]

        start_idx = ((highlight_df['lat'] - start_coord[0])**2 + (highlight_df['lon'] - start_coord[1])**2).idxmin()
        end_idx = ((highlight_df['lat'] - end_coord[0])**2 + (highlight_df['lon'] - end_coord[1])**2).idxmin()

        if start_idx <= end_idx:
            highlight_df = highlight_df.iloc[start_idx:end_idx + 1].copy()
        else:
            highlight_df = pd.concat([
                highlight_df.iloc[start_idx:],
                highlight_df.iloc[:end_idx + 1]
            ])     
        zone_label = normalize_filename(f"{start_zone}_to_{end_zone}")
    
    # Case 2: Single Segment Chosen
    elif segment:
        selected_df = plot_df[plot_df['zone'] == segment].copy()
        if segment == "s21":
            start_coord = SEGMENTS[segment][0]
            end_coord = SEGMENTS[segment][1]
            start_idx = ((highlight_df['lat'] - start_coord[0])**2 + (highlight_df['lon'] - start_coord[1])**2).idxmin()
            end_idx = ((highlight_df['lat'] - end_coord[0])**2 + (highlight_df['lon'] - end_coord[1])**2).idxmin()
                    
            highlight_df = pd.concat([
                highlight_df.iloc[start_idx:],
                highlight_df.iloc[:end_idx + 1]
            ])  
        else:      
            highlight_df = highlight_df[highlight_df['zone'] == segment].copy()
        
        zone_label = normalize_filename(segment)
    
    # Case 3: Full track if no zone or range is selected
    if selected_df is None:
        selected_df = plot_df.copy()
        highlight_df = highlight_df.copy()
        zone_label = "full"
        
    # Highlight from default, plots from uploaded
    highlight_points = None if zone_label == "full" else list(zip(highlight_df['lat'], highlight_df['lon']))
    hover_label = f"{zone_label}"      
    
    return jsonify({
        "zone": zone_label,
        "highlight_points": highlight_points,
        "highlight_label": hover_label,
        "lat": selected_df['lat'].tolist(),
        "lon": selected_df['lon'].tolist()
    })


@app.route("/upload_csv", methods=["POST"])
def upload_csv():
    global STAT_FILE_NAME, UPLOADED_FILES_LIST
    
    output_path = os.path.join("data", "combined_uploaded.csv")

    if "files" not in request.files:
        return jsonify({"message": "No files part"}), 400

    files = request.files.getlist("files")
    save_path = os.path.join("data", "uploads")
    os.makedirs(save_path, exist_ok=True)

    other_files = {}
    temp_uploaded = []
    skipped_files = []

    # Also load existing files
    existing_files = {
        os.path.splitext(f)[0]: pd.read_csv(os.path.join(save_path, f))
        for f in os.listdir(save_path)
        if f.endswith(".csv") and os.path.isfile(os.path.join(save_path, f))
    }

    COORD_COLS = ["position_x", "position_y", "position_z", "lat", "lon", "alt", "latitude", "longitude", "altitude"]

    # Handle new incoming files
    for file in files:
        if not file.filename.endswith(".csv"):
            return jsonify({"message": f"Invalid file type: {file.filename}"}), 400

        filepath = os.path.join(save_path, file.filename)
        file.save(filepath)

        try:
            df = pd.read_csv(filepath)
        except Exception as e:
            os.remove(filepath)
            return jsonify({"message": f"Failed to read {file.filename}: {str(e)}"}), 400

        name = os.path.splitext(file.filename)[0]
        if not file.filename.lower().endswith("_stat.csv"):
            df.drop(columns=[c for c in COORD_COLS if c in df.columns], inplace=True)

        other_files[name] = df
        existing_files[name] = df
        temp_uploaded.append(file.filename)

        # Update stat file if it's one
        if file.filename.lower().endswith("_stat.csv") and STAT_FILE_NAME is None:
            STAT_FILE_NAME = file.filename

    if STAT_FILE_NAME is None:
        # Try to detect from already existing files
        for fname in os.listdir(save_path):
            if fname.lower().endswith("_stat.csv"):
                STAT_FILE_NAME = fname
                break

    if STAT_FILE_NAME is None or not os.path.exists(os.path.join(save_path, STAT_FILE_NAME)):
        return jsonify({"message": "Missing required '*_stat.csv' file."}), 400

    stat_name = os.path.splitext(STAT_FILE_NAME)[0]
    # Convert timestamps and sort
    try:
        for name in list(existing_files.keys()):
            df = existing_files[name]
            timestamp_col = None
            
            for candidate in ['stamp_seconds', 'stamp', 'time']:
                if candidate in df.columns:
                    timestamp_col = candidate
                    break

            if not timestamp_col:
                print(f"[SKIPPED] No timestamp column in file '{name}'. Expected one of: stamp_seconds, stamp, time.")
                skipped_files.append(name)
                existing_files.pop(name, None)
                continue

            if timestamp_col in ['stamp', 'time']:
                try:
                    df['stamp_seconds'] = df[timestamp_col].apply(ros_time_to_seconds_fast)
                except Exception as e:
                    print(f"[ERROR] Failed to convert '{timestamp_col}' in file '{name}': {str(e)}")
                    continue
            else:
                df['stamp_seconds'] = df[timestamp_col]

            df.sort_values(by="stamp_seconds", inplace=True, ignore_index=True)
            existing_files[name] = df

        stat_df = convert_stat_file(existing_files[stat_name].copy())
        stat_df, num_laps = detect_laps(stat_df)
        stat_df.to_csv(output_path, index=False)
        existing_files[stat_name] = stat_df


    except Exception as e:
        return jsonify({"message": str(e)}), 400

    valid_uploaded = [f for f in temp_uploaded if os.path.splitext(f)[0] not in skipped_files]
    for fname in valid_uploaded:
        if fname not in UPLOADED_FILES_LIST:
            UPLOADED_FILES_LIST.append(fname)

    if len(existing_files) == 1:
        output_path = os.path.join("data", "combined_uploaded.csv")
        stat_df.to_csv(output_path, index=False)
        return jsonify({
            "message": "Stat file uploaded and processed successfully.",
            "base_file": stat_name,
            "output_file": output_path,
            "skipped_files": skipped_files
        })

    # Multi-file alignment
    
    # Step 1: Compute frequencies
    frequencies = {}
    for name, df in existing_files.items():
        if 'stamp_seconds' not in df.columns:
            return jsonify({"message": f"'stamp_seconds' missing in file: {name}."}), 400
        stamps = df['stamp_seconds'].dropna().values
        if len(stamps) > 1:
            diffs = np.diff(stamps)
            median_diff = np.median(diffs)
            freq = 1 / median_diff if median_diff > 0 else 0
            frequencies[name] = {"frequency": freq}

    if not frequencies:
        return jsonify({"message": "No valid timestamp data found."}), 400
    
    # Step 2: Detect duplicate columns (excluding time columns)
    all_columns = {}
    for name, df in existing_files.items():
        for col in df.columns:
            col = col.lower()
            if col in ['stamp', 'stamp_seconds']:
                continue
            all_columns[col] = all_columns.get(col, 0) + 1

    duplicate_columns = {col for col, count in all_columns.items() if count > 1}
    
    # Step 3: Pick base file and begin alignment
    base_file = max(frequencies, key=lambda x: frequencies[x]["frequency"])
    base_df = existing_files[base_file]
    reference_stamps = base_df['stamp_seconds'].values
    
    # Rename base file columns
    match = re.search(r'rosbag2_\d{4}_\d{2}_\d{2}-\d{2}_\d{2}_\d{2}_(.+)', base_file)
    base_suffix = normalize_filename(match.group(1)) if match else normalize_filename(base_file)
    is_imu_side_base = "imu_side" in base_suffix.lower()    
    
    new_base_names = {}
    for col in base_df.columns:
        base = col.lower()
        if base in ['stamp', 'stamp_seconds']:
            new_base_names[col] = col
        elif is_imu_side_base:
            new_base_names[col] = f"{base}_imu_side"
        elif base in duplicate_columns:
            new_base_names[col] = f"{base}_{base_suffix}"
        else:
            new_base_names[col] = col

    base_df.rename(columns=new_base_names, inplace=True)
    aligned_data = base_df.copy()    
    
    # Step 4: Process and align the rest
    for name, df in existing_files.items():
        if name == base_file:
            continue

        aligned_indices = find_closest_indices(reference_stamps, df['stamp_seconds'].values)
        df_clean = df.drop(columns=[col for col in ['stamp', 'stamp_seconds'] if col in df.columns])
        df_clean = df_clean.reset_index(drop=True).iloc[aligned_indices]

        match = re.search(r'rosbag2_\d{4}_\d{2}_\d{2}-\d{2}_\d{2}_\d{2}_(.+)', name)
        file_suffix = normalize_filename(match.group(1)) if match else normalize_filename(name)
        is_imu_side = "imu_side" in file_suffix.lower()

        new_names = {}
        for col in df_clean.columns:
            base = col.lower()
            if base in ['stamp', 'stamp_seconds']:
                new_names[col] = col
            elif is_imu_side:
                new_names[col] = f"{base}_imu_side"
            elif base in duplicate_columns:
                new_names[col] = f"{base}_{file_suffix}"
            else:
                new_names[col] = col

        df_clean.rename(columns=new_names, inplace=True)
        aligned_data = aligned_data.join(df_clean.reset_index(drop=True))
    
    # Step 5: Final check
    if not all(c in aligned_data.columns for c in ['lat', 'lon']):
        return jsonify({"message": "Coordinate columns (lat/lon) missing after alignment."}), 400

    output_path = os.path.join("data", "combined_uploaded.csv")
    aligned_data.to_csv(output_path, index=False)

    return jsonify({
        "message": f"{len(temp_uploaded)} file(s) uploaded and aligned successfully.",
        "base_file": base_file,
        "output_file": output_path
    })


@app.route("/reset_uploads", methods=["POST"])
def reset_uploads():
    global UPLOADED_FILES_LIST, STAT_FILE_NAME
    save_path = os.path.join("data", "uploads")

    # Delete files in uploads dir
    if os.path.exists(save_path):
        for filename in os.listdir(save_path):
            fpath = os.path.join(save_path, filename)
            if os.path.isfile(fpath):
                os.remove(fpath)

    # Reset globals
    UPLOADED_FILES_LIST = []
    STAT_FILE_NAME = None

    # Clear output if exists
    output_path = os.path.join("data", "combined_uploaded.csv")
    if os.path.exists(output_path):
        os.remove(output_path)

    return jsonify({"message": "Upload storage has been reset."})



@app.route("/get_combined_columns", methods=["GET"])
def get_combined_columns():
    path = os.path.join("data", "combined_uploaded.csv")
    if not os.path.exists(path):
        return jsonify([])

    df = pd.read_csv(path)
    cols_lower = [c.lower() for c in df.columns]

    return jsonify([
        {"label": label, "value": kw}
        for kw, label in KEYWORD_MAP.items()
        if any(kw in c for c in cols_lower) or kw == "gg_plot"
    ])


@app.route("/get_column_data")
def get_column_data():
    kw_raw = request.args.get("keyword")
    kw_list = [k.strip().lower() for k in kw_raw.split(",")] if kw_raw else []
    segment = request.args.get("zone")
    start_zone = request.args.get("start_zone")
    end_zone = request.args.get("end_zone")   
    lap = request.args.get("lap")
    
    # Use uploaded data if available
    if os.path.exists(CSV_PATH_DATA):
        path = CSV_PATH_DATA
    elif os.path.exists(CSV_PATH_TRACK):
        path = CSV_PATH_TRACK
    else:
        return jsonify({"error": "no data file available"}), 400
    
    df = pd.read_csv(path)
    
    # Filter by selected laps
    if lap:
        try:
            lap = int(lap)
            if "lap" in df.columns:
                df = df[df["lap"] == lap]
            else:
                return jsonify({"error": "Lap column not found"}), 400
        except ValueError:
            return jsonify({"error": f"Invalid lap value: {lap}"}), 400    
    
    # Filter by selected zones
    if start_zone and end_zone:
        segment_keys = list(SEGMENTS.keys())
        try:
            i1, i2 = segment_keys.index(start_zone), segment_keys.index(end_zone)
        except ValueError:
            return jsonify({"error": f"Invalid zones: {start_zone}, {end_zone}"}), 400
        
        if i1 <= i2:
            selected_zones = segment_keys[i1:i2 + 1]
        else:
            selected_zones = segment_keys[i1:] + segment_keys[:i2 + 1]
        
        df = df[df['zone'].isin(selected_zones)]

    elif segment and segment != "full":
        df = df[df['zone'] == segment]
        
    # Check if any data remains
    if df.empty:
        return jsonify({"error": "No data available for the selected segment(s)."}), 400 
    
    # Return base data if no keywords are selected
    if not kw_list:
        if path == CSV_PATH_DATA and "cumulative_distance" not in df.columns:
            return jsonify({"error": "no cumulative_distance"}), 400

        return jsonify({
            "x": df["cumulative_distance"].tolist() if "cumulative_distance" in df.columns else [],
            "lat": df["lat"].tolist() if "lat" in df.columns else [],
            "lon": df["lon"].tolist() if "lon" in df.columns else [],
            "cols": {}
        })
    
    cols = {}
    
    for kw in kw_list:
        # Special handling for G-G Plot
        if kw == "gg_plot":
            if "target_lateral_acceleration" in df.columns and "target_long_acceleration" in df.columns:
                return jsonify({
                    "x": (df["target_lateral_acceleration"] / 9.80665).tolist(),
                    "y": (df["target_long_acceleration"] / 9.80665).tolist(),
                    "lat": df["lat"].tolist() if "lat" in df.columns else [],
                    "lon": df["lon"].tolist() if "lon" in df.columns else []
                })
            else:
                return jsonify({"error": "Required G-G plot columns not found."}), 400

        # If keyword is provided, find matches and return plots
        matches = [c for c in df.columns if kw in c.lower()]
        for c in matches:
            c_lower = c.lower()
            # Plot velocity in mph
            if "velocity" in c_lower and "mps" in c_lower:
                cols[c.replace("_mps", "_mph")] = (df[c] * 2.23694).tolist()
            elif "velocity_error" in c_lower:
                cols[c + "_mph"] = (df[c] * 2.23694).tolist()
            # Plot acceleration in g
            elif "acceleration" in c_lower and "angular" not in c_lower:
                cols[c + "_g"] = (df[c] / 9.80665).tolist()
            else:
                cols[c] = df[c].tolist()

    return jsonify({
        "x": df["cumulative_distance"].tolist() if "cumulative_distance" in df.columns else [],
        "lat": df["lat"].tolist() if "lat" in df.columns else [],
        "lon": df["lon"].tolist() if "lon" in df.columns else [],
        "cols": cols
    })      
    

@app.route("/get_uploaded_files")
def get_uploaded_files():
    return jsonify(UPLOADED_FILES_LIST)


@app.route("/get_available_laps")
def get_available_laps():
    if not os.path.exists(CSV_PATH_DATA):
        return jsonify([])

    df = pd.read_csv(CSV_PATH_DATA)
    if "lap" not in df.columns:
        return jsonify([])

    laps = sorted(df["lap"].dropna().unique().astype(int).tolist())
    return jsonify(laps)


@app.route("/get_uploaded_trajectory")
def get_uploaded_trajectory():
    if not os.path.exists(CSV_PATH_DATA):
        return jsonify({"lat": [], "lon": []})

    df = pd.read_csv(CSV_PATH_DATA)
    return jsonify({
        "lat": df["lat"].tolist() if "lat" in df.columns else [],
        "lon": df["lon"].tolist() if "lon" in df.columns else []
    })


if __name__ == '__main__':
    app.run(debug=True)