# analysis_dashboard

A flask-based interactive dashboard to visualize data across segments of the  **Laguna Seca Race Track**.

---
## Setup

1. Clone the repository.
2. Install requirements.

```bash
pip install -r requirements.txt
```

3. Run the flask app.
   
```bash
python app.py
```

4. Open browser to:
http://localhost:5000

---
## Structure
```
.
├── app.py                     # Main Flask application
├── data/
│   ├── ls_centerline_lla.kml  # KML track layout file
│   └── uploads/               # Uploaded CSV files
├── static/
│   ├── js/
│   │   └── index.js           # Frontend JS logic
│   └── css/
│       └── styles.css         # Custom styles 
├── templates/
│   ├── index.html             # Main UI
│   └── map.html               # Embedded Leaflet map
├── README.md                  # This file

```

---
## Features

### Initial Load Behavior
- On first load, no plot is shown.


### Segment Selection Options

1. **Single Segment**:
   - Click on any segment directly on the map.
   - That segment gets highlighted in yellow.

2. **Segment Range**:
   - Use the dropdowns labeled **"Start Segment"** and **"End Segment"** to define a zone.
   - Click **"Apply Segment Range"** to apply changes.
   - That segment range gets highlighted in yellow.

   - **Clear** button resets zone selection and displays full track data again.


### Upload Files

- Upload one or more CSV files to visualize the data.

- Important: The *_stat.csv file must be uploaded first, as it contains the positional data used to align other files.

- After uploading, the Select Column(s) dropdown will automatically populate with available data groups. Select one or more columns to plot.

- All selected data will be plotted with shared X-axes (cumulative distance).

- On hover, the corresponding map location will be highlighted in red.

- The title below Segment Information dynamically updates to show the Selected `Zone`.

- ***Reset*** button removes all the files uploaded, allowing to start fresh with new uploads.


### Map Interface 
- **Start/Finish point** is marked in **red**
- **Segment boundary points** are marked in **blue**
- **Bridges** are shown in **black**
- **Segment numbers** and **Bridge numbers** are displayed interactively on hover
- Hover tooltip on segments also shows Zone name.


---
## Customizing

### 1. Add New Columns to Plot
To configure which  data columns appear in the "Select Column(s)" dropdown, edit the `KEYWORD_MAP` dictionary in app.py.

Identify the base keyword in the CSV columns (e.g., tire_pressure) and then add a new entry to the dictionary.

```python
"tire_pressure": "Tire Pressure",
```
Once uploaded, all columns that contain the keyword (e.g., tire_pressure_fl, tire_pressure_bl) will be grouped together and available for plotting.